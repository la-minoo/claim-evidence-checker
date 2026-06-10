"""
infer.py  —  v2
Core inference module for the Claim–Evidence Consistency Checker.

Loads the fine-tuned DeBERTa-v3-base SciFact model once at startup and
exposes two interfaces:
  1. predict(claim, evidence) → dict          ← used by attributions.py and retrieval.py
  2. POST /predict via FastAPI                ← used by Streamlit demo and TS frontend

Changes in v2 (Phase 4b):
  - CORS middleware added (allows localhost:3000 and localhost:5173)
  - POST /attribute endpoint added (wraps attributions.attribute())
  - POST /chat endpoint added (multi-turn Phi3 follow-up via Ollama)

Tokeniser input convention (mirrors MultiNLI training format exactly):
  premise   = claim text
  hypothesis = evidence text

Attribution note:
  Returns raw input_ids and tokens so attributions.py can run
  token-level saliency via input embedding gradients without re-tokenising.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
from typing import Optional, List, Dict, Any

# ── ML / tokenisation ─────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Sentence splitting ────────────────────────────────────────────────────────
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# ── API framework ─────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── HTTP (for Ollama proxy) ───────────────────────────────────────────────────
import requests as http_requests

# ── Ollama chat helper ────────────────────────────────────────────────────────
# Imported after sys.path is set — llama_chat.py lives in src\ alongside infer.py
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))
from llama_chat import ollama_chat

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit MODEL_PATH to match your local folder
# ══════════════════════════════════════════════════════════════════════════════

MODEL_PATH = os.getenv(
    "SCIFACT_MODEL_PATH",
    r"C:\Users\1\buss305-project\claim-evidence-checker\models\scifact_ckpt\buss305-scifact-bestmodel"
)

MAX_LENGTH = 256          # must match training
DEVICE     = torch.device("cpu")   # local machine is CPU-only

# Label scheme locked in during SciFact fine-tuning
ID2LABEL = {0: "SUPPORT", 1: "NOT_ENOUGH_INFO", 2: "CONTRADICT"}
LABEL2ID = {"SUPPORT": 0, "NOT_ENOUGH_INFO": 1, "CONTRADICT": 2}

# Ollama config
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"
OLLAMA_TIMEOUT = 180   # seconds — Phi3 cold load ~40s on CPU

# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  — runs once at import time, not per request
# ══════════════════════════════════════════════════════════════════════════════

print(f"[infer] Loading tokenizer from {MODEL_PATH} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(f"[infer] Loading model …")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()   # disables dropout — required for deterministic inference and attribution

print(f"[infer] Model ready on {DEVICE}.")
print(f"[infer] Labels: {model.config.id2label}")

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP + CORS
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Claim-Evidence Checker", version="2.0")

# CORS — allow the Vite dev server (5173) and any other local frontend (3000)
# Must be added BEFORE routes are declared
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://claimai-sepia.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════════════
# CORE PREDICT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(claim: str, evidence: str) -> dict:
    """
    Run NLI inference on a (claim, evidence) pair.

    Tokeniser convention (mirrors MultiNLI training):
        premise    = claim
        hypothesis = evidence

    Returns
    -------
    {
        "label":         str,    e.g. "SUPPORT"
        "label_id":      int,    e.g. 0
        "confidence":    float,  softmax prob of predicted class
        "probabilities": list,   [p_SUPPORT, p_NEI, p_CONTRADICT]
        "tokens":        list,   token strings (for attribution overlay)
        "input_ids":     tensor  (for attribution — avoids re-tokenising)
    }
    """
    # Tokenise — premise=claim, hypothesis=evidence, exactly as in training
    encoding = tokenizer(
        claim,
        evidence,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # DeBERTa-v3 does not use token_type_ids — remove if tokeniser added them
    encoding.pop("token_type_ids", None)

    # Move to device (CPU in our case)
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    # Forward pass — no gradient needed for plain inference
    with torch.no_grad():
        logits = model(**encoding).logits          # shape: (1, 3)

    # Softmax → probabilities
    probs    = F.softmax(logits, dim=-1).squeeze(0)   # shape: (3,)
    label_id = int(probs.argmax().item())

    # Decode tokens for the attribution overlay
    input_ids = encoding["input_ids"].squeeze(0)
    tokens    = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    return {
        "label":         ID2LABEL[label_id],
        "label_id":      label_id,
        "confidence":    round(float(probs[label_id].item()), 4),
        "probabilities": [round(float(p.item()), 4) for p in probs],
        "tokens":        tokens,
        "input_ids":     input_ids,    # kept as tensor — attributions.py needs it
    }

# ══════════════════════════════════════════════════════════════════════════════
# SENTENCE-LEVEL SCORING
# ══════════════════════════════════════════════════════════════════════════════

WINNER_THRESHOLD  = 0.65   # minimum confidence for a non-NEI sentence to be
                           # trusted as winner. Below this, fall back to NEI.
SUPPORT_THRESHOLD = 0.70   # minimum confidence for supporting sentences.
MAX_SUPPORTING    = 2      # max additional sentences shown

def split_and_score(claim: str, text: str) -> dict:
    """
    Split a long evidence text into sentences, score each against the claim,
    and return the winner plus up to MAX_SUPPORTING supporting sentences.

    Returns
    -------
    {
        "winner": {
            "sentence":       str,
            "label":          str,
            "confidence":     float,
            "sentence_index": int
        },
        "supporting": [ ... ],
        "all_scores":  [ ... ]
    }
    """
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return {"winner": None, "supporting": [], "all_scores": []}

    scored = []
    for idx, sentence in enumerate(sentences):
        result = predict(claim, sentence)
        scored.append({
            "sentence":       sentence,
            "label":          result["label"],
            "confidence":     result["confidence"],
            "sentence_index": idx,
        })

    all_scores = sorted(scored, key=lambda x: x["confidence"], reverse=True)

    # Three-way winner selection logic
    confident_non_nei = [
        s for s in all_scores
        if s["label"] != "NOT_ENOUGH_INFO"
        and s["confidence"] >= WINNER_THRESHOLD
    ]
    nei_sentences = [s for s in all_scores if s["label"] == "NOT_ENOUGH_INFO"]
    fallback       = nei_sentences[0] if nei_sentences else all_scores[0]
    winner         = confident_non_nei[0] if confident_non_nei else fallback

    supporting = [
        s for s in all_scores
        if s["label"]       == winner["label"]
        and s["confidence"] >= SUPPORT_THRESHOLD
        and s["sentence"]   != winner["sentence"]
    ][:MAX_SUPPORTING]

    return {
        "winner":     winner,
        "supporting": supporting,
        "all_scores":  all_scores,
    }

# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA HELPER  — shared by /chat endpoint
# ══════════════════════════════════════════════════════════════════════════════

def _call_ollama(prompt: str) -> str:
    """
    Send a single prompt to Phi3 via Ollama and return the response string.
    Raises HTTPException on connection failure so callers get a clean 503.
    """
    try:
        resp = http_requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except http_requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama error: {exc}")

# ══════════════════════════════════════════════════════════════════════════════
# PYDANTIC SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    claim:    str
    evidence: str

class PredictResponse(BaseModel):
    label:         str
    label_id:      int
    confidence:    float
    probabilities: list[float]
    tokens:        list[str]

class SentenceScore(BaseModel):
    sentence:       str
    label:          str
    confidence:     float
    sentence_index: int

class AnalyzeRequest(BaseModel):
    claim:    str
    evidence: str

class AnalyzeResponse(BaseModel):
    winner:                SentenceScore
    supporting:            List[SentenceScore]
    all_scores:            List[SentenceScore]
    attribution_available: bool

class AttributeRequest(BaseModel):
    claim:    str
    evidence: str
    label_id: int

class AttributeToken(BaseModel):
    token: str
    score: float

class ChatMessage(BaseModel):
    role:    str   # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    # Full analysis context — injected as system prompt for the first turn
    claim:           str
    evidence:        str
    label:           str
    confidence:      float
    probabilities:   List[float]          # [p_support, p_nei, p_contradict]
    top5_tokens:     List[List[Any]]      # [[token, score], ...]
    model_reasoning: str                  # already-generated model reasoning panel text
    science_explanation: str             # already-generated scientific explanation text
    # Conversation history — all turns so far, NOT including the new user message
    history:         List[ChatMessage]
    # The new user message
    message:         str

class ExplainReasoningRequest(BaseModel):
    claim:         str
    evidence:      str          # winner sentence only
    label:         str
    confidence:    float
    probabilities: List[float]  # [p_support, p_nei, p_contradict]
    top5_tokens:   List[List[Any]]  # [[token, score], ...]

class ExplainScienceRequest(BaseModel):
    claim:    str
    evidence: str   # winner sentence only
    label:    str

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Quick check that the server is up and model is loaded."""
    return {"status": "ok", "model": MODEL_PATH, "device": str(DEVICE)}


@app.post("/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    """
    POST /predict
    Body: { "claim": "...", "evidence": "..." }
    Returns label, confidence, per-class probabilities, and token list.
    Single sentence only — use /analyze for multi-sentence evidence.
    """
    if not req.claim.strip():
        raise HTTPException(status_code=422, detail="claim cannot be empty")
    if not req.evidence.strip():
        raise HTTPException(status_code=422, detail="evidence cannot be empty")

    result = predict(req.claim, req.evidence)
    return PredictResponse(
        label         = result["label"],
        label_id      = result["label_id"],
        confidence    = result["confidence"],
        probabilities = result["probabilities"],
        tokens        = result["tokens"],
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def api_analyze(req: AnalyzeRequest):
    """
    POST /analyze
    Body: { "claim": "...", "evidence": "..." }
    Splits evidence into sentences, scores each, returns winner + supporting.
    attribution_available=False when winner is NEI — frontend skips attribution.
    """
    if not req.claim.strip():
        raise HTTPException(status_code=422, detail="claim cannot be empty")
    if not req.evidence.strip():
        raise HTTPException(status_code=422, detail="evidence cannot be empty")

    result = split_and_score(req.claim, req.evidence)

    if result["winner"] is None:
        raise HTTPException(status_code=422, detail="evidence text produced no scoreable sentences")

    attribution_available = result["winner"]["label"] != "NOT_ENOUGH_INFO"

    return AnalyzeResponse(
        winner                = SentenceScore(**result["winner"]),
        supporting            = [SentenceScore(**s) for s in result["supporting"]],
        all_scores            = [SentenceScore(**s) for s in result["all_scores"]],
        attribution_available = attribution_available,
    )


@app.post("/attribute", response_model=List[AttributeToken])
def api_attribute(req: AttributeRequest):
    """
    POST /attribute
    Body: { "claim": "...", "evidence": "...", "label_id": 0 }
    Returns token-level saliency scores via Integrated Gradients.

    Always call with the WINNER sentence only, not the full evidence paragraph.
    label_id must match the winner's predicted label (0=SUPPORT, 1=NEI, 2=CONTRADICT).

    Frontend should skip this call when attribution_available=False
    (i.e. winner label is NOT_ENOUGH_INFO).
    """
    if not req.claim.strip():
        raise HTTPException(status_code=422, detail="claim cannot be empty")
    if not req.evidence.strip():
        raise HTTPException(status_code=422, detail="evidence cannot be empty")
    if req.label_id not in (0, 1, 2):
        raise HTTPException(status_code=422, detail="label_id must be 0, 1, or 2")

    # Import here to avoid circular import at module load time.
    # attributions.py imports model/tokenizer from this file, so we defer
    # until after this module is fully initialised.
    from attributions import attribute

    token_scores = attribute(req.claim, req.evidence, req.label_id)

    return [
        AttributeToken(token=token, score=score)
        for token, score in token_scores
    ]


@app.post("/explain/reasoning")
def api_explain_reasoning(req: ExplainReasoningRequest):
    """
    POST /explain/reasoning
    Body: { "claim", "evidence", "label", "confidence", "probabilities", "top5_tokens" }
    Returns: { "response": "..." }

    Asks Phi3 to narrate how the model derived its verdict, grounded in the
    attribution scores and class probabilities. Mirrors the 🧠 panel in demo.py.
    One-shot — no conversation history.
    """
    if not req.claim.strip():
        raise HTTPException(status_code=422, detail="claim cannot be empty")
    if not req.evidence.strip():
        raise HTTPException(status_code=422, detail="evidence cannot be empty")

    prob_line = (
        f"SUPPORT={req.probabilities[0]:.1%}  "
        f"NOT_ENOUGH_INFO={req.probabilities[1]:.1%}  "
        f"CONTRADICT={req.probabilities[2]:.1%}"
    )
    token_lines = "\n".join(
        f"  {i+1}. {str(t)!r:20s} {float(s):+.4f}  "
        f"({'toward predicted label' if float(s) > 0 else 'away from predicted label'})"
        for i, (t, s) in enumerate(req.top5_tokens[:10])
    )

    prompt = (
        f"You are explaining how an NLI model reached its verdict. Use ONLY the token scores listed below — "
        f"do not speculate about or mention any tokens not in this list.\n\n"
        f"Claim: {req.claim}\n"
        f"Evidence: {req.evidence}\n"
        f"Predicted label: {req.label} (confidence {req.confidence:.1%})\n"
        f"Class probabilities: {prob_line}\n"
        f"Attribution tokens (gradient-based saliency, all tokens you may reference):\n{token_lines}\n\n"
        f"In 2-3 sentences, explain what the model focused on to reach this verdict. "
        f"Reference only tokens from the list above with their exact scores. "
        f"Never mention tokens that are not in this list."
    )

    return {"response": ollama_chat([
        {"role": "system", "content": "You explain NLI model verdicts in 2-3 sentences using only the token scores provided. Never speculate about tokens not in the list. No outside knowledge. Stop after your explanation. Do not generate additional examples, claims, or verdicts."},
        {"role": "user",   "content": prompt},
    ])}


@app.post("/explain/science")
def api_explain_science(req: ExplainScienceRequest):
    """
    POST /explain/science
    Body: { "claim", "evidence", "label" }
    Returns: { "response": "..." }

    Asks Phi3 to explain the verdict from a scientific/content perspective.
    Mirrors the 🔬 panel in demo.py. One-shot — no conversation history.
    """
    if not req.claim.strip():
        raise HTTPException(status_code=422, detail="claim cannot be empty")
    if not req.evidence.strip():
        raise HTTPException(status_code=422, detail="evidence cannot be empty")

    label_phrase = req.label.lower().replace("_", " ")
    prompt = (
        f"Claim: {req.claim}\n"
        f"Evidence: {req.evidence}\n"
        f"Verdict: {req.label}\n\n"
        f"In exactly one sentence, explain why the evidence {label_phrase}s the claim scientifically. "
        f"Be specific. Do not add anything after the sentence."
    )

    return {"response": ollama_chat([
        {"role": "system", "content": "You write exactly one sentence explaining a scientific verdict. One sentence only. Stop after the full stop. Do not generate any additional content."},
        {"role": "user",   "content": prompt},
    ])}


@app.post("/chat")
def api_chat(req: ChatRequest):
    """
    POST /chat
    Multi-turn conversational follow-up about an analysis result.

    The full analysis context is injected into every prompt so Phi3 can
    answer grounded follow-up questions (e.g. "why did the model focus on
    'not'?", "what does thromboxane A2 mean?", "is this peer-reviewed?").

    Conversation history is passed by the client on every call (stateless
    server — no session storage). The server assembles the full prompt,
    calls Phi3, and returns only the new assistant response.

    Request body:
    {
        "claim":               "...",
        "evidence":            "...",
        "label":               "SUPPORT",
        "confidence":          0.94,
        "probabilities":       [0.94, 0.03, 0.03],
        "top5_tokens":         [["aspirin", 0.88], ...],
        "model_reasoning":     "...",
        "science_explanation": "...",
        "history":             [{"role": "user", "content": "..."}, ...],
        "message":             "why did the model focus on 'not'?"
    }

    Returns:
    { "response": "..." }
    """
    # Build a context block that grounds every follow-up answer
    prob_line = (
        f"SUPPORT={req.probabilities[0]:.1%}  "
        f"NOT_ENOUGH_INFO={req.probabilities[1]:.1%}  "
        f"CONTRADICT={req.probabilities[2]:.1%}"
    )
    token_lines = "\n".join(
        f"  {i+1}. {t!r:20s} {s:+.4f}"
        for i, (t, s) in enumerate(req.top5_tokens[:5])
    )

    # Truncate reasoning/science to first 300 chars to keep prompt within Phi3's context
    reasoning_short = req.model_reasoning[:300].strip()  + ("…" if len(req.model_reasoning)  > 300 else "")
    science_short   = req.science_explanation[:300].strip() + ("…" if len(req.science_explanation) > 300 else "")

    system_content = (
        f"You are a concise assistant explaining a claim-verification result. "
        f"Answer in 2 sentences maximum. Never write more than 2 sentences. "
        f"Be direct and brief. Stop after your answer.\n\n"
        f"Claim: {req.claim}\n"
        f"Evidence: {req.evidence}\n"
        f"Verdict: {req.label} ({req.confidence:.1%})\n"
        f"Probabilities: {prob_line}\n"
        f"Top tokens: {token_lines}\n"
        f"Model reasoning: {reasoning_short}\n"
        f"Science note: {science_short}"
    )

    # Build messages array — system context + history + new user message
    messages = [{"role": "system", "content": system_content}]
    for msg in req.history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": req.message})

    return {"response": ollama_chat(messages)}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT  — smoke test then start server
# Run with:  python src\infer.py
# Or purely as API:  uvicorn infer:app --reload
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":

    # ── Smoke test ────────────────────────────────────────────────────────────
    _claim    = "Aspirin inhibits the production of thromboxane A2."
    _evidence = ("Aspirin irreversibly inhibits cyclooxygenase, thereby reducing "
                 "thromboxane A2 synthesis in platelets.")

    print("\n" + "═" * 60)
    print("SMOKE TEST")
    print("═" * 60)
    print(f"  Claim    : {_claim}")
    print(f"  Evidence : {_evidence}")
    print("  Running predict() …")

    _result = predict(_claim, _evidence)

    print(f"\n  Label        : {_result['label']}")
    print(f"  Confidence   : {_result['confidence']:.4f}")
    print(f"  Probabilities: SUPPORT={_result['probabilities'][0]:.4f}  "
          f"NEI={_result['probabilities'][1]:.4f}  "
          f"CONTRADICT={_result['probabilities'][2]:.4f}")
    print(f"  Token count  : {len(_result['tokens'])}")
    print(f"  input_ids    : shape {_result['input_ids'].shape}")

    assert _result["label"] == "SUPPORT", \
        f"Expected SUPPORT, got {_result['label']} — check model path or label mapping"
    assert 0.0 < _result["confidence"] <= 1.0, "Confidence out of range"
    assert len(_result["tokens"]) == _result["input_ids"].shape[0], \
        "Token count does not match input_ids length"

    print("\n  ✓ All assertions passed — model working correctly")
    print("═" * 60)

    # ── Start server ──────────────────────────────────────────────────────────
    print("\n[infer] Starting FastAPI server on http://localhost:8000")
    print("[infer] Docs: http://localhost:8000/docs")
    print("[infer] CORS enabled for localhost:3000 and localhost:5173")
    print("[infer] Endpoints: /health /predict /analyze /attribute /explain/reasoning /explain/science /chat /retrieve")
    print("[infer] Press Ctrl+C to stop\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)