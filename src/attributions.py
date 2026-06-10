"""
attributions.py  —  v1
Token-level saliency via input embedding gradients for the
Claim–Evidence Consistency Checker.

Uses Captum's IntegratedGradients to compute how much each token
contributed to the model's predicted label.

Official framing:
    "Token-level saliency via input embedding gradients"
    Scores reflect gradient flow through word embeddings only.
    DeBERTa-v3's disentangled attention interactions are not captured —
    appropriate and sufficient for token highlight overlays in a demo UI.

Pipeline:
    (claim, evidence, label_id)
        → tokenise → input_ids
        → look up word embeddings for each token       [shape: seq_len x 768]
        → build PAD baseline embeddings                [shape: seq_len x 768]
        → IntegratedGradients: interpolate baseline → real input (N_STEPS)
        → sum gradients across embedding dim           [shape: seq_len]
        → normalise to [-1, 1]
        → return [(token_str, score), ...]

DeBERTa-v3 specifics handled here:
    - No token_type_ids (excluded in infer.py, not needed here either)
    - Word embeddings at: model.deberta.embeddings.word_embeddings
    - PAD token id: tokenizer.pad_token_id (= 1 for DeBERTa-v3, not 0)

Dependency on infer.py:
    model and tokenizer are imported directly — never loaded twice.
"""

# ── Standard library ──────────────────────────────────────────────────────────
from typing import List, Tuple

# ── ML ────────────────────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F

# ── Captum — PyTorch XAI library ──────────────────────────────────────────────
from captum.attr import IntegratedGradients

# ── Internal — reuse model and tokenizer already loaded by infer.py ───────────
from infer import model, tokenizer, DEVICE, MAX_LENGTH

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

N_STEPS = 25    # Integrated Gradients interpolation steps.
                # Empirically validate against 50/100 steps on short sequences
                # (< 50 tokens) before final submission — see convergence note
                # in carry-over summary. Original IG paper recommends 20–300
                # depending on model complexity; 25 is the demo-quality floor.

# DeBERTa-v3 word embedding layer — confirmed path for this architecture.
# Do NOT use model.bert.embeddings (that's BERT) or model.roberta.embeddings.
EMBEDDING_LAYER = model.deberta.embeddings.word_embeddings

# PAD token id — used as the baseline (zero-information input).
# DeBERTa-v3: pad_token_id = 1. Resolved from tokenizer, not hardcoded.
PAD_ID = tokenizer.pad_token_id

# Tokens excluded from UI display and top-N ranking.
# These are structurally meaningful to the model but uninterpretable
# as content signals for a human reader:
#   - Special tokens: [CLS], [SEP], [PAD]
#   - Punctuation: all standard marks that SentencePiece tokenises separately
# Imported by Phase 4 UI to filter the attribution overlay.
DISPLAY_SKIP_TOKENS = {
    "[CLS]", "[SEP]", "[PAD]",
    tokenizer.pad_token,
    ".", ",", "!", "?", ":", ";",
    "''", "'s", "'", '"',
    "/", "\\", "-", "—", "–", "--", "(", ")",
    "[", "]", "{", "}", "@", "#", "%",
    "&", "*", "^", "~",
}

# ══════════════════════════════════════════════════════════════════════════════
# FORWARD FUNCTION  — called by Captum at each interpolation step
# ══════════════════════════════════════════════════════════════════════════════

def forward_from_embeddings(
    input_embeds:   torch.Tensor,   # shape: (1, seq_len, 768) — interpolated embeddings
    attention_mask: torch.Tensor,   # shape: (1, seq_len)      — real attention mask
) -> torch.Tensor:
    """
    Forward pass using pre-computed embeddings instead of token IDs.

    Captum interpolates between baseline and real embeddings at each step
    and calls this function with the interpolated embeddings. We pass them
    directly to DeBERTa's encoder, bypassing the normal embedding lookup.

    Why inputs_embeds instead of input_ids:
        Gradients cannot flow through integer token IDs. By passing float
        embedding vectors directly, gradients flow back through the
        embedding space — one score per token per embedding dimension.

    Returns
    -------
    torch.Tensor of shape (1, 3) — raw logits for [SUPPORT, NEI, CONTRADICT]
    Captum uses the logit at target=label_id as the scalar to differentiate.
    """
    output = model(
        inputs_embeds = input_embeds,
        attention_mask = attention_mask,
        # token_type_ids intentionally omitted — DeBERTa-v3 does not use it
    )
    return output.logits   # shape: (1, 3)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ATTRIBUTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def attribute(
    claim:    str,
    evidence: str,
    label_id: int,
) -> List[Tuple[str, float]]:
    """
    Compute token-level saliency via input embedding gradients.

    Parameters
    ----------
    claim    : str — the claim text (premise)
    evidence : str — the winning sentence only, not full passage
    label_id : int — predicted label (0=SUPPORT, 1=NEI, 2=CONTRADICT)
                     gradients are computed w.r.t. this class logit

    Returns
    -------
    List of (token_str, score) tuples, one per token including [CLS]/[SEP].
    Scores normalised to [-1, 1]:
        +1.0  → token strongly pushes toward predicted label
        -1.0  → token strongly pushes away from predicted label
         0.0  → token had no influence on prediction
    """

    # ── 1. Tokenise ───────────────────────────────────────────────────────────
    encoding = tokenizer(
        claim,
        evidence,
        truncation    = True,
        max_length    = MAX_LENGTH,
        return_tensors = "pt",
    )
    encoding.pop("token_type_ids", None)   # DeBERTa-v3 does not use this
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    input_ids      = encoding["input_ids"]       # shape: (1, seq_len)
    attention_mask = encoding["attention_mask"]  # shape: (1, seq_len)

    # ── 2. Look up real input embeddings ──────────────────────────────────────
    # Detach from any existing graph — we build a fresh one for IG
    input_embeds = EMBEDDING_LAYER(input_ids).detach()   # shape: (1, seq_len, 768)

    # ── 3. Build PAD baseline embeddings ──────────────────────────────────────
    # Baseline = sequence of PAD token embeddings — carries zero information.
    # Same shape as input_embeds so Captum can interpolate between them.
    pad_ids      = torch.full_like(input_ids, PAD_ID)    # shape: (1, seq_len)
    baseline_embeds = EMBEDDING_LAYER(pad_ids).detach()  # shape: (1, seq_len, 768)

    # ── 4. Run Integrated Gradients ───────────────────────────────────────────
    ig = IntegratedGradients(forward_from_embeddings)

    attributions = ig.attribute(
        inputs          = input_embeds,
        baselines       = baseline_embeds,
        additional_forward_args = (attention_mask,),
        target          = label_id,   # differentiate w.r.t. this class logit
        n_steps         = N_STEPS,
        return_convergence_delta = False,
    )
    # attributions shape: (1, seq_len, 768)

    # ── 5. Aggregate across embedding dimension ────────────────────────────────
    # Sum across the 768 embedding dims → one scalar per token
    token_scores = attributions.sum(dim=-1).squeeze(0)   # shape: (seq_len,)

    # ── 6. Decode tokens ──────────────────────────────────────────────────────
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    # ── 7. Normalise to [-1, 1] using content tokens only ────────────────────
    # Problem: [SEP] and punctuation dominate the raw scores, stealing the
    # normalisation anchor and making content tokens look artificially weak.
    # Fix: compute max absolute value over content tokens only.
    # Special tokens and punctuation will exceed [-1, 1] after this —
    # that is intentional and acceptable since they are filtered at display time.
    content_indices = [
        i for i, t in enumerate(tokens)
        if t not in DISPLAY_SKIP_TOKENS
    ]

    if content_indices:
        content_scores = token_scores[content_indices]
        max_abs = content_scores.abs().max().item()
    else:
        # Degenerate case: no content tokens at all — fall back to global max
        max_abs = token_scores.abs().max().item()

    if max_abs > 1e-8:
        token_scores = token_scores / max_abs

    # ── 8. Return (token, score) pairs ───────────────────────────────────────
    return [
        (token, round(float(score.item()), 4))
        for token, score in zip(tokens, token_scores)
    ]

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT  — smoke test
# Run with:  python attributions.py
# (infer.py does not need to be running — model is loaded via import)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # Real SciFact pair — same as infer.py smoke test for consistency
    # Expected: SUPPORT, so label_id = 0
    _claim    = "Aspirin inhibits the production of thromboxane A2."
    _evidence = "Aspirin irreversibly inhibits cyclooxygenase, thereby reducing " \
                "thromboxane A2 synthesis in platelets."
    _label_id = 0   # SUPPORT

    print("\n" + "═" * 60)
    print("SMOKE TEST — attributions.py")
    print("═" * 60)
    print(f"  Claim    : {_claim}")
    print(f"  Evidence : {_evidence}")
    print(f"  Label    : SUPPORT (id={_label_id})")
    print(f"  N_STEPS  : {N_STEPS}")
    print("\n  Running attribute() …")

    _results = attribute(_claim, _evidence, _label_id)

    print(f"\n  {'Token':<25} {'Score':>8}")
    print(f"  {'─'*25} {'─'*8}")
    for token, score in _results:
        # Visual bar to make scores easy to read
        bar_len = int(abs(score) * 20)
        bar     = ("█" * bar_len) if score >= 0 else ("░" * bar_len)
        sign    = "+" if score >= 0 else "-"
        print(f"  {token:<25} {sign}{abs(score):.4f}  {bar}")

    print(f"\n  Total tokens: {len(_results)}")

    # Sanity checks
    scores = [s for _, s in _results]
    content_scores = [s for t, s in _results if t not in DISPLAY_SKIP_TOKENS]
    assert len(_results) > 0,         "No attributions returned"
    # Note: special tokens and punctuation intentionally exceed [-1, 1] —
    # normalisation anchor is content tokens only. Check content tokens only.
    assert max(abs(s) for s in content_scores) <= 1.0 + 1e-6,         "Content token scores exceed [-1, 1] range — normalisation failed"
    assert any(abs(s) > 0.01 for s in content_scores),         "All content scores near zero — gradient may not be flowing"

    # Top 5 most influential tokens — filtered using DISPLAY_SKIP_TOKENS
    filtered = [
        (t, s) for t, s in _results
        if t not in DISPLAY_SKIP_TOKENS
    ]
    top5 = sorted(filtered, key=lambda x: abs(x[1]), reverse=True)[:5]
    print(f"\n  Top 5 most influential tokens:")
    for rank, (token, score) in enumerate(top5, 1):
        direction = "→ predicted label" if score > 0 else "← away from label"
        print(f"    {rank}. {token:<20} {score:+.4f}  {direction}")

    print("\n  ✓ All assertions passed — attributions working correctly")
    print("═" * 60)