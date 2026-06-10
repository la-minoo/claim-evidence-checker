"""
demo.py  —  Streamlit demo for the Claim–Evidence Consistency Checker
Run with:  streamlit run app/demo.py

Modes:
    Mode 1 — Claim Review   : user provides claim + evidence text
    Mode 2 — Claim Check    : user provides claim only → RAG via Semantic Scholar

Pipeline (both modes after evidence is known):
    claim + evidence sentence
        → split_and_score()   → winner sentence + all_scores
        → predict()           → class probabilities
        → attribute()         → token attribution scores
        → Phi3 via Ollama     → natural language explanation
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sys
import os
from pathlib import Path

# ── Add src\ to path so imports work from app\ ────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # claim-evidence-checker\
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

# ── Streamlit ─────────────────────────────────────────────────────────────────
import streamlit as st

# ── Internal ──────────────────────────────────────────────────────────────────
from infer        import predict, split_and_score, ID2LABEL
from attributions import attribute, DISPLAY_SKIP_TOKENS
from retrieval    import retrieve

# ── Ollama ────────────────────────────────────────────────────────────────────
import requests as http_requests

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

LABEL_COLORS = {
    "SUPPORT":          "#2ecc71",   # green
    "CONTRADICT":       "#e74c3c",   # red
    "NOT_ENOUGH_INFO":  "#f39c12",   # amber
}

LABEL_ICONS = {
    "SUPPORT":         "✅",
    "CONTRADICT":      "❌",
    "NOT_ENOUGH_INFO": "⚠️",
}

# ══════════════════════════════════════════════════════════════════════════════
# OLLAMA  — Phi3 explanation
# ══════════════════════════════════════════════════════════════════════════════

def _ollama(prompt: str) -> str:
    """Send a prompt to Phi3 via Ollama and return the response string."""
    try:
        response = http_requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=180,
        )
        response.raise_for_status()
        return response.json().get("response", "No response from model.").strip()
    except http_requests.exceptions.ConnectionError:
        return "⚠️ Ollama is not running. Start it with: `ollama serve`"
    except Exception as exc:
        return f"⚠️ Unavailable: {exc}"


def get_model_reasoning(
    claim: str,
    evidence: str,
    label: str,
    confidence: float,
    probs: list,
    top5_tokens: list,   # list of (token_str, score) tuples
) -> str:
    """
    Ask Phi3 to narrate how the model derived the verdict,
    grounded in the attribution scores and class probabilities.
    """
    token_lines = "\n".join(
        f"  {i+1}. {t.replace(chr(9649), '').strip()!r:20s} {s:+.4f}  "
        f"({'toward CONTRADICT/SUPPORT' if s > 0 else 'away from predicted label'})"
        for i, (t, s) in enumerate(top5_tokens)
    )
    prob_line = (
        f"SUPPORT={probs[0]:.1%}  NOT_ENOUGH_INFO={probs[1]:.1%}  CONTRADICT={probs[2]:.1%}"
    )
    prompt = f"""You are explaining how an NLI model reached its verdict. Use only the data below.

Claim: {claim}
Evidence: {evidence}
Predicted label: {label} (confidence {confidence:.1%})
Class probabilities: {prob_line}
Top-5 attribution tokens (gradient-based saliency):
{token_lines}

In 2-3 sentences, explain what the model focused on to reach this verdict. Reference specific tokens and probabilities. Do not add outside knowledge."""
    return _ollama(prompt)


def get_scientific_explanation(claim: str, evidence: str, label: str) -> str:
    """
    Ask Phi3 to explain the verdict from a scientific/content perspective.
    """
    prompt = f"""Claim: {claim}
Evidence: {evidence}
Verdict: {label}

In 2 sentences, explain why the evidence {label.lower().replace('_', ' ')}s the claim from a scientific perspective. Be specific and concise."""
    return _ollama(prompt)

# ══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def render_verdict_banner(label: str, confidence: float) -> None:
    """Render a coloured verdict banner with label and confidence."""
    color = LABEL_COLORS[label]
    icon  = LABEL_ICONS[label]
    label_display = label.replace("_", " ")
    st.markdown(
        f"""
        <div style="
            background: {color}22;
            border-left: 5px solid {color};
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 1rem;
        ">
            <span style="font-size: 1.4rem; font-weight: 700; color: {color};">
                {icon} {label_display}
            </span>
            <span style="font-size: 0.95rem; color: #888; margin-left: 12px;">
                confidence {confidence:.1%}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(probabilities: list) -> None:
    """Render SUPPORT / NEI / CONTRADICT probability bars."""
    labels = ["SUPPORT", "NOT_ENOUGH_INFO", "CONTRADICT"]
    st.markdown("**Class probabilities**")
    for lbl, prob in zip(labels, probabilities):
        color       = LABEL_COLORS[lbl]
        display_lbl = lbl.replace("_", " ")
        st.markdown(
            f"""
            <div style="margin-bottom: 6px;">
                <div style="display:flex; justify-content:space-between; font-size:0.82rem; margin-bottom:2px;">
                    <span style="color:#ccc;">{display_lbl}</span>
                    <span style="color:#ccc;">{prob:.1%}</span>
                </div>
                <div style="background:#333; border-radius:4px; height:10px;">
                    <div style="width:{prob*100:.1f}%; background:{color}; height:10px; border-radius:4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_attribution_overlay(attributions: list) -> None:
    """
    Render token attribution as a highlighted text overlay + top-5 token table.
    Green = pushes toward predicted label.
    Red   = pushes away from predicted label.
    Tokens in DISPLAY_SKIP_TOKENS are rendered plain (no highlight).
    """
    st.markdown("**Token attributions** — green: supports verdict · red: opposes verdict")

    html_parts = []
    for token, score in attributions:
        # Clean SentencePiece leading-space marker
        display = token.replace("▁", " ").replace("[CLS]", "").replace("[SEP]", "")
        if not display.strip():
            html_parts.append(f'<span style="color:#333;">{display}</span>')
            continue

        if token in DISPLAY_SKIP_TOKENS:
            html_parts.append(f'<span style="color:#333;">{display}</span>')
            continue

        # Clamp score to [-1, 1] for colour calculation
        s = max(-1.0, min(1.0, score))
        if s > 0:
            bg   = f"rgba(46,204,113,{s*0.7:.2f})"
            text = "#003300" if s > 0.5 else "#005500"
        else:
            bg   = f"rgba(231,76,60,{abs(s)*0.7:.2f})"
            text = "#330000" if abs(s) > 0.5 else "#550000"

        html_parts.append(
            f'<span style="background:{bg}; color:{text}; border-radius:3px; '
            f'padding:2px 4px; margin:0 1px; font-size:0.93rem; font-weight:500;" '
            f'title="score: {score:+.3f}">{display}</span>'
        )

    st.markdown(
        f'<div style="line-height:2.4; padding:12px 14px; background:#f8f8f8; '
        f'border-radius:6px; font-family:monospace; border:1px solid #ddd;">{"".join(html_parts)}</div>',
        unsafe_allow_html=True,
    )

    # ── Top-5 most influential content tokens ─────────────────────────────────
    filtered = [
        (t, s) for t, s in attributions
        if t.replace("▁", "").strip() not in DISPLAY_SKIP_TOKENS
        and t.replace("▁", "").strip()
    ]
    top5 = sorted(filtered, key=lambda x: abs(x[1]), reverse=True)[:5]

    if top5:
        st.markdown("**Top 5 most influential tokens**")
        rows = []
        for rank, (token, score) in enumerate(top5, 1):
            display = token.replace("▁", " ").strip()
            direction = "→ predicted label" if score > 0 else "← away from label"
            color = "#2ecc71" if score > 0 else "#e74c3c"
            rows.append(
                f'<tr><td style="color:#888;padding:4px 10px;">{rank}</td>'
                f'<td style="font-family:monospace;padding:4px 10px;font-weight:600;">{display}</td>'
                f'<td style="font-family:monospace;color:{color};padding:4px 10px;">{score:+.4f}</td>'
                f'<td style="color:#888;padding:4px 10px;">{direction}</td></tr>'
            )
        st.markdown(
            f'<table style="font-size:0.88rem;border-collapse:collapse;">'
            + "".join(rows) +
            f'</table>',
            unsafe_allow_html=True,
        )


def run_full_pipeline(claim: str, evidence_sentence: str, label_id: int) -> None:
    """
    Run the full analysis pipeline on a single (claim, evidence_sentence) pair
    and render all output panels: verdict, probabilities, attributions, explanation.
    """
    # ── Predict ───────────────────────────────────────────────────────────────
    result = predict(claim, evidence_sentence)
    label  = result["label"]
    conf   = result["confidence"]
    probs  = result["probabilities"]

    render_verdict_banner(label, conf)

    col1, col2 = st.columns([1, 1])

    with col1:
        render_probability_bars(probs)

    with col2:
        st.markdown("**Winning evidence sentence**")
        st.markdown(
            f'<div style="background:#1e1e1e; border-radius:6px; padding:10px; '
            f'font-size:0.88rem; color:#ddd; line-height:1.6;">{evidence_sentence}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Attribution ───────────────────────────────────────────────────────────
    if label != "NOT_ENOUGH_INFO":
        with st.spinner("Computing token attributions…"):
            attrs = attribute(claim, evidence_sentence, result["label_id"])
        render_attribution_overlay(attrs)
    else:
        st.info("Token attributions not available for NOT_ENOUGH_INFO verdicts.")

    st.divider()

    # ── Phi3 — model reasoning ───────────────────────────────────────────────
    st.markdown("**🧠 Model reasoning** — how the model derived this verdict")
    if label != "NOT_ENOUGH_INFO":
        with st.spinner("Asking Phi3 to narrate model reasoning…"):
            reasoning = get_model_reasoning(
                claim, evidence_sentence, label, conf, probs,
                top5_tokens = sorted(
                    [(t, s) for t, s in attrs
                     if t.replace("▁", "").strip() not in DISPLAY_SKIP_TOKENS
                     and t.replace("▁", "").strip()],
                    key=lambda x: abs(x[1]), reverse=True
                )[:5] if label != "NOT_ENOUGH_INFO" else [],
            )
        st.markdown(
            f'<div style="background:#f0f4ff; border-left:4px solid #4a90d9; '
            f'border-radius:6px; padding:14px; font-size:0.92rem; color:#222; line-height:1.7;">{reasoning}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("Model reasoning not available for NOT_ENOUGH_INFO verdicts.")

    st.divider()

    # ── Phi3 — scientific explanation ────────────────────────────────────────
    st.markdown("**🔬 Scientific explanation** — content-level interpretation")
    with st.spinner("Asking Phi3 for scientific explanation…"):
        science = get_scientific_explanation(claim, evidence_sentence, label)
    st.markdown(
        f'<div style="background:#f0fff4; border-left:4px solid #2ecc71; '
        f'border-radius:6px; padding:14px; font-size:0.92rem; color:#222; line-height:1.7;">{science}</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Claim–Evidence Checker",
    page_icon  = "🔬",
    layout     = "wide",
)

# Dark-ish override for a cleaner look
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .stTextArea textarea { font-size: 0.9rem; }
    h1 { font-size: 1.6rem !important; }
    .stDivider { margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🔬 ClaimCheck")
    st.caption("Claim–Evidence Consistency Checker")
    st.divider()

    mode = st.radio(
        "Mode",
        ["Mode 1 — Claim Review", "Mode 2 — Claim Check (RAG)"],
        help="Mode 1: paste your own evidence. Mode 2: retrieve evidence from Semantic Scholar.",
    )

    st.divider()
    st.markdown("**Model**")
    st.caption("DeBERTa-v3-base → MultiNLI → SciFact")
    st.markdown("**Attribution**")
    st.caption("Integrated Gradients (N=25)")
    st.markdown("**Explanation**")
    st.caption(f"Phi3 via Ollama")
    st.markdown("**RAG**")
    st.caption("Semantic Scholar API")

# ══════════════════════════════════════════════════════════════════════════════
# MODE 1 — CLAIM REVIEW
# ══════════════════════════════════════════════════════════════════════════════

if mode == "Mode 1 — Claim Review":
    st.title("Claim Review")
    st.caption("Paste a claim and evidence text. The model will find the most relevant sentence and analyse it.")

    claim    = st.text_area("Claim", placeholder="e.g. Aspirin inhibits the production of thromboxane A2.", height=80)
    evidence = st.text_area("Evidence", placeholder="Paste a sentence or full paragraph…", height=150)

    if st.button("Analyse", type="primary", use_container_width=True):
        if not claim.strip():
            st.warning("Please enter a claim.")
        elif not evidence.strip():
            st.warning("Please enter evidence text.")
        else:
            st.divider()
            with st.spinner("Splitting and scoring sentences…"):
                analysis = split_and_score(claim, evidence)

            winner = analysis["winner"]
            if winner is None:
                st.error("Could not extract any sentences from the evidence text.")
            else:
                # Show all sentence scores in an expander
                with st.expander(f"All sentence scores ({len(analysis['all_scores'])} sentences)", expanded=False):
                    for s in analysis["all_scores"]:
                        color = LABEL_COLORS[s["label"]]
                        icon  = LABEL_ICONS[s["label"]]
                        st.markdown(
                            f'<div style="padding:6px 0; border-bottom:1px solid #333; font-size:0.85rem;">'
                            f'<span style="color:{color};">{icon} {s["label"]} {s["confidence"]:.1%}</span>'
                            f'<br><span style="color:#ccc;">{s["sentence"]}</span></div>',
                            unsafe_allow_html=True,
                        )

                run_full_pipeline(claim, winner["sentence"], winner.get("label_id", 0))

# ══════════════════════════════════════════════════════════════════════════════
# MODE 2 — CLAIM CHECK (RAG)
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.title("Claim Check")
    st.caption("Enter a claim. The system retrieves relevant papers from Semantic Scholar and scores them.")

    claim = st.text_area("Claim", placeholder="e.g. COVID-19 originated in Wuhan, China.", height=80)

    if st.button("Retrieve Papers", type="primary", use_container_width=True):
        if not claim.strip():
            st.warning("Please enter a claim.")
        else:
            with st.spinner("Querying Semantic Scholar and scoring abstracts…"):
                candidates = retrieve(claim)

            if not candidates:
                st.error("No papers with usable abstracts were returned. Try rephrasing the claim.")
            else:
                st.session_state["candidates"] = candidates
                st.session_state["claim"]      = claim
                st.session_state["selected"]   = None

    # ── Paper cards ───────────────────────────────────────────────────────────
    if "candidates" in st.session_state and st.session_state.get("claim") == claim:
        candidates = st.session_state["candidates"]

        st.divider()
        st.markdown(f"**{len(candidates)} papers retrieved** — click a card to analyse")

        for i, paper in enumerate(candidates):
            color        = LABEL_COLORS[paper["label"]]
            icon         = LABEL_ICONS[paper["label"]]
            label_display = paper["label"].replace("_", " ")
            year_str     = str(paper["year"]) if paper.get("year") else "n/a"
            authors_str  = ", ".join(paper["authors"][:3]) +                            (" et al." if len(paper["authors"]) > 3 else "")                            if paper["authors"] else "Authors unavailable"
            authors_str  = f"{authors_str} ({year_str})"

            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        border: 1px solid {color}55;
                        border-left: 4px solid {color};
                        border-radius: 8px;
                        padding: 14px 16px;
                        margin-bottom: 10px;
                        background: #1a1a1a;
                    ">
                        <div style="font-size:1rem; font-weight:600; color:#eee; margin-bottom:4px;">
                            {paper['paper_title']}
                        </div>
                        <div style="font-size:0.78rem; color:#888; margin-bottom:8px;">
                            {authors_str}
                        </div>
                        <div style="font-size:0.85rem; color:#bbb; margin-bottom:10px; line-height:1.5;">
                            {paper['sentence']}
                        </div>
                        <span style="
                            background:{color}22; color:{color};
                            border-radius:4px; padding:2px 8px;
                            font-size:0.78rem; font-weight:600;
                        ">{icon} {label_display} — {paper['confidence']:.1%}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if st.button(f"Analyse this paper", key=f"paper_{i}", use_container_width=False):
                    st.session_state["selected"] = i

        # ── Analysis panel for selected paper ────────────────────────────────
        if st.session_state.get("selected") is not None:
            selected = candidates[st.session_state["selected"]]
            st.divider()
            st.markdown(f"### Analysis — *{selected['paper_title']}*")
            run_full_pipeline(
                claim             = st.session_state["claim"],
                evidence_sentence = selected["sentence"],
                label_id          = {"SUPPORT": 0, "NOT_ENOUGH_INFO": 1, "CONTRADICT": 2}[selected["label"]],
            )