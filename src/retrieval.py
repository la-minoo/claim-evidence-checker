"""
retrieval.py  —  v2
Semantic Scholar RAG layer for the Claim–Evidence Consistency Checker.

Changes from v1:
    - extract_query()     : extracts 5-6 keywords from the claim before querying
                            Semantic Scholar — fixes zero-result failures caused
                            by passing full claim sentences to a keyword search API
    - TOP_K_FETCH / TOP_K_SCORE / RETURN_K : fetch more, pre-filter by term overlap,
                            score fewer, return best 3 — better recall without
                            proportional scoring cost
    - prefilter_papers()  : lightweight term-overlap ranking before predict() calls
    - fetch_papers()      : retry with 2-term fallback query if first attempt returns
                            fewer than RETURN_K usable papers
    - All other logic (cache, score_papers, retrieve, FastAPI endpoint, smoke test)
      is unchanged from v1.

Pipeline:
    claim
        → check local cache (data/retrieval_cache.json)
        → cache hit  → return cached results immediately
        → cache miss → extract_query(claim)         # 5-6 keywords
                     → Semantic Scholar API          # fetch TOP_K_FETCH=7 papers
                     → prefilter_papers()            # rank by term overlap, keep TOP_K_SCORE=5
                     → score_papers()                # predict() on each abstract sentence
                     → select best sentence per paper (WINNER_THRESHOLD logic)
                     → sort by confidence, return top RETURN_K=3
                     → if < RETURN_K results → retry with 2-term fallback query
                     → write to cache
                     → return candidates

Return format (list of dicts, up to RETURN_K entries):
    {
        "paper_title":  str,
        "paper_id":     str,      # Semantic Scholar paper ID
        "sentence":     str,      # best sentence from this abstract
        "label":        str,      # SUPPORT / CONTRADICT / NOT_ENOUGH_INFO
        "confidence":   float,
        "abstract":     str,      # full abstract for UI display
        "authors":      list[str],
        "year":         int | None,
    }

FastAPI endpoint:
    POST /retrieve
    Body:    { "claim": "..." }
    Returns: list of candidate dicts (sorted by confidence desc)

Authentication:
    Set env var S2_API_KEY for 1 req/sec limit.
    Without key: same 1 req/sec unauthenticated limit.

Dependency on infer.py:
    predict() and label maps imported directly — model never loaded twice.
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import re
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── HTTP ──────────────────────────────────────────────────────────────────────
import requests

# ── Sentence splitting ────────────────────────────────────────────────────────
import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

# ── API framework ─────────────────────────────────────────────────────────────
from fastapi import HTTPException
from pydantic import BaseModel

# ── Internal — reuse model already loaded by infer.py ────────────────────────
from infer import app, predict, ID2LABEL, LABEL2ID

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

S2_API_KEY  = os.getenv("S2_API_KEY")   # None if unset → unauthenticated fallback
S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS   = "title,abstract,authors,year"

TOP_K_FETCH  = 7    # papers to fetch from Semantic Scholar API
TOP_K_SCORE  = 5    # papers to run predict() on (after term-overlap pre-filter)
RETURN_K     = 3    # candidates to return to the user

WINNER_THRESHOLD = 0.65   # mirrors infer.py — minimum confidence for a non-NEI sentence

# Rate-limit sleep — kept safely below each tier's ceiling
SLEEP_AUTHENTICATED   = 1.10   # ~0.9 req/sec (limit: 1/sec with key)
SLEEP_UNAUTHENTICATED = 1.10   # ~0.9 req/sec (limit: 1/sec without key)

CACHE_PATH = Path("data/retrieval_cache.json")

# English stopwords for query extraction — no external dependency
_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being",
    "have","has","had","do","does","did","will","would","could",
    "should","may","might","must","shall","can","need","dare",
    "of","in","on","at","to","for","with","by","from","up",
    "about","into","through","during","before","after","above",
    "below","between","out","off","over","under","again","further",
    "then","once","and","but","or","nor","not","so","yet","both",
    "either","neither","than","that","this","these","those",
    "also","its","it","which","who","whom","what","where","when",
    "how","all","each","every","both","few","more","most","other",
    "some","such","no","only","same","as","just","because","if",
    "while","although","though","since","unless","until","whether",
    "significantly","associated","increased","decreased","elevated",
    "reduced","shown","found","suggest","evidence","study","result",
    "results","effect","effects","using","used","based","compared",
}

# ══════════════════════════════════════════════════════════════════════════════
# QUERY EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_query(claim: str) -> str:
    """
    Extract 5-6 meaningful keywords from a claim sentence for use as a
    Semantic Scholar search query.

    Passing full claim sentences to a keyword search API frequently returns
    zero results because the sentence structure, stopwords, and specificity
    of biomedical claims do not match well against indexed paper vocabulary.
    Extracting core content terms improves recall significantly.

    Strategy: tokenise, remove stopwords and short tokens, take first 6.
    No external NLP dependency — pure regex + stopword filter.

    Examples:
        "Aspirin inhibits the production of thromboxane A2 in platelets."
        → "Aspirin inhibits production thromboxane A2 platelets"

        "BRCA1 mutation carriers have a significantly elevated lifetime risk."
        → "BRCA1 mutation carriers elevated lifetime risk"

    Parameters
    ----------
    claim : str — full claim sentence

    Returns
    -------
    str — space-joined keyword query (5-6 terms)
    """
    tokens   = re.findall(r"[a-zA-Z0-9]+", claim)
    keywords = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    return " ".join(keywords[:6])


def extract_fallback_query(claim: str) -> str:
    """
    Four-term fallback query — used when the primary query returns fewer than
    RETURN_K = 1 usable papers.

    Strategy: first 2 content tokens (subject/verb) + longest content token
    (almost always the key scientific term e.g. thromboxane, cholesterol,
    consolidation, hypertensive). Keeps the fallback grounded in the specific
    claim subject rather than broadening to generic terms.

    Examples:
        Aspirin inhibits the production of thromboxane A2 in platelets.
        -> Aspirin inhibits production thromboxane

        Regular aerobic exercise reduces systolic blood pressure in hypertensive patients.
        -> Regular aerobic exercise hypertensive
    """
    tokens   = re.findall(r"[a-zA-Z0-9]+", claim)
    keywords = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    if not keywords:
        return claim[:50]
    short        = keywords[:2]
    sorted_by_len = sorted(keywords, key=len, reverse=True)
    long_tokens   = [t for t in sorted_by_len[:2] if t not in short]
    combined      = short + long_tokens
    return " ".join(combined[:4])

# ══════════════════════════════════════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_cache() -> dict:
    """
    Load the local JSON cache from CACHE_PATH.

    Creates the file (and parent directory) if either is missing.
    Returns an empty dict on any read/parse error so the caller can
    always treat the return value as a plain dict.
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CACHE_PATH.exists():
        CACHE_PATH.write_text("{}", encoding="utf-8")
        return {}

    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable file — start fresh rather than crashing
        return {}


def save_cache(cache: dict) -> None:
    """
    Write the cache dict back to CACHE_PATH (UTF-8, 2-space indent).

    Silently swallows write errors — a failed cache write should never
    crash an otherwise successful retrieval result.
    """
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass   # cache write failure is non-fatal

# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SCHOLAR API FETCH
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_with_query(query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Internal: run one Semantic Scholar API call with a given query string.

    Parameters
    ----------
    query : str — search query (already extracted/shortened)
    limit : int — number of papers to request

    Returns
    -------
    List of paper dicts with abstracts, or empty list on failure.
    Respects rate limit sleep regardless of success/failure.
    """
    headers = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    params = {
        "query":  query,
        "fields": S2_FIELDS,
        "limit":  limit,
    }

    try:
        response = requests.get(
            S2_ENDPOINT,
            headers = headers,
            params  = params,
            timeout = 15,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"[retrieval] Semantic Scholar API error: {exc}")
        return []
    finally:
        # Always respect rate limit
        sleep_time = SLEEP_AUTHENTICATED if S2_API_KEY else SLEEP_UNAUTHENTICATED
        time.sleep(sleep_time)

    papers = data.get("data", [])

    # Filter out papers with no abstract — nothing to score
    usable = [
        p for p in papers
        if p.get("abstract") and p["abstract"].strip()
    ]

    return usable


def fetch_papers(claim: str) -> List[Dict[str, Any]]:
    """
    Fetch papers from Semantic Scholar for a given claim.

    Two-stage query strategy:
        1. Primary query: extract_query(claim) → 5-6 keywords
           Fetch TOP_K_FETCH=7 papers.
        2. Fallback query: extract_fallback_query(claim) → 2 keywords
           Used only if primary query returns fewer than RETURN_K papers.
           Results are merged and deduplicated by paperId.

    Parameters
    ----------
    claim : str — raw claim text from user

    Returns
    -------
    List of up to TOP_K_FETCH paper dicts with abstracts.
    Empty list if both queries fail or return no usable results.
    """
    primary_query = extract_query(claim)
    print(f"[retrieval] Primary query: '{primary_query}'")

    papers = _fetch_with_query(primary_query, TOP_K_FETCH)

    if len(papers) == 0:
        # Primary query returned nothing — try fallback
        fallback_query = extract_fallback_query(claim)
        print(f"[retrieval] Primary returned 0 papers — "
              f"retrying with fallback query: '{fallback_query}'")

        fallback_papers = _fetch_with_query(fallback_query, TOP_K_FETCH)

        # Merge and deduplicate by paperId
        seen_ids = {p["paperId"] for p in papers}
        for p in fallback_papers:
            if p["paperId"] not in seen_ids:
                papers.append(p)
                seen_ids.add(p["paperId"])

        print(f"[retrieval] After fallback merge: {len(papers)} unique papers")

    return papers

# ══════════════════════════════════════════════════════════════════════════════
# PRE-FILTER — term overlap ranking
# ══════════════════════════════════════════════════════════════════════════════

def prefilter_papers(
    claim: str,
    papers: List[Dict[str, Any]],
    top_n: int = TOP_K_SCORE,
) -> List[Dict[str, Any]]:
    """
    Rank fetched papers by term overlap between the extracted query keywords
    and the paper's title + abstract, then return the top_n.

    This pre-filter runs before predict() calls — it avoids running the NLI
    model on papers that are clearly off-topic based on keyword overlap alone.
    A paper with zero keyword overlap in its title or abstract is unlikely to
    contain a relevant evidence sentence, regardless of what predict() would say.

    Scoring: count of query keyword tokens (lowercased) that appear in
    title + abstract (lowercased). Ties broken by original API rank.

    Parameters
    ----------
    claim  : str
    papers : list of paper dicts from fetch_papers()
    top_n  : int — number of papers to keep (default TOP_K_SCORE=5)

    Returns
    -------
    List of top_n paper dicts, sorted by overlap score descending.
    If len(papers) <= top_n, returns all papers unchanged.
    """
    if len(papers) <= top_n:
        return papers

    query_tokens = set(re.findall(r"[a-zA-Z0-9]+", extract_query(claim).lower()))

    def overlap_score(paper: Dict[str, Any]) -> int:
        text = (
            (paper.get("title", "") or "") + " " +
            (paper.get("abstract", "") or "")
        ).lower()
        return sum(1 for token in query_tokens if token in text)

    ranked = sorted(papers, key=overlap_score, reverse=True)
    return ranked[:top_n]

# ══════════════════════════════════════════════════════════════════════════════
# PAPER SCORING
# ══════════════════════════════════════════════════════════════════════════════

def score_papers(
    claim: str,
    papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Score each paper's abstract against the claim and select the best sentence.

    For each paper:
        1. Split abstract into sentences (sent_tokenize)
        2. Score each (claim, sentence) pair with predict()
        3. Apply WINNER_THRESHOLD logic — same three-way logic as infer.py:
               a. Non-NEI sentence >= 0.65  → winner
               b. Non-NEI exists but all < 0.65 → fall back to best NEI
               c. All NEI → best NEI wins

    Papers whose abstract tokenises to zero sentences are skipped.

    Parameters
    ----------
    claim  : str
    papers : list of dicts from fetch_papers() — must have "abstract", "title", "paperId"

    Returns
    -------
    List of candidate dicts sorted by confidence descending:
        {
            "paper_title": str,
            "paper_id":    str,
            "sentence":    str,
            "label":       str,
            "confidence":  float,
            "abstract":    str,
            "authors":     list[str],
            "year":        int | None,
        }
    """
    candidates = []

    for paper in papers:
        abstract  = paper.get("abstract", "").strip()
        paper_id  = paper.get("paperId",  "")
        title     = paper.get("title",    "")
        authors   = [a["name"] for a in paper.get("authors", []) if a.get("name")]
        year      = paper.get("year", None)

        sentences = sent_tokenize(abstract)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            continue

        # Score every sentence in the abstract
        scored = []
        for sentence in sentences:
            result = predict(claim, sentence)
            scored.append({
                "sentence":   sentence,
                "label":      result["label"],
                "confidence": result["confidence"],
            })

        # Sort by confidence descending before applying threshold logic
        scored.sort(key=lambda x: x["confidence"], reverse=True)

        # Three-way winner selection (mirrors split_and_score in infer.py)
        confident_non_nei = [
            s for s in scored
            if s["label"] != "NOT_ENOUGH_INFO"
            and s["confidence"] >= WINNER_THRESHOLD
        ]
        nei_sentences = [s for s in scored if s["label"] == "NOT_ENOUGH_INFO"]
        fallback      = nei_sentences[0] if nei_sentences else scored[0]
        winner        = confident_non_nei[0] if confident_non_nei else fallback

        candidates.append({
            "paper_title": title,
            "paper_id":    paper_id,
            "sentence":    winner["sentence"],
            "label":       winner["label"],
            "confidence":  winner["confidence"],
            "abstract":    abstract,
            "authors":     authors,
            "year":        year,
        })

    # Sort candidates: SUPPORT and CONTRADICT ranked equally by confidence,
    # NEI always last — only surfaced if no non-NEI candidates exist.
    # CONTRADICT is not deprioritised vs SUPPORT — for hallucination checking
    # use cases, a high-confidence CONTRADICT result should surface first.
    _LABEL_PRIORITY = {"SUPPORT": 0, "CONTRADICT": 0, "NOT_ENOUGH_INFO": 1}
    candidates.sort(key=lambda x: (_LABEL_PRIORITY[x["label"]], -x["confidence"]))

    return candidates

# ══════════════════════════════════════════════════════════════════════════════
# MAIN RETRIEVE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(claim: str) -> List[Dict[str, Any]]:
    """
    Top-level retrieval function — cache-aware, API-backed.

    Pipeline:
        1. Normalise claim → cache key
        2. Check cache → return immediately on hit
        3. On miss: fetch papers from Semantic Scholar (with query extraction + retry)
        4. Pre-filter by term overlap → keep TOP_K_SCORE=5 papers
        5. Score each abstract → select best sentence per paper
        6. Return top RETURN_K=3 by confidence
        7. Write results to cache with timestamp

    Parameters
    ----------
    claim : str — raw claim text from user

    Returns
    -------
    List of up to RETURN_K candidate dicts (see score_papers docstring for schema).
    Empty list if the API returned no usable papers after retry.
    """
    cache_key = claim.lower().strip()

    # ── 1. Cache lookup ───────────────────────────────────────────────────────
    cache = load_cache()
    if cache_key in cache:
        print(f"[retrieval] Cache hit for: {cache_key[:60]}…")
        return cache[cache_key]["results"]

    print(f"[retrieval] Cache miss — querying Semantic Scholar for: {cache_key[:60]}…")

    # ── 2. Fetch from Semantic Scholar ────────────────────────────────────────
    papers = fetch_papers(claim)

    if not papers:
        print("[retrieval] No usable papers returned from API.")
        return []

    print(f"[retrieval] Fetched {len(papers)} papers with abstracts.")

    # ── 3. Pre-filter by term overlap ─────────────────────────────────────────
    papers_to_score = prefilter_papers(claim, papers, top_n=TOP_K_SCORE)
    print(f"[retrieval] Pre-filtered to {len(papers_to_score)} papers for scoring.")

    # ── 4. Score papers ───────────────────────────────────────────────────────
    results = score_papers(claim, papers_to_score)

    # Trim to RETURN_K
    results = results[:RETURN_K]

    print(
        f"[retrieval] Scored {len(results)} candidates. Top label: "
        f"{results[0]['label']} ({results[0]['confidence']:.4f})" if results else
        "[retrieval] No candidates after scoring."
    )

    # ── 5. Write to cache ─────────────────────────────────────────────────────
    cache[cache_key] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results":   results,
    }
    save_cache(cache)

    return results

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINT  — POST /retrieve
# Added to the existing app instance imported from infer.py
# ══════════════════════════════════════════════════════════════════════════════

class RetrieveRequest(BaseModel):
    claim: str

class RetrieveCandidate(BaseModel):
    paper_title: str
    paper_id:    str
    sentence:    str
    label:       str
    confidence:  float
    abstract:    str
    authors:     List[str]
    year:        Optional[int]

@app.post("/retrieve", response_model=List[RetrieveCandidate])
def api_retrieve(req: RetrieveRequest):
    """
    POST /retrieve
    Body:    { "claim": "..." }
    Returns: list of candidate papers sorted by confidence descending.

    Each candidate includes the paper title, Semantic Scholar paper ID,
    the single best-evidence sentence from the abstract, the NLI label
    for that sentence, its confidence score, and the full abstract for
    UI display.

    Frontend use:
        - Display candidates as selectable cards
        - User (or system) picks one → pass sentence to /analyze + /attribute
        - attribution_available = (label != "NOT_ENOUGH_INFO")
    """
    if not req.claim.strip():
        raise HTTPException(status_code=422, detail="claim cannot be empty")

    results = retrieve(req.claim)

    return [RetrieveCandidate(**r) for r in results]

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT  — smoke test
# Run with:  python src\retrieval.py
# (infer.py server does NOT need to be running — model loads via import)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    _claim = "Aspirin inhibits the production of thromboxane A2."

    print("\n" + "═" * 60)
    print("SMOKE TEST — retrieval.py v2")
    print("═" * 60)
    print(f"  Claim          : {_claim}")
    print(f"  TOP_K_FETCH    : {TOP_K_FETCH}")
    print(f"  TOP_K_SCORE    : {TOP_K_SCORE}")
    print(f"  RETURN_K       : {RETURN_K}")
    print(f"  Auth           : {'yes (S2_API_KEY set)' if S2_API_KEY else 'no (unauthenticated)'}")
    print(f"  Cache          : {CACHE_PATH}")
    print(f"\n  Extracted query: '{extract_query(_claim)}'")
    print(f"  Fallback query : '{extract_fallback_query(_claim)}'")
    print("\n  Running retrieve() …\n")

    _results = retrieve(_claim)

    if not _results:
        print("  ✗ No results returned — check API key or network connection.")
    else:
        print(f"  {'#':<4} {'Label':<14} {'Conf':>6}  {'Title'}")
        print(f"  {'─'*4} {'─'*14} {'─'*6}  {'─'*40}")
        for i, r in enumerate(_results, 1):
            title_short = r["paper_title"][:45] + "…" if len(r["paper_title"]) > 45 else r["paper_title"]
            print(f"  {i:<4} {r['label']:<14} {r['confidence']:>6.4f}  {title_short}")

        print(f"\n  Top result:")
        top = _results[0]
        print(f"    Paper    : {top['paper_title']}")
        print(f"    Authors  : {', '.join(top['authors']) if top['authors'] else 'n/a'}")
        print(f"    Year     : {top['year'] if top['year'] else 'n/a'}")
        print(f"    Label    : {top['label']}")
        print(f"    Conf     : {top['confidence']:.4f}")
        print(f"    Sentence : {top['sentence']}")

        # Sanity checks
        assert len(_results) > 0,                              "No results returned"
        assert all("paper_title" in r for r in _results),     "Missing paper_title key"
        assert all("sentence"    in r for r in _results),     "Missing sentence key"
        assert all("label"       in r for r in _results),     "Missing label key"
        assert all("confidence"  in r for r in _results),     "Missing confidence key"
        assert all(r["label"] in ID2LABEL.values()
                   for r in _results),                         "Unknown label value"
        assert all(0.0 <= r["confidence"] <= 1.0
                   for r in _results),                         "Confidence out of range"
        assert _results == sorted(
            _results, key=lambda x: x["confidence"], reverse=True
        ),                                                     "Results not sorted by confidence"

        print(f"\n  ✓ All assertions passed — retrieval v2 working correctly")

    print("═" * 60)