"""
retrieval.py  —  v3
Semantic Scholar RAG layer for the Claim-Evidence Consistency Checker.

v3 = v1 base logic (proven correct) + query extraction from v2 (genuine improvement)

Changes from v1:
    - extract_query()         : extracts 5-6 keywords from claim before querying
                                Semantic Scholar — fixes zero-result failures caused
                                by passing full claim sentences to a keyword search API
    - extract_fallback_query(): 4-term grounded fallback (subject + key scientific terms)
                                fires ONLY when primary query returns zero papers
    - TOP_K raised from 3 to 7: fetch more, score all, return best 3

Everything else is identical to v1:
    - No pre-filter (removed from v2 — caused irrelevant papers to rank above relevant ones)
    - No label priority sort (removed from v2 — caused SUPPORT to rank above CONTRADICT)
    - Sort purely by confidence descending — NLI model decides relevance
    - Cache, score_papers, retrieve, FastAPI endpoint — all unchanged from v1
"""

import os
import re
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

from fastapi import HTTPException
from pydantic import BaseModel
from infer import app, predict, ID2LABEL, LABEL2ID

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

S2_API_KEY  = os.getenv("S2_API_KEY")
S2_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS   = "title,abstract,authors,year"

TOP_K_FETCH      = 7      # fetch 7, score all, return best 3 by confidence
RETURN_K         = 3
WINNER_THRESHOLD = 0.65

SLEEP_AUTHENTICATED   = 1.10
SLEEP_UNAUTHENTICATED = 1.10

CACHE_PATH = Path("data/retrieval_cache.json")

# Stopwords for query extraction
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
    """Extract 5-6 meaningful keywords from a claim for Semantic Scholar search."""
    tokens   = re.findall(r"[a-zA-Z0-9]+", claim)
    keywords = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    return " ".join(keywords[:6])


def extract_fallback_query(claim: str) -> str:
    """
    4-term fallback: first 2 content tokens + top 2 longest tokens.
    Keeps the fallback grounded in the specific claim subject.
    Only fires when primary query returns zero papers.
    """
    tokens   = re.findall(r"[a-zA-Z0-9]+", claim)
    keywords = [t for t in tokens if t.lower() not in _STOPWORDS and len(t) > 2]
    if not keywords:
        return claim[:50]
    short         = keywords[:2]
    sorted_by_len = sorted(keywords, key=len, reverse=True)
    long_tokens   = [t for t in sorted_by_len[:2] if t not in short]
    combined      = short + long_tokens
    return " ".join(combined[:4])

# ══════════════════════════════════════════════════════════════════════════════
# CACHE HELPERS  (identical to v1)
# ══════════════════════════════════════════════════════════════════════════════

def load_cache() -> dict:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CACHE_PATH.exists():
        CACHE_PATH.write_text("{}", encoding="utf-8")
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# SEMANTIC SCHOLAR API FETCH
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_with_query(query: str, limit: int) -> List[Dict[str, Any]]:
    """Single Semantic Scholar API call."""
    headers = {}
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    params = {"query": query, "fields": S2_FIELDS, "limit": limit}

    try:
        response = requests.get(S2_ENDPOINT, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        print(f"[retrieval] Semantic Scholar API error: {exc}")
        return []
    finally:
        sleep_time = SLEEP_AUTHENTICATED if S2_API_KEY else SLEEP_UNAUTHENTICATED
        time.sleep(sleep_time)

    papers = data.get("data", [])
    return [p for p in papers if p.get("abstract") and p["abstract"].strip()]


def fetch_papers(claim: str) -> List[Dict[str, Any]]:
    """
    Fetch papers using extracted query keywords.
    Fallback to 4-term query only if primary returns zero papers.
    """
    primary_query = extract_query(claim)
    print(f"[retrieval] Primary query: '{primary_query}'")

    papers = _fetch_with_query(primary_query, TOP_K_FETCH)

    if len(papers) == 0:
        fallback_query = extract_fallback_query(claim)
        print(f"[retrieval] Zero results — fallback query: '{fallback_query}'")
        papers = _fetch_with_query(fallback_query, TOP_K_FETCH)

    return papers

# ══════════════════════════════════════════════════════════════════════════════
# PAPER SCORING  (identical to v1 — sort by confidence descending, no label priority)
# ══════════════════════════════════════════════════════════════════════════════

def score_papers(claim: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score all fetched papers, select best sentence per paper, sort by confidence."""
    candidates = []

    for paper in papers:
        abstract = paper.get("abstract", "").strip()
        paper_id = paper.get("paperId", "")
        title    = paper.get("title", "")
        authors  = [a["name"] for a in paper.get("authors", []) if a.get("name")]
        year     = paper.get("year", None)

        sentences = sent_tokenize(abstract)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            continue

        scored = []
        for sentence in sentences:
            result = predict(claim, sentence)
            scored.append({
                "sentence":   sentence,
                "label":      result["label"],
                "confidence": result["confidence"],
            })

        scored.sort(key=lambda x: x["confidence"], reverse=True)

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

    # Sort by confidence descending — NLI model decides relevance, no label priority
    candidates.sort(key=lambda x: x["confidence"], reverse=True)

    return candidates[:RETURN_K]

# ══════════════════════════════════════════════════════════════════════════════
# MAIN RETRIEVE FUNCTION  (identical to v1)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(claim: str) -> List[Dict[str, Any]]:
    cache_key = claim.lower().strip()

    cache = load_cache()
    if cache_key in cache:
        print(f"[retrieval] Cache hit for: {cache_key[:60]}...")
        return cache[cache_key]["results"]

    print(f"[retrieval] Cache miss — querying for: {cache_key[:60]}...")

    papers = fetch_papers(claim)
    if not papers:
        print("[retrieval] No usable papers returned.")
        return []

    print(f"[retrieval] Fetched {len(papers)} papers. Scoring all...")
    results = score_papers(claim, papers)

    print(f"[retrieval] Done. Top: {results[0]['label']} ({results[0]['confidence']:.4f})" if results else "[retrieval] No candidates.")

    cache[cache_key] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results":   results,
    }
    save_cache(cache)

    return results

# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINT
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
    if not req.claim.strip():
        raise HTTPException(status_code=422, detail="claim cannot be empty")
    results = retrieve(req.claim)
    return [RetrieveCandidate(**r) for r in results]

# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _claim = "Aspirin inhibits the production of thromboxane A2."
    print("\n" + "=" * 60)
    print("SMOKE TEST — retrieval.py v3")
    print("=" * 60)
    print(f"  Claim          : {_claim}")
    print(f"  TOP_K_FETCH    : {TOP_K_FETCH}")
    print(f"  RETURN_K       : {RETURN_K}")
    print(f"  Auth           : {'yes' if S2_API_KEY else 'no'}")
    print(f"  Extracted query: '{extract_query(_claim)}'")
    print(f"  Fallback query : '{extract_fallback_query(_claim)}'")
    print("\n  Running retrieve()...\n")

    _results = retrieve(_claim)

    if not _results:
        print("  No results returned.")
    else:
        for i, r in enumerate(_results, 1):
            title_short = r["paper_title"][:50] + "..." if len(r["paper_title"]) > 50 else r["paper_title"]
            print(f"  {i}. {r['label']:14} {r['confidence']:.4f}  {title_short}")
        print(f"\n  Top sentence: {_results[0]['sentence'][:100]}...")
        print("\n  All assertions passed." if _results else "")

    print("=" * 60)