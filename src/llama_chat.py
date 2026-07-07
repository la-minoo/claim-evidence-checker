"""
llama_chat.py  —  v2
LLM chat helper for clAIm.

Supports two backends, selected automatically via environment variable:
    - Groq API (hosted) — used when GROQ_API_KEY is set
      Model: gpt oss 20b
      Fast (~0.3s), free tier, no local dependency
      Used for: web deployment on Render

    - Ollama (local) — used when GROQ_API_KEY is not set
      Model: phi3
      Private, no API cost, requires Ollama running locally
      Used for: local demo on Windows machine

This means:
    - Local machine (no GROQ_API_KEY set) → Ollama/Phi3 unchanged
    - Render deployment (GROQ_API_KEY set as env var) → Groq/LLaMA

No other code changes needed — ollama_chat() interface is identical in both cases.

Exported:
    ollama_chat(messages) → str
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        returns: response string, or raises HTTPException on failure
"""

import os
import requests as http_requests
from fastapi import HTTPException

# ── Groq configuration ────────────────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_API_URL  = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL    = "openai/gpt-oss-20b"
GROQ_TIMEOUT  = 30   # Groq responds in ~0.3s warm; 30s is generous

# ── Ollama configuration (local fallback) ─────────────────────────────────────
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = "phi3"
OLLAMA_TIMEOUT  = 180   # Phi3 cold load on CPU takes ~40s

# ── Backend selection ─────────────────────────────────────────────────────────
_USE_GROQ = bool(GROQ_API_KEY)

if _USE_GROQ:
    print("[llama_chat] Backend: Groq API (llama-3.1-8b-instant)")
else:
    print("[llama_chat] Backend: Ollama local (phi3)")


def _groq_chat(messages: list[dict]) -> str:
    """
    Send a messages array to Groq's OpenAI-compatible chat endpoint.

    Groq uses the OpenAI API format — messages array with role/content pairs,
    same as Ollama /api/chat but with Authorization header and different URL.

    Parameters
    ----------
    messages : list of {"role": ..., "content": ...} dicts

    Returns
    -------
    str — assistant response text

    Raises
    ------
    HTTPException 503 if Groq is unreachable or returns an error
    """
    try:
        resp = http_requests.post(
            GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model":    GROQ_MODEL,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.3,   # low temperature for factual explanation tasks
            },
            timeout=GROQ_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible response format:
        # { "choices": [{ "message": { "role": "assistant", "content": "..." } }] }
        return data["choices"][0]["message"]["content"].strip()

    except http_requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Groq API is unreachable. Check network connection.",
        )
    except http_requests.exceptions.HTTPError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Groq API error: {exc.response.status_code} — {exc.response.text}",
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unexpected Groq response format: {exc}",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Groq error: {exc}")


def _ollama_chat(messages: list[dict]) -> str:
    """
    Send a messages array to Phi3 via Ollama's /api/chat endpoint.

    /api/chat takes a messages array and handles the instruct prompt
    template internally. /api/generate is a raw completion endpoint
    and produces empty responses when given chat-formatted prompts.

    Parameters
    ----------
    messages : list of {"role": ..., "content": ...} dicts

    Returns
    -------
    str — assistant response text

    Raises
    ------
    HTTPException 503 if Ollama is unreachable or returns an error
    """
    try:
        resp = http_requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model":    OLLAMA_MODEL,
                "messages": messages,
                "stream":   False,
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # /api/chat returns: { "message": { "role": "assistant", "content": "..." } }
        return data["message"]["content"].strip()

    except http_requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Unexpected Ollama response format: {exc}",
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama error: {exc}")


def ollama_chat(messages: list[dict]) -> str:
    """
    Unified LLM chat interface — routes to Groq or Ollama automatically.

    Uses Groq if GROQ_API_KEY environment variable is set (hosted deployment).
    Falls back to local Ollama/Phi3 otherwise (local demo).

    Parameters
    ----------
    messages : list of {"role": "system"|"user"|"assistant", "content": str} dicts

    Returns
    -------
    str — assistant response text

    Raises
    ------
    HTTPException 503 on any backend failure
    """
    if _USE_GROQ:
        return _groq_chat(messages)
    else:
        return _ollama_chat(messages)