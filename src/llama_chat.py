"""
llama_chat.py  —  Ollama chat helper for ClaimCheck
Uses /api/chat instead of /api/generate — correct endpoint for multi-turn
instruct models like Phi3.

/api/chat takes a messages array (role/content pairs) and handles the
prompt template internally. /api/generate is a raw completion endpoint
and produces empty responses when given chat-formatted prompts.

Exported:
    ollama_chat(messages) → str
        messages: list of {"role": "system"|"user"|"assistant", "content": str}
        returns: response string, or raises HTTPException on failure
"""

import requests as http_requests
from fastapi import HTTPException

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL    = "phi3"
OLLAMA_TIMEOUT  = 180


def ollama_chat(messages: list[dict]) -> str:
    """
    Send a messages array to Phi3 via Ollama's /api/chat endpoint.

    Parameters
    ----------
    messages : list of {"role": ..., "content": ...} dicts
        Roles: "system", "user", "assistant"

    Returns
    -------
    str — the assistant's response text

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
        # /api/chat returns: { "message": { "role": "assistant", "content": "..." }, ... }
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