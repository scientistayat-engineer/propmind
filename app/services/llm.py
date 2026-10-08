"""Groq chat completion client with graceful offline fallback."""
import os

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

class LLMUnavailable(Exception):
    pass

def configured() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))

def complete(messages, temperature=0.3, max_tokens=600) -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key: raise LLMUnavailable("GROQ_API_KEY is not set")
    import httpx
    try:
        r = httpx.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"}, timeout=40,
                       json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), "messages": messages,
                             "temperature": temperature, "max_tokens": max_tokens})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise LLMUnavailable(str(e))
