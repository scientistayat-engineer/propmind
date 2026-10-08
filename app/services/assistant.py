"""AI assistant orchestration: general chat, RAG answers and property summaries."""
import re
from . import llm
from .prompts import SYSTEM_PROMPT, RAG_PROMPT, SUMMARY_PROMPT
from .predictor import STATS
from .rag import KB

def _offline_chat(message: str) -> str:
    m = message.lower(); med = STATS["neighborhood_median"]
    for k in med:
        if k.lower() in m:
            return f"{k} has a median sale price of ${med[k]:,.0f} (about ${STATS['neighborhood_ppsf'][k]:,.0f} per sqft) in the Ames data. Use the Price Prediction page for a home-specific estimate. (Offline mode: add GROQ_API_KEY for full AI answers.)"
    if any(w in m for w in ("cheap", "afford", "lowest")):
        k = min(med, key=med.get); return f"The most affordable neighborhood by median price is {k} at ${med[k]:,.0f}. (Offline mode.)"
    if any(w in m for w in ("expens", "premium", "highest", "luxury")):
        k = max(med, key=med.get); return f"The highest median price is in {k} at ${med[k]:,.0f}. (Offline mode.)"
    return "I'm in offline mode, so I can only share neighborhood price facts. Ask about a neighborhood, or try the Documents assistant for buying, legal and investment guidance. Add GROQ_API_KEY to enable full AI answers."

def chat(message: str, history=None) -> dict:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + [h for h in (history or [])[-8:] if h.get("role") in ("user", "assistant")] + [{"role": "user", "content": message}]
    try: return {"answer": llm.complete(msgs), "mode": "groq"}
    except llm.LLMUnavailable: return {"answer": _offline_chat(message), "mode": "offline"}

def rag_answer(question: str, k=4) -> dict:
    hits = KB.search(question, k=k)
    sources = [{"n": i + 1, "doc": h["doc"], "section": h["section"], "score": h["score"], "text": h["text"]} for i, h in enumerate(hits)]
    if not hits: return {"answer": "I could not find this in the uploaded documents. Try rephrasing, or upload a document that covers it.", "sources": [], "mode": "none"}
    ctx = "\n\n".join(f"[{s['n']}] ({s['doc']} - {s['section']}) {s['text']}" for s in sources)
    try:
        ans = llm.complete([{"role": "system", "content": "You are a careful document assistant for a real estate platform."},
                            {"role": "user", "content": RAG_PROMPT.format(context=ctx, question=question)}], temperature=0.1)
        return {"answer": ans, "sources": sources, "mode": "groq"}
    except llm.LLMUnavailable:
        best = sources[0]; snippet = best["text"][:420].rsplit(" ", 1)[0] + "..."
        return {"answer": f"From {best['doc']} ({best['section']}): {snippet} [1]", "sources": sources, "mode": "offline"}

def summarize(prop: dict) -> dict:
    data = ", ".join(f"{k}: {v}" for k, v in prop.items() if k not in ("id", "rank", "reasons", "match_score"))
    try: return {"summary": llm.complete([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": SUMMARY_PROMPT.format(data=data)}], temperature=0.4, max_tokens=250), "mode": "groq"}
    except llm.LLMUnavailable:
        gap = prop.get("value_gap_pct", 0); tone = "below" if gap > 0 else "above"
        return {"summary": (f"This {prop['bedrooms']}-bedroom home in {prop['neighborhood']} offers {prop['sqft']:,} sqft, was built in {prop['year_built']} and has a quality rating of {prop['quality']}/10. "
                            f"It is priced at ${prop['price']:,}, about {abs(gap):.0f}% {tone} the AI fair-value estimate of ${prop['ai_price']:,}. Verify condition with an inspection before making an offer."), "mode": "offline"}
