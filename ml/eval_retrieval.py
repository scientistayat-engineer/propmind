"""Retrieval evaluation: does the RAG retriever return the right document for known questions?"""
import sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from app.services.rag import KB
BUY, LEGAL, INV, AREA = "Property Buying Guide", "Registry & Legal Process", "Investment Tips", "Ames Area Information"
EVAL = [("How much down payment does an FHA loan need?", {BUY}), ("What is PMI and when can it be removed?", {BUY}), ("How much are closing costs?", {BUY}),
        ("What should I check during a home inspection?", {BUY}), ("What is earnest money?", {BUY, LEGAL}), ("What is the transfer tax in Iowa?", {LEGAL}),
        ("How are property taxes billed in Iowa?", {LEGAL}), ("What is title insurance?", {LEGAL}), ("Difference between a warranty deed and a quit claim deed?", {LEGAL}),
        ("How is cap rate calculated?", {INV}), ("What is house hacking?", {INV}), ("Why can negative leverage hurt cash flow?", {INV}),
        ("Which neighborhoods are the most expensive?", {AREA}), ("What is the typical price in NridgHt?", {AREA}), ("Is MeadowV an affordable area?", {AREA})]
def run():
    t1 = t3 = 0; rows = []
    for q, ok in EVAL:
        hits = KB.search(q, 3, min_score=0.0); docs = [h["doc"] for h in hits]
        t1 += docs[0] in ok; t3 += any(d in ok for d in docs); rows.append((q, docs[0], docs[0] in ok))
    return {"queries": len(EVAL), "top1": t1, "top3": t3, "top1_pct": round(100 * t1 / len(EVAL)), "top3_pct": round(100 * t3 / len(EVAL)), "rows": rows}
if __name__ == "__main__":
    r = run(); print({k: v for k, v in r.items() if k != "rows"})
    for q, d, ok in r["rows"]: print("OK " if ok else "MISS", q, "->", d)
    json.dump(r, open(ROOT / "models/retrieval_eval.json", "w"), indent=1)
