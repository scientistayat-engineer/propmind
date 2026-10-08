"""Lightweight RAG retriever: chunk documents, index with TF-IDF, return the best passages with sources."""
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from .predictor import ROOT

DOCS_DIR = ROOT / "docs"
TITLES = {"property_buying_guide.md": "Property Buying Guide", "legal_registration_guide.md": "Registry & Legal Process",
          "investment_tips.md": "Investment Tips", "area_information.md": "Ames Area Information"}


_SUFFIXES = ("ations", "ation", "ings", "ing", "ions", "ion", "ors", "or", "ers", "er", "ies", "ied", "es", "s", "ed", "ly")
def _stem(w: str) -> str:
    """Very light suffix stripping so 'inspection', 'inspector' and 'inspections' share one term."""
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 4: return w[: -len(suf)]
    return w

def _analyze(text: str):
    toks = [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]
    return toks + [a + " " + b for a, b in zip(toks, toks[1:])]

class KnowledgeBase:
    def __init__(self):
        self.chunks, self.docs = [], {}
        for f in sorted(DOCS_DIR.glob("*.md")):
            self.add_document(TITLES.get(f.name, f.stem.replace("_", " ").title()), f.read_text(), f.name, builtin=True, _fit=False)
        self._fit()

    def _split(self, text, max_words=170):
        title, parts, sec, buf = "Introduction", [], "Introduction", []
        for line in text.splitlines():
            if line.startswith("#"):
                if buf: parts.append((sec, " ".join(buf)))
                sec, buf = line.lstrip("# ").strip(), []
            elif line.strip():
                t = re.sub(r"\*\*|\*", "", line.strip())
                buf.append("• " + t[2:] if t.startswith("- ") else t)
        if buf: parts.append((sec, " ".join(buf)))
        out = []
        for sec, body in parts:
            words = body.split()
            for i in range(0, len(words), max_words - 30):
                piece = " ".join(words[i:i + max_words])
                if len(piece.split()) > 12: out.append((sec, piece))
        return out

    def add_document(self, title, text, filename=None, builtin=False, _fit=True):
        new = self._split(text)
        self.chunks = [c for c in self.chunks if c["doc"] != title]
        self.chunks += [{"doc": title, "section": s, "text": t} for s, t in new]
        self.docs[title] = {"title": title, "filename": filename or title, "words": len(text.split()), "chunks": len(new), "builtin": builtin}
        if _fit: self._fit()
        return self.docs[title]

    def _fit(self):
        self.vec = TfidfVectorizer(analyzer=_analyze, sublinear_tf=True)
        self.mat = self.vec.fit_transform([f'{c["doc"]} {c["section"]} {c["doc"]} {c["section"]} {c["text"]}' for c in self.chunks])  # titles counted twice

    def search(self, query, k=4, min_score=0.06):
        sims = cosine_similarity(self.vec.transform([query]), self.mat).ravel()
        idx = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 3)} for i in idx if sims[i] >= min_score]

    def list_docs(self): return list(self.docs.values())

KB = KnowledgeBase()
