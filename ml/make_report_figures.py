"""Diagrams + final evaluation numbers used by the README, report and presentation."""
import json, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]; IMG = ROOT / "docs/images"; IMG.mkdir(parents=True, exist_ok=True)
BG, GOLD, GOLD2, FG, MUT, RED, CARD = "#0b0b10", "#d4af6a", "#b8883a", "#f3ead7", "#a39a85", "#ff5a3c", "#17150f"

def box(ax, x, y, w, h, title, sub="", hl=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=CARD, ec=GOLD if hl else "#5a4a2a", lw=2.2 if hl else 1.4))
    ax.text(x + w / 2, y + h / 2 + (0.14 if sub else 0), title, ha="center", va="center", color=GOLD if hl else FG, fontsize=13, fontweight="bold")
    if sub: ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center", color=MUT, fontsize=9.5)
def arrow(ax, p, q, label=""):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, color=GOLD, lw=1.8, shrinkA=2, shrinkB=2))
    if label: ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.16, label, ha="center", color=MUT, fontsize=8.5)
def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.text(w / 2, h - 0.35, title, ha="center", va="center", color=GOLD, fontsize=17, fontweight="bold", family="serif"); return fig, ax

# 1. architecture
fig, ax = canvas(16, 8.2, "PropMind system architecture")
box(ax, 0.5, 3.2, 2.7, 1.6, "Browser", "HTML · CSS · JavaScript\n7 pages", True)
box(ax, 4.3, 3.2, 2.9, 1.6, "FastAPI", "REST /api/*  +  Jinja2 pages", True)
for i, (t, s) in enumerate([("Predictor", "price + what-if"), ("Recommender", "scoring + reasons"), ("RAG retriever", "chunk + TF-IDF"), ("AI assistant", "prompts + chat")]):
    y = 6.15 - i * 1.55; box(ax, 8.4, y - 0.55, 2.7, 1.1, t, s)
    arrow(ax, (7.2, 4.0), (8.4, y)); 
for i, (t, s) in enumerate([("Random Forest", "price_model.joblib"), ("Property data", "properties_clean.csv"), ("Knowledge base", "4 docs + user uploads"), ("Groq API", "Llama 3.3 70B")]):
    y = 6.15 - i * 1.55; box(ax, 12.7, y - 0.55, 2.8, 1.1, t, s, i == 3); arrow(ax, (11.1, y), (12.7, y))
arrow(ax, (3.2, 4.0), (4.3, 4.0), "fetch JSON")
ax.annotate("", xy=(9.75, 2.5), xytext=(9.75, 2.05), arrowprops=dict(arrowstyle="-|>", color=GOLD2, lw=1.4)); ax.text(9.95, 2.28, "RAG context", color=MUT, fontsize=8.5, va="center")
fig.savefig(IMG / "architecture.png", dpi=110, facecolor=BG, bbox_inches="tight"); plt.close(fig)

# 2. flow diagrams
def flow(name, title, steps, w=16, h=3.6):
    fig, ax = canvas(w, h, title); n = len(steps); bw = (w - 1) / n - 0.35
    for i, (t, s) in enumerate(steps):
        x = 0.5 + i * ((w - 1) / n); box(ax, x, 0.9, bw, 1.6, t, s, i in (0, n - 1))
        if i < n - 1: arrow(ax, (x + bw, 1.7), (x + (w - 1) / n, 1.7))
    fig.savefig(IMG / name, dpi=110, facecolor=BG, bbox_inches="tight"); plt.close(fig)
flow("rag_pipeline.png", "RAG pipeline", [("Documents", "MD · PDF · TXT"), ("Chunking", "~170 words\nby section"), ("TF-IDF index", "stemmed words\n+ word pairs"), ("Retrieve", "top-4 passages\ncosine similarity"), ("Prompt", "answer only from\ncontext, cite [n]"), ("Answer", "Groq LLM +\nsource cards")])
flow("ml_pipeline.png", "Machine learning workflow", [("Raw data", "1,460 sales\n81 columns"), ("Cleaning", "NA meaning,\nimpute, outliers"), ("Features", "9 inputs +\nlog target"), ("Split", "80% train\n20% test"), ("3 models", "target-encode\n+ regressors"), ("Evaluate", "R² · MAE\n5-fold CV"), ("Deploy", "joblib +\nFastAPI")])

# 3. final evaluation on the production model (same split as train.py)
df = pd.read_csv(ROOT / "data/properties_clean.csv"); st = json.load(open(ROOT / "models/model_stats.json"))
X = df[st["features_num"] + st["features_cat"]]; y = np.log1p(df.SalePrice)
_, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
pipe = joblib.load(ROOT / "models/price_model.joblib"); p, t = np.expm1(pipe.predict(Xte)), np.expm1(yte); ape = np.abs(p - t) / t
final = {"r2": round(r2_score(t, p), 4), "mae": round(float(mean_absolute_error(t, p))), "within10": round(float((ape < .10).mean() * 100)), "within15": round(float((ape < .15).mean() * 100)),
         "within20": round(float((ape < .20).mean() * 100)), "median_ape": round(float(np.median(ape) * 100), 1), "test_n": int(len(t)), "metrics": st["metrics"]}
json.dump(final, open(ROOT / "models/final_eval.json", "w"), indent=2); print(final)

# 4. model comparison + error distribution
M = st["metrics"]; names = list(M); fig, axs = plt.subplots(1, 2, figsize=(13, 4.6)); fig.patch.set_facecolor(BG)
for a in axs: a.set_facecolor(BG); [s.set_color("#3a3426") for s in a.spines.values()]; a.tick_params(colors="#cfc6b0")
cols = [GOLD2, "#6b5a38", GOLD]
b = axs[0].bar(names, [M[n]["r2"] for n in names], color=cols); axs[0].set_ylim(0.7, 0.92); axs[0].set_title("R² on test set (higher is better)", color=FG, fontweight="bold")
for r, n in zip(b, names): axs[0].text(r.get_x() + r.get_width() / 2, r.get_height() + .003, f'{M[n]["r2"]:.3f}', ha="center", color=FG, fontweight="bold")
b = axs[1].bar(names, [M[n]["mae"] for n in names], color=cols); axs[1].set_ylim(0, 25000); axs[1].set_title("Mean absolute error in $ (lower is better)", color=FG, fontweight="bold")
for r, n in zip(b, names): axs[1].text(r.get_x() + r.get_width() / 2, r.get_height() + 400, f'${M[n]["mae"]:,}', ha="center", color=FG, fontweight="bold")
for a in axs: a.tick_params(axis="x", labelsize=10)
fig.tight_layout(); fig.savefig(IMG / "model_comparison.png", dpi=110, facecolor=BG); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4.4)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG); [s.set_color("#3a3426") for s in ax.spines.values()]; ax.tick_params(colors="#cfc6b0")
ax.hist((p - t) / 1000, bins=36, color=GOLD); ax.axvline(0, color=RED, ls="--"); ax.set_title("Prediction error distribution (test set)", color=FG, fontweight="bold"); ax.set_xlabel("Error ($ thousands)", color=FG)
fig.tight_layout(); fig.savefig(IMG / "error_distribution.png", dpi=110, facecolor=BG); plt.close(fig)
