"""Price prediction service: loads the trained model and explains its estimates."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL = joblib.load(ROOT / "models/price_model.joblib")
STATS = json.load(open(ROOT / "models/model_stats.json"))
FEATURES = STATS["features_num"] + STATS["features_cat"]
NEIGHBORHOODS = STATS["neighborhoods"]

def _price(d: dict) -> float:
    return float(np.expm1(MODEL.predict(pd.DataFrame([d])[FEATURES])[0]))

def predict(d: dict) -> dict:
    """Return the estimate, an 80% likely range, and what-if effects of key features."""
    log_p = float(MODEL.predict(pd.DataFrame([d])[FEATURES])[0]); price = float(np.expm1(log_p))
    s = STATS["residual_std_log"]
    low, high = float(np.expm1(log_p - 1.28 * s)), float(np.expm1(log_p + 1.28 * s))
    nb_med = STATS["neighborhood_median"].get(d["Neighborhood"], STATS["median_price"])
    what_if = []
    if d["OverallQual"] < 10:
        what_if.append({"change": "+1 overall quality", "effect": round(_price({**d, "OverallQual": d["OverallQual"] + 1}) - price)})
    what_if.append({"change": "+200 sqft living area", "effect": round(_price({**d, "GrLivArea": d["GrLivArea"] + 200}) - price)})
    if d["GarageCars"] < 4:
        what_if.append({"change": "+1 garage space", "effect": round(_price({**d, "GarageCars": d["GarageCars"] + 1}) - price)})
    return {"price": round(price), "low": round(low), "high": round(high),
            "price_per_sqft": round(price / max(d["GrLivArea"], 1)),
            "neighborhood_median": round(nb_med), "vs_neighborhood_pct": round((price / nb_med - 1) * 100, 1),
            "confidence": round(max(0.0, min(1.0, 1 - (high - low) / (2 * price))) * 100),
            "what_if": what_if, "model": STATS["model"], "r2": STATS["metrics"][STATS["model"]]["r2"]}
