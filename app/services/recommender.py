"""Content-based property recommender with plain-language explanations."""
import numpy as np, pandas as pd
from .predictor import MODEL, FEATURES, ROOT

_COLS = ["Neighborhood","SalePrice","BedroomAbvGr","FullBath","GrLivArea","OverallQual","YearBuilt","GarageCars","TotalBsmtSF","LotArea","PricePerSqft"]
DF = pd.read_csv(ROOT / "data/properties_clean.csv")[_COLS].reset_index().rename(columns={"index": "id"})
DF["AIPrice"] = np.expm1(MODEL.predict(DF[FEATURES]))          # AI fair-value estimate for every home
DF["ValueGap"] = (DF["AIPrice"] - DF["SalePrice"]) / DF["AIPrice"]  # positive = priced below AI estimate
PREFS = {"garage", "new_build", "high_quality", "large_lot", "big_home", "basement"}
_PREF_TEST = {"garage": lambda r: r.GarageCars >= 2, "new_build": lambda r: r.YearBuilt >= 2000, "high_quality": lambda r: r.OverallQual >= 7,
              "large_lot": lambda r: r.LotArea >= 11000, "big_home": lambda r: r.GrLivArea >= 1800, "basement": lambda r: r.TotalBsmtSF >= 1000}
_PREF_TEXT = {"garage": "has a {g}-car garage", "new_build": "built in {y} (newer construction)", "high_quality": "high build quality ({q}/10)",
              "large_lot": "generous {l:,}-sqft lot", "big_home": "spacious {s:,} sqft of living area", "basement": "large {b:,}-sqft basement"}

def card(r) -> dict:
    return {"id": int(r.id), "neighborhood": r.Neighborhood, "price": int(r.SalePrice), "ai_price": int(round(r.AIPrice)),
            "bedrooms": int(r.BedroomAbvGr), "bathrooms": int(r.FullBath), "sqft": int(r.GrLivArea), "quality": int(r.OverallQual),
            "year_built": int(r.YearBuilt), "garage": int(r.GarageCars), "basement": int(r.TotalBsmtSF), "lot": int(r.LotArea),
            "price_per_sqft": int(round(r.PricePerSqft)), "value_gap_pct": round(float(r.ValueGap) * 100, 1)}

def get_properties(ids):
    return [card(r) for r in DF[DF.id.isin(ids)].itertuples()]

def recommend(budget: float, neighborhoods=None, min_bedrooms=1, prefs=None, limit=6) -> dict:
    prefs = [p for p in (prefs or []) if p in PREFS]; neighborhoods = neighborhoods or []
    d = DF[(DF.SalePrice <= budget) & (DF.BedroomAbvGr >= min_bedrooms)]
    if neighborhoods: d = d[d.Neighborhood.isin(neighborhoods)]
    if d.empty: return {"count": 0, "results": [], "message": "No properties match. Try a higher budget or fewer filters."}
    util = 1 - (budget - d.SalePrice).abs() / budget            # prefer homes that use the budget well
    pref_hit = sum(d.apply(_PREF_TEST[p], axis=1).astype(float) for p in prefs) / len(prefs) if prefs else 0
    d = d.assign(score=0.30 * util.clip(0, 1) + 0.30 * d.ValueGap.clip(-0.2, 0.3).add(0.2).div(0.5)
                 + 0.20 * d.OverallQual / 10 + (0.20 * pref_hit if prefs else 0.20 * (d.GrLivArea / d.GrLivArea.max())))
    out = []
    for i, r in enumerate(d.sort_values("score", ascending=False).head(limit).itertuples(), 1):
        why = [f"Priced at ${r.SalePrice:,.0f}, ${budget - r.SalePrice:,.0f} under your ${budget:,.0f} budget"]
        if r.ValueGap > 0.03: why.append(f"Listed {r.ValueGap * 100:.0f}% below the AI fair-value estimate of ${r.AIPrice:,.0f}")
        elif r.ValueGap < -0.05: why.append(f"Priced about {abs(r.ValueGap) * 100:.0f}% above the AI estimate, so negotiate")
        else: why.append(f"Priced close to the AI fair-value estimate of ${r.AIPrice:,.0f}")
        why.append(f"{int(r.BedroomAbvGr)} bedrooms" + (f", meeting your {min_bedrooms}+ requirement" if min_bedrooms > 1 else ""))
        if neighborhoods and r.Neighborhood in neighborhoods: why.append(f"Located in {r.Neighborhood}, your preferred area")
        for p in prefs:
            if _PREF_TEST[p](r): why.append("Matches your preference: " + _PREF_TEXT[p].format(g=int(r.GarageCars), y=int(r.YearBuilt), q=int(r.OverallQual), l=int(r.LotArea), s=int(r.GrLivArea), b=int(r.TotalBsmtSF)))
        out.append({**card(r), "rank": i, "match_score": round(float(r.score) * 100), "reasons": why})
    return {"count": int(len(d)), "results": out}
