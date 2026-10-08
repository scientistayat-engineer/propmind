"""PropMind: data cleaning, EDA charts, model training and evaluation."""
import json, joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
CH = ROOT / "app/static/charts"; CH.mkdir(parents=True, exist_ok=True)
GOLD, BG = "#d4af6a", "#0b0b10"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#3a3426", "axes.labelcolor": "#f3ead7", "text.color": "#f3ead7",
    "xtick.color": "#cfc6b0", "ytick.color": "#cfc6b0", "axes.titleweight": "bold", "grid.color": "#2a2620"})

# 1. LOAD + CLEAN
df = pd.read_csv(ROOT / "data/train.csv")
print("raw shape:", df.shape)
df = df.drop(columns=["Id"])
none_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
             "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature","MasVnrType"]
df[none_cols] = df[none_cols].fillna("None")   # NA means "feature absent"
df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(lambda s: s.fillna(s.median()))
df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
df = df[~((df.GrLivArea > 4000) & (df.SalePrice < 300000))]   # two known outliers
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["PricePerSqft"] = df["SalePrice"] / df["GrLivArea"]
print("clean shape:", df.shape, "| missing left:", int(df.isna().sum().sum()))
df.to_csv(ROOT / "data/properties_clean.csv", index=False)

# 2. EDA CHARTS
def save(name): plt.tight_layout(); plt.savefig(CH / name, dpi=130); plt.close()
plt.figure(figsize=(8, 4.5)); sns.histplot(df.SalePrice, bins=40, color=GOLD, kde=True)
plt.title("Sale price distribution"); plt.xlabel("Sale price ($)"); save("price_distribution.png")
order = df.groupby("Neighborhood").SalePrice.median().sort_values(ascending=False)
plt.figure(figsize=(10, 5)); sns.barplot(x=order.index, y=order.values, color=GOLD)
plt.xticks(rotation=60, ha="right"); plt.title("Median price by neighborhood"); plt.ylabel("Median price ($)"); save("neighborhood_median.png")
plt.figure(figsize=(8, 5)); sns.scatterplot(data=df, x="GrLivArea", y="SalePrice", hue="OverallQual", palette="YlOrBr", s=28)
plt.title("Living area vs price"); plt.xlabel("Living area (sqft)"); save("area_vs_price.png")
plt.figure(figsize=(8, 5)); sns.boxplot(data=df, x="OverallQual", y="SalePrice", color=GOLD)
plt.title("Price by overall quality"); save("quality_boxplot.png")
yr = df.groupby("YearBuilt").SalePrice.median().rolling(5, min_periods=1).mean()
plt.figure(figsize=(9, 4.5)); plt.plot(yr.index, yr.values, color=GOLD, lw=2.5)
plt.title("Price trend by year built (5-yr rolling median)"); plt.xlabel("Year built"); plt.grid(alpha=.3); save("price_trend.png")
num = df.select_dtypes("number").corr()["SalePrice"].abs().sort_values(ascending=False).head(11).index
plt.figure(figsize=(8, 6.5)); sns.heatmap(df[num].corr(), annot=True, fmt=".2f", cmap="YlOrBr", annot_kws={"size": 7})
plt.title("Top correlated features"); save("correlation_heatmap.png")

# 3. MODEL
FEATURES_NUM = ["GrLivArea","BedroomAbvGr","FullBath","OverallQual","YearBuilt","GarageCars","TotalBsmtSF","LotArea"]
FEATURES_CAT = ["Neighborhood"]
X, y = df[FEATURES_NUM + FEATURES_CAT], np.log1p(df["SalePrice"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
def prep(): return ColumnTransformer([("nb", TargetEncoder(target_type="continuous", random_state=42), FEATURES_CAT)], remainder="passthrough")
models = {"Linear Regression": LinearRegression(),
          "Decision Tree": DecisionTreeRegressor(max_depth=9, min_samples_leaf=5, random_state=42),
          "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=42)}
results, fitted = {}, {}
for name, m in models.items():
    pipe = Pipeline([("prep", prep()), ("model", m)]).fit(Xtr, ytr)
    p = np.expm1(pipe.predict(Xte)); t = np.expm1(yte)
    cv = cross_val_score(Pipeline([("prep", prep()), ("model", m)]), Xtr, ytr, cv=5, scoring="r2").mean()
    results[name] = {"r2": round(r2_score(t, p), 4), "mae": round(mean_absolute_error(t, p)),
                     "rmse": round(float(np.sqrt(mean_squared_error(t, p)))), "cv_r2": round(cv, 4)}
    fitted[name] = pipe; print(f"{name:18s}", results[name])
best = min(results, key=lambda k: results[k]["mae"]); pipe = fitted[best]; print("best model:", best)
pred = np.expm1(pipe.predict(Xte)); true = np.expm1(yte)
plt.figure(figsize=(6.5, 6.5)); plt.scatter(true, pred, s=14, color=GOLD, alpha=.7)
lim = [true.min(), true.max()]; plt.plot(lim, lim, color="#ff5a3c", ls="--")
plt.title(f"{best}: predicted vs actual"); plt.xlabel("Actual ($)"); plt.ylabel("Predicted ($)"); save("pred_vs_actual.png")
if hasattr(pipe.named_steps["model"], "feature_importances_"):
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    imp = pd.Series(pipe.named_steps["model"].feature_importances_, index=names)
    imp.index = [n.replace("nb__Neighborhood", "Neighborhood").replace("remainder__", "") for n in imp.index]
    imp = imp.sort_values(ascending=False).head(10)[::-1]
    plt.figure(figsize=(8, 5)); plt.barh(imp.index, imp.values, color=GOLD); plt.title("Top 10 feature importances"); save("feature_importance.png")

# 4. SAVE
joblib.dump(pipe, ROOT / "models/price_model.joblib", compress=5)
stats = {"model": best, "metrics": results, "features_num": FEATURES_NUM, "features_cat": FEATURES_CAT,
         "neighborhoods": sorted(df.Neighborhood.unique().tolist()), "rows": int(len(df)),
         "median_price": float(df.SalePrice.median()), "residual_std_log": float(np.std(yte - pipe.predict(Xte))),
         "neighborhood_median": df.groupby("Neighborhood").SalePrice.median().round().to_dict(),
         "neighborhood_ppsf": df.groupby("Neighborhood").PricePerSqft.median().round(1).to_dict()}
json.dump(stats, open(ROOT / "models/model_stats.json", "w"), indent=2)
print("saved. model size KB:", round((ROOT / "models/price_model.joblib").stat().st_size / 1024))
