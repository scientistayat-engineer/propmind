"""PropMind FastAPI application: pages + JSON API."""
from io import BytesIO
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .services import assistant, llm
from .services.predictor import NEIGHBORHOODS, STATS, predict
from .services.rag import KB
from .services.recommender import PREFS, DF, get_properties, recommend

BASE = Path(__file__).resolve().parent
app = FastAPI(title="PropMind", description="AI Real Estate Intelligence Platform", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

class PredictIn(BaseModel):
    GrLivArea: int = Field(1500, ge=300, le=6000)
    BedroomAbvGr: int = Field(3, ge=0, le=8)
    FullBath: int = Field(2, ge=0, le=4)
    OverallQual: int = Field(6, ge=1, le=10)
    YearBuilt: int = Field(2000, ge=1870, le=2026)
    GarageCars: int = Field(2, ge=0, le=4)
    TotalBsmtSF: int = Field(900, ge=0, le=6000)
    LotArea: int = Field(9000, ge=1000, le=60000)
    Neighborhood: str

class RecommendIn(BaseModel):
    budget: float = Field(..., ge=30000, le=1_000_000)
    neighborhoods: List[str] = []
    min_bedrooms: int = Field(1, ge=0, le=6)
    preferences: List[str] = []
    limit: int = Field(6, ge=1, le=12)

class ChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=1500)
    history: List[dict] = []

class AskIn(BaseModel):
    question: str = Field(..., min_length=2, max_length=800)

class SummaryIn(BaseModel):
    property_id: int

PAGES = {"home": "/", "predict": "/predict", "discover": "/discover", "insights": "/insights", "documents": "/documents", "assistant": "/assistant", "about": "/about"}

def _page(name):
    async def view(request: Request):
        return templates.TemplateResponse(request, f"{name}.html", {"neighborhoods": NEIGHBORHOODS, "page": name, "P": PAGES, "S": "/static/", "API_BASE": ""})
    return view

for _name, _path in PAGES.items():
    app.add_api_route(_path, _page(_name), response_class=HTMLResponse, include_in_schema=False)

@app.get("/api/health")
def health():
    return {"status": "ok", "model": STATS["model"], "ai": "groq" if llm.configured() else "offline", "documents": len(KB.list_docs())}

@app.get("/api/meta")
def meta():
    return {"neighborhoods": NEIGHBORHOODS, "preferences": sorted(PREFS), "metrics": STATS["metrics"], "model": STATS["model"], "rows": STATS["rows"], "median_price": STATS["median_price"]}

@app.post("/api/predict")
def api_predict(body: PredictIn):
    if body.Neighborhood not in NEIGHBORHOODS: raise HTTPException(422, "Unknown neighborhood")
    return predict(body.model_dump())

@app.post("/api/recommend")
def api_recommend(body: RecommendIn):
    return recommend(body.budget, body.neighborhoods, body.min_bedrooms, body.preferences, body.limit)

@app.get("/api/properties")
def api_properties(ids: str):
    return {"results": get_properties([int(i) for i in ids.split(",") if i.strip().isdigit()][:24])}

@app.post("/api/summary")
def api_summary(body: SummaryIn):
    props = get_properties([body.property_id])
    if not props: raise HTTPException(404, "Property not found")
    return {**assistant.summarize(props[0]), "property": props[0]}

@app.post("/api/chat")
def api_chat(body: ChatIn):
    return assistant.chat(body.message, body.history)

@app.post("/api/rag/ask")
def api_rag(body: AskIn):
    return assistant.rag_answer(body.question)

@app.get("/api/rag/docs")
def api_docs():
    return {"documents": KB.list_docs()}

@app.post("/api/rag/upload")
async def api_upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 3_000_000: raise HTTPException(413, "File too large (max 3 MB)")
    name = Path(file.filename or "document").name
    if name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n\n".join((p.extract_text() or "") for p in PdfReader(BytesIO(raw)).pages)
    elif name.lower().endswith((".txt", ".md")):
        text = raw.decode("utf-8", errors="ignore")
    else:
        raise HTTPException(415, "Upload a PDF, TXT or MD file")
    if len(text.split()) < 30: raise HTTPException(422, "Could not read enough text from this file")
    return KB.add_document(Path(name).stem.replace("_", " ").title(), text, name)

@app.get("/api/insights")
def api_insights():
    g = DF.groupby("Neighborhood").agg(median_price=("SalePrice", "median"), ppsf=("PricePerSqft", "median"), homes=("SalePrice", "size"), year=("YearBuilt", "median"), quality=("OverallQual", "median")).round(0).reset_index()
    return {"neighborhoods": g.sort_values("median_price", ascending=False).to_dict("records"), "overall": {"median": STATS["median_price"], "homes": STATS["rows"]}, "metrics": STATS["metrics"], "model": STATS["model"]}

@app.post("/api/report")
def api_report(body: PredictIn):
    """Downloadable PDF valuation report."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    r = predict(body.model_dump()); buf = BytesIO(); c = canvas.Canvas(buf, pagesize=A4); w, h = A4
    c.setFillColor(colors.HexColor("#0b0b10")); c.rect(0, 0, w, h, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#d4af6a")); c.setFont("Helvetica-Bold", 26); c.drawString(50, h - 70, "PropMind Valuation Report")
    c.setFillColor(colors.HexColor("#f3ead7")); c.setFont("Helvetica", 11); c.drawString(50, h - 92, "AI-estimated property value - educational estimate, not an appraisal")
    c.setFillColor(colors.HexColor("#d4af6a")); c.setFont("Helvetica-Bold", 40); c.drawString(50, h - 170, f"${r['price']:,}")
    c.setFillColor(colors.HexColor("#f3ead7")); c.setFont("Helvetica", 12)
    c.drawString(50, h - 195, f"Likely range: ${r['low']:,} - ${r['high']:,}   |   ${r['price_per_sqft']}/sqft   |   {r['vs_neighborhood_pct']:+}% vs {body.Neighborhood} median")
    y = h - 250; c.setFont("Helvetica-Bold", 13); c.drawString(50, y, "Property details"); c.setFont("Helvetica", 12)
    for k, v in [("Neighborhood", body.Neighborhood), ("Living area", f"{body.GrLivArea:,} sqft"), ("Bedrooms / Baths", f"{body.BedroomAbvGr} / {body.FullBath}"), ("Overall quality", f"{body.OverallQual}/10"), ("Year built", body.YearBuilt), ("Garage", f"{body.GarageCars} car"), ("Basement", f"{body.TotalBsmtSF:,} sqft"), ("Lot", f"{body.LotArea:,} sqft")]:
        y -= 20; c.drawString(60, y, f"{k}: {v}")
    y -= 36; c.setFont("Helvetica-Bold", 13); c.drawString(50, y, "What-if effects on price"); c.setFont("Helvetica", 12)
    for wi in r["what_if"]: y -= 20; c.drawString(60, y, f"{wi['change']}: {wi['effect']:+,} USD")
    c.setFont("Helvetica-Oblique", 9); c.drawString(50, 50, f"Model: {r['model']} (R2 {r['r2']}). Trained on Ames, Iowa sales 2006-2010. Generated by PropMind.")
    c.save()
    return Response(buf.getvalue(), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=propmind-report.pdf"})
