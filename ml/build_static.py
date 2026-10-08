"""Build a static copy of the website for Firebase Hosting (frontend only).

Usage:  python ml/build_static.py --api https://your-app.vercel.app
The static pages call the FastAPI backend (deployed on Vercel) at the given URL.
"""
import argparse, json, shutil
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--api", required=True, help="Base URL of the deployed backend, no trailing slash")
args = ap.parse_args()
api = args.api.rstrip("/")

PAGES = {"home": "index.html", "predict": "predict.html", "discover": "discover.html", "insights": "insights.html",
         "documents": "documents.html", "assistant": "assistant.html", "about": "about.html"}
neighborhoods = json.load(open(ROOT / "models/model_stats.json"))["neighborhoods"]
env = Environment(loader=FileSystemLoader(ROOT / "app/templates"), autoescape=True)
out = ROOT / "public"
if out.exists(): shutil.rmtree(out)
out.mkdir()
shutil.copytree(ROOT / "app/static", out / "static")
for name, fname in PAGES.items():
    html = env.get_template(f"{name}.html").render(neighborhoods=neighborhoods, page=name, P=PAGES, S="static/", API_BASE=api)
    (out / fname).write_text(html, encoding="utf-8")
print(f"Static site written to {out} ({len(PAGES)} pages), API = {api}")
