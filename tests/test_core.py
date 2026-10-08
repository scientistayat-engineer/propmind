"""Core logic tests (no web server needed).  Run:  python -m pytest -q   or   python tests/test_core.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.services.predictor import predict, NEIGHBORHOODS
from app.services.recommender import recommend, get_properties
from app.services.rag import KB
from app.services import assistant

HOME = dict(GrLivArea=1710, BedroomAbvGr=3, FullBath=2, OverallQual=7, YearBuilt=2003, GarageCars=2, TotalBsmtSF=856, LotArea=8450, Neighborhood="CollgCr")

def test_predict_shape_and_range():
    r = predict(HOME)
    assert 100_000 < r["price"] < 400_000 and r["low"] < r["price"] < r["high"] and r["what_if"]

def test_quality_increases_price():
    assert predict({**HOME, "OverallQual": 9})["price"] > predict({**HOME, "OverallQual": 5})["price"]

def test_premium_neighborhood_costs_more():
    assert predict({**HOME, "Neighborhood": "NridgHt"})["price"] > predict({**HOME, "Neighborhood": "MeadowV"})["price"]

def test_recommender_respects_filters():
    r = recommend(250000, ["CollgCr"], 3, ["garage"], 5)
    assert r["results"] and all(x["price"] <= 250000 and x["bedrooms"] >= 3 and x["neighborhood"] == "CollgCr" and x["reasons"] for x in r["results"])

def test_recommender_no_match_message():
    assert recommend(40000, [], 5)["count"] == 0

def test_get_properties():
    assert get_properties([0, 1])[0]["id"] in (0, 1)

def test_rag_retrieves_right_document():
    assert KB.search("transfer tax Iowa", 1)[0]["doc"] == "Registry & Legal Process"
    assert KB.search("down payment FHA loan", 1)[0]["doc"] == "Property Buying Guide"
    assert KB.search("cap rate NOI", 1)[0]["doc"] == "Investment Tips"

def test_rag_stemming_finds_inspection_guidance():
    assert KB.search("What should I check during a home inspection?", 1)[0]["doc"] == "Property Buying Guide"

def test_rag_upload_is_searchable():
    KB.add_document("Test Doc", "The quick purple zebra approval clause requires notarized signatures from every guarantor. " * 6)
    assert KB.search("purple zebra notarized guarantor", 1)[0]["doc"] == "Test Doc"

def test_assistant_offline_fallbacks():
    assert assistant.chat("What is NridgHt like?")["answer"]
    assert assistant.rag_answer("what is the transfer tax in Iowa")["sources"]
    assert assistant.summarize(get_properties([5])[0])["summary"]

if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"): f(); print("PASS", n)
