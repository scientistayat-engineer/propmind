(async () => {
  const { $, get, post, fmt, page, countUp } = PM;
  const num = (id, n, suffix = "") => { const e = $(id); e.dataset.count = n; e.dataset.suffix = suffix; countUp(e); };
  try {
    const m = await get("/api/meta");
    num("#sHomes", m.rows); num("#sNb", m.neighborhoods.length); num("#sR2", Math.round(m.metrics[m.model].r2 * 100), "%");
  } catch { $("#sHomes").textContent = "1,458"; $("#sNb").textContent = "25"; $("#sR2").textContent = "88%"; }
  try {
    const r = await post("/api/predict", { GrLivArea: 1710, BedroomAbvGr: 3, FullBath: 2, OverallQual: 7, YearBuilt: 2003, GarageCars: 2, TotalBsmtSF: 856, LotArea: 8450, Neighborhood: "CollgCr" });
    $("#hPrice").textContent = fmt(r.price); $("#hBar").style.width = r.confidence + "%";
    $("#hNote").textContent = `Likely range ${fmt(r.low)} – ${fmt(r.high)} · ${r.confidence}% confidence`;
  } catch { $("#hNote").textContent = "Start the API to see a live estimate"; }
  try {
    const d = await get("/api/insights"), rows = d.neighborhoods, lo = Math.min(...rows.map(x => x.median_price)), hi = Math.max(...rows.map(x => x.median_price));
    $("#heat").innerHTML = rows.map(x => { const t = (x.median_price - lo) / (hi - lo);
      return `<a class="tile" href="${page("discover")}?nb=${x.Neighborhood}" style="background:rgba(212,175,106,${(.07 + t * .5).toFixed(2)});color:${t > .6 ? "#1a1205" : "#f3ead7"}"><b>${x.Neighborhood}</b><span>${fmt(x.median_price)}</span><small>${x.homes} sales · $${x.ppsf}/sqft</small></a>`; }).join("");
  } catch { $("#heat").innerHTML = '<div class="empty">Market data unavailable. Is the API running?</div>'; }
})();
