(async () => {
  const { $, $$, fmt, get, modal, page } = PM;
  const CH = [["price_distribution", "Sale price distribution"], ["neighborhood_median", "Median price by neighborhood"], ["area_vs_price", "Living area vs price"], ["quality_boxplot", "Price by overall quality"], ["price_trend", "Price trend by year built"], ["correlation_heatmap", "Most correlated features"], ["pred_vs_actual", "Predicted vs actual (test set)"], ["feature_importance", "Top feature importances"]];
  $("#gal").innerHTML = CH.map(([f, t]) => `<div><img loading="lazy" src="${window.CHARTS}charts/${f}.png" alt="${t}" data-t="${t}"><p>${t}</p></div>`).join("");
  $("#gal").addEventListener("click", e => { if (e.target.tagName === "IMG") { const m = modal(`<img src="${e.target.src}" style="width:100%;border-radius:10px"><p class="mut" style="margin-top:8px">${e.target.dataset.t}</p>`, true); } });
  try {
    const d = await get("/api/insights"), rows = d.neighborhoods, M = d.metrics;
    $("#k1").textContent = d.overall.homes.toLocaleString(); $("#k2").textContent = fmt(d.overall.median); $("#k3").textContent = rows.length; $("#k4").textContent = M[d.model].r2;
    const fm = { median_price: fmt, ppsf: v => "$" + v, quality: v => v + "/10", year: v => v };
    const draw = k => { const s = [...rows].sort((a, b) => b[k] - a[k]), hi = Math.max(...s.map(r => r[k])), lo = k === "median_price" ? 0 : Math.min(...s.map(r => r[k])) * 0.7;
      $("#bars").innerHTML = s.map(r => `<a class="hb" href="${page("discover")}?nb=${r.Neighborhood}"><span>${r.Neighborhood}</span><div class="t"><i style="width:${((r[k] - lo) / (hi - lo)) * 100}%"></i></div><b>${fm[k](r[k])}</b></a>`).join(""); };
    draw("median_price");
    $("#metric").addEventListener("click", e => { const b = e.target.closest(".chip"); if (!b) return; $$("#metric .chip").forEach(x => x.classList.toggle("on", x === b)); draw(b.dataset.m); });
    $("#mt").innerHTML = "<tr><th>Model</th><th>R²</th><th>MAE</th><th>RMSE</th></tr>" + Object.entries(M).map(([n, v]) => `<tr><td>${n}${n === d.model ? ' <span class="pill neu">in use</span>' : ""}</td><td>${v.r2}</td><td>${fmt(v.mae)}</td><td>${fmt(v.rmse)}</td></tr>`).join("");
  } catch (e) { $("#bars").innerHTML = '<div class="empty">Could not load insights. Is the API running?</div>'; }
})();
