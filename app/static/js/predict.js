(() => {
  const { $, $$, fmt, post, store, uk, toast, debounce, page, esc } = PM;
  const NUM = ["GrLivArea", "OverallQual", "YearBuilt", "TotalBsmtSF", "LotArea"];
  const QUAL = ["", "Very poor", "Poor", "Fair", "Below average", "Average", "Above average", "Good", "Very good", "Excellent", "Luxury"];
  const state = { BedroomAbvGr: 3, FullBath: 2, GarageCars: 2 };
  let last = null;

  const body = () => { const b = { Neighborhood: $("#Neighborhood").value, ...state }; NUM.forEach(k => b[k] = +$("#" + k).value); return b; };
  function labels() {
    $("#o_GrLivArea").textContent = (+$("#GrLivArea").value).toLocaleString() + " sqft"; $("#o_OverallQual").textContent = $("#OverallQual").value + "/10 · " + QUAL[+$("#OverallQual").value];
    $("#o_YearBuilt").textContent = $("#YearBuilt").value; $("#o_TotalBsmtSF").textContent = (+$("#TotalBsmtSF").value).toLocaleString() + " sqft"; $("#o_LotArea").textContent = (+$("#LotArea").value).toLocaleString() + " sqft";
  }
  let shown = 0;
  function animate(to) { const from = shown, t0 = performance.now(); shown = to; (function f(x) { const k = Math.min((x - t0) / 450, 1); $("#price").textContent = fmt(from + (to - from) * k); if (k < 1) requestAnimationFrame(f); })(t0); }

  async function run() {
    labels();
    try {
      const r = await post("/api/predict", body()); last = r; animate(r.price);
      $("#range").textContent = `80% likely range: ${fmt(r.low)} – ${fmt(r.high)}`;
      const lo = r.low * .8, hi = r.high * 1.2, pc = v => ((v - lo) / (hi - lo)) * 100;
      $("#gauge").style.left = pc(r.low) + "%"; $("#gauge").style.width = (pc(r.high) - pc(r.low)) + "%"; $("#dot").style.left = pc(r.price) + "%";
      $("#ppsf").textContent = "$" + r.price_per_sqft; $("#conf").textContent = r.confidence + "%";
      $("#vsn").textContent = (r.vs_neighborhood_pct >= 0 ? "+" : "") + r.vs_neighborhood_pct + "%"; $("#vsn").style.color = r.vs_neighborhood_pct >= 0 ? "var(--green)" : "var(--red)";
      $("#whatif").innerHTML = r.what_if.map(w => `<div class="doc"><span>${esc(w.change)}</span><b class="gold">${w.effect >= 0 ? "+" : "−"}${fmt(Math.abs(w.effect))}</b></div>`).join("");
      $("#similar").href = `${page("discover")}?budget=${Math.round(r.price * 1.1 / 5000) * 5000}&nb=${$("#Neighborhood").value}`;
      $("#model").textContent = `Model: ${r.model} · R² ${r.r2}. Educational estimate based on Ames, Iowa sales (2006-2010), not an appraisal.`;
    } catch (e) { toast("Could not reach the prediction API: " + e.message, true); }
  }
  const auto = debounce(run, 350);

  $$(".seg").forEach(g => { const f = g.dataset.f; const mark = v => $$("button", g).forEach(b => b.classList.toggle("on", +b.dataset.v === v)); mark(state[f]);
    $$("button", g).forEach(b => b.onclick = () => { state[f] = +b.dataset.v; mark(state[f]); auto(); }); });
  $("#form").addEventListener("input", auto);

  $("#report").onclick = async e => {
    const b = e.target; b.disabled = true;
    try { const r = await PM.request("/api/report", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body()) });
      const url = URL.createObjectURL(await r.blob()), a = Object.assign(document.createElement("a"), { href: url, download: "propmind-report.pdf" }); a.click(); URL.revokeObjectURL(url); toast("Report downloaded");
    } catch (er) { toast(er.message, true); } b.disabled = false;
  };

  const key = () => "hist:" + uk();
  function renderHist() {
    const h = store.get(key(), []);
    $("#hist").innerHTML = h.length ? h.map((x, i) => `<div class="doc"><span>${esc(x.in.Neighborhood)} · ${x.in.GrLivArea.toLocaleString()} sqft · ${x.in.BedroomAbvGr} bed<br><small>${new Date(x.t).toLocaleString()}</small></span><span><b class="gold">${fmt(x.price)}</b> <button class="btn ghost sm" data-i="${i}">Load</button></span></div>`).join("") : '<p class="mut" style="font-size:14px">No saved valuations yet.</p>';
    $$("#hist [data-i]").forEach(b => b.onclick = () => load(h[+b.dataset.i].in));
  }
  function load(v) {
    $("#Neighborhood").value = v.Neighborhood; NUM.forEach(k => $("#" + k).value = v[k]);
    ["BedroomAbvGr", "FullBath", "GarageCars"].forEach(f => { state[f] = v[f]; $$(`.seg[data-f=${f}] button`).forEach(b => b.classList.toggle("on", +b.dataset.v === v[f])); }); run();
  }
  $("#saveH").onclick = () => { if (!last) return; const h = store.get(key(), []); h.unshift({ in: body(), price: last.price, t: Date.now() }); store.set(key(), h.slice(0, 6)); renderHist(); toast("Valuation saved"); };

  const q = new URLSearchParams(location.search); if (q.get("nb")) $("#Neighborhood").value = q.get("nb");
  renderHist(); run();
})();
