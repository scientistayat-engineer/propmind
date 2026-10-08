(() => {
  const { $, get, post, esc, rich, toast, request } = PM;
  async function docs() {
    try { const r = await get("/api/rag/docs"); $("#docs").innerHTML = r.documents.map(d => `<div class="doc"><span>📄 ${esc(d.title)}<br><small>${d.words.toLocaleString()} words · ${d.chunks} passages</small></span><small class="gold">indexed ✓</small></div>`).join(""); }
    catch { $("#docs").innerHTML = '<div class="empty">Could not load documents. Is the API running?</div>'; }
  }
  async function ask(q) {
    q = (q || $("#q").value).trim(); if (!q) return; $("#q").value = q; const b = $("#ask"); b.disabled = true; $("#ans").innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
    try {
      const r = await post("/api/rag/ask", { question: q });
      $("#ans").innerHTML = `<div class="m a" style="max-width:100%">${rich(r.answer)}${r.mode === "offline" ? "" : ""}<small>${r.mode === "groq" ? "Answered by Groq AI from the passages below" : r.mode === "none" ? "No relevant passage found" : "Offline mode: showing the best matching passage"}</small></div>` +
        (r.sources.length ? `<div class="tag" style="margin:16px 0 4px">Sources</div>` + r.sources.map(s => `<div class="src"><b>[${s.n}] ${esc(s.doc)}</b> · ${esc(s.section)} <span class="mut">(relevance ${s.score})</span><br>${esc(s.text.length > 300 ? s.text.slice(0, 300) + "…" : s.text)}</div>`).join("") : "");
    } catch (e) { $("#ans").innerHTML = `<div class="empty">${esc(e.message)}</div>`; } b.disabled = false;
  }
  $("#ask").onclick = () => ask(); $("#q").onkeydown = e => { if (e.key === "Enter") ask(); };
  $("#sugg").onclick = e => { if (e.target.classList.contains("chip")) ask(e.target.textContent); };
  const drop = $("#drop"), file = $("#file");
  async function upload(f) {
    if (!f) return; const fd = new FormData(); fd.append("file", f); drop.innerHTML = "Indexing " + esc(f.name) + "…";
    try { await request("/api/rag/upload", { method: "POST", body: fd }); toast("Document indexed"); await docs(); } catch (e) { toast(e.message, true); }
    drop.innerHTML = '<div style="font-size:26px">⬆</div>Drop a PDF, TXT or MD file<br><small>or click to upload (max 3 MB)</small>';
  }
  drop.onclick = () => file.click(); file.onchange = () => upload(file.files[0]);
  ["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => upload(e.dataTransfer.files[0]));
  docs();
})();
