(() => {
  const { $, post, store, uk, rich, toast } = PM;
  const key = () => "chat:" + uk(); let hist = store.get(key(), []);
  const WELCOME = "Hello! I'm PropMind, your AI real estate advisor. I can compare neighborhoods, explain prices and walk you through buying decisions. What would you like to know?";
  function add(role, text, meta) { const d = document.createElement("div"); d.className = "m " + (role === "user" ? "u" : "a"); d.innerHTML = rich(text) + (meta ? `<small>${meta}</small>` : ""); $("#msgs").append(d); $("#msgs").scrollTop = 1e6; return d; }
  function render() { $("#msgs").innerHTML = ""; add("assistant", WELCOME); hist.forEach(m => add(m.role, m.content)); }
  async function send(text) {
    text = (text || $("#in").value).trim(); if (!text) return; $("#in").value = ""; add("user", text);
    const t = document.createElement("div"); t.className = "m a typing"; t.innerHTML = "<span></span><span></span><span></span>"; $("#msgs").append(t); $("#msgs").scrollTop = 1e6; $("#send").disabled = true;
    try {
      const r = await post("/api/chat", { message: text, history: hist.slice(-8) }); t.remove();
      add("assistant", r.answer, r.mode === "groq" ? "Groq AI" : "Offline mode"); $("#mode").textContent = r.mode === "groq" ? "● Groq AI connected" : "○ Offline mode";
      hist.push({ role: "user", content: text }, { role: "assistant", content: r.answer }); hist = hist.slice(-40); store.set(key(), hist);
    } catch (e) { t.remove(); toast(e.message, true); } $("#send").disabled = false;
  }
  $("#send").onclick = () => send(); $("#in").onkeydown = e => { if (e.key === "Enter") send(); };
  $("#sugg").onclick = e => { if (e.target.classList.contains("chip")) send(e.target.textContent); };
  $("#clear").onclick = () => { hist = []; store.set(key(), hist); render(); };
  render();
})();
