/* PropMind shared client: API, auth, favourites, compare, UI helpers */
const PM = (() => {
  const cfg = window.PM_CONFIG || {}, API = cfg.api || "", PG = cfg.pages || {};
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const fmt = n => "$" + Math.round(n).toLocaleString("en-US");
  const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const store = {
    get(k, d) { try { const v = JSON.parse(localStorage.getItem("pm:" + k)); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem("pm:" + k, JSON.stringify(v)); } catch {} }
  };

  async function request(path, opts) {
    const r = await fetch(API + path, opts);
    if (!r.ok) {
      let m = "Something went wrong";
      try { const j = await r.json(); m = typeof j.detail === "string" ? j.detail : "Please check your inputs"; } catch {}
      throw new Error(m);
    }
    return r;
  }
  const get = p => request(p).then(r => r.json());
  const post = (p, body) => request(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then(r => r.json());
  const page = n => PG[n] || "/";

  function toast(msg, err) {
    const t = document.createElement("div"); t.className = "toast" + (err ? " err" : ""); t.textContent = msg;
    $("#toasts").append(t); setTimeout(() => t.remove(), 3600);
  }
  const debounce = (fn, ms = 400) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  /* ---------- auth (demo accounts stored in this browser) ---------- */
  async function sha(s) { const b = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(s)); return [...new Uint8Array(b)].map(x => x.toString(16).padStart(2, "0")).join(""); }
  const user = () => store.get("session", null);
  const uk = () => (user() ? user().email : "guest");
  async function register(name, email, pw) {
    email = email.trim().toLowerCase(); const users = store.get("users", {});
    if (!name.trim() || !email.includes("@") || pw.length < 6) throw new Error("Enter a name, valid email and a password of 6+ characters");
    if (users[email]) throw new Error("This email is already registered");
    users[email] = { name: name.trim(), hash: await sha(pw) }; store.set("users", users);
    store.set("session", { name: name.trim(), email });
  }
  async function login(email, pw) {
    email = email.trim().toLowerCase(); const u = store.get("users", {})[email];
    if (!u || u.hash !== await sha(pw)) throw new Error("Incorrect email or password");
    store.set("session", { name: u.name, email });
  }
  const logout = () => { localStorage.removeItem("pm:session"); location.reload(); };

  /* ---------- favourites + compare ---------- */
  const favs = () => store.get("favs:" + uk(), []);
  function toggleFav(id) {
    let f = favs(); const on = !f.includes(id);
    f = on ? [...f, id] : f.filter(x => x !== id); store.set("favs:" + uk(), f); renderAcct();
    toast(on ? "Saved to your shortlist" : "Removed from shortlist"); return on;
  }
  let cmp = [];
  const compare = { get: () => cmp, has: id => cmp.includes(id),
    toggle(id) { if (cmp.includes(id)) cmp = cmp.filter(x => x !== id); else if (cmp.length >= 3) { toast("You can compare up to 3 homes", true); return false; } else cmp.push(id); document.dispatchEvent(new Event("cmp")); return true; },
    clear() { cmp = []; document.dispatchEvent(new Event("cmp")); } };

  /* ---------- modal ---------- */
  function modal(html, wide) {
    const ov = document.createElement("div"); ov.className = "ov show";
    ov.innerHTML = `<div class="modal ${wide ? "wide" : ""}"><button class="x" aria-label="Close">×</button>${html}</div>`;
    const close = () => ov.remove(); ov.onclick = e => { if (e.target === ov) close(); }; $(".x", ov).onclick = close;
    document.body.append(ov); return { el: ov, close };
  }

  function authModal(mode = "login") {
    const reg = mode === "register";
    const m = modal(`<h3 class="gold" style="margin-bottom:6px">${reg ? "Create your account" : "Welcome back"}</h3>
      <p class="mut" style="font-size:13px;margin-bottom:16px">${reg ? "Save shortlists and valuation history." : "Sign in to open your saved homes."}</p>
      ${reg ? '<label>Full name</label><input type="text" id="aN" autocomplete="name">' : ""}
      <label>Email</label><input type="email" id="aE" autocomplete="email"><label>Password</label><input type="password" id="aP" autocomplete="${reg ? "new-password" : "current-password"}">
      <button class="btn" style="width:100%;justify-content:center;margin-top:20px" id="aGo">${reg ? "Create account" : "Sign in"}</button>
      <p class="mut" style="font-size:13px;margin-top:14px;text-align:center">${reg ? "Have an account?" : "New to PropMind?"} <a href="#" id="aSw" class="gold">${reg ? "Sign in" : "Create account"}</a></p>
      <p class="disc">Demo accounts are stored only in this browser.</p>`);
    $("#aSw", m.el).onclick = e => { e.preventDefault(); m.close(); authModal(reg ? "login" : "register"); };
    $("#aGo", m.el).onclick = async () => {
      try { reg ? await register($("#aN", m.el).value, $("#aE", m.el).value, $("#aP", m.el).value) : await login($("#aE", m.el).value, $("#aP", m.el).value); location.reload(); }
      catch (e) { toast(e.message, true); }
    };
  }

  function renderAcct() {
    const u = user(), n = favs().length, box = $("#acct");
    if (!box) return;
    box.innerHTML = u
      ? `<a class="btn ghost sm" href="${page("discover")}?saved=1">♥ Saved<span class="badge">${n}</span></a><div class="av" id="avBtn" title="${esc(u.email)}">${esc(u.name[0].toUpperCase())}</div>`
      : `<a class="btn ghost sm" href="${page("discover")}?saved=1">♥<span class="badge">${n}</span></a><button class="btn sm" id="siBtn">Sign in</button>`;
    if (u) $("#avBtn").onclick = () => { const m = modal(`<h3 class="gold">${esc(u.name)}</h3><p class="mut" style="margin:6px 0 18px">${esc(u.email)}</p><button class="btn ghost" id="lo" style="width:100%;justify-content:center">Sign out</button>`); $("#lo", m.el).onclick = logout; };
    else $("#siBtn").onclick = () => authModal("login");
  }

  /* ---------- UI effects ---------- */
  function countUp(el) {
    const n = +el.dataset.count, s = el.dataset.suffix || "", d = +(el.dataset.dec || 0), t0 = performance.now();
    (function f(x) { const k = Math.min((x - t0) / 1400, 1), v = n * (1 - Math.pow(1 - k, 3)); el.textContent = v.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d }) + s; if (k < 1) requestAnimationFrame(f); })(t0);
  }
  function initFx() {
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); if (e.target.dataset.count) countUp(e.target); io.unobserve(e.target); } }), { threshold: .15 });
    $$(".rv,[data-count]").forEach(e => io.observe(e));
    document.addEventListener("mousemove", e => { const c = e.target.closest && e.target.closest(".card"); if (c) { const r = c.getBoundingClientRect(); c.style.setProperty("--x", e.clientX - r.left + "px"); c.style.setProperty("--y", e.clientY - r.top + "px"); } });
  }
  function rich(t) {   // tiny, safe markdown: **bold**, bullets, [n] citations, line breaks
    let h = esc(t).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>").replace(/\[(\d)\]/g, '<span class="cite">$1</span>');
    const lines = h.split("\n"), out = []; let ul = false;
    for (const l of lines) { if (/^\s*[-•*] /.test(l)) { if (!ul) { out.push("<ul>"); ul = true; } out.push("<li>" + l.replace(/^\s*[-•*] /, "") + "</li>"); } else { if (ul) { out.push("</ul>"); ul = false; } out.push(l + "<br>"); } }
    if (ul) out.push("</ul>"); return out.join("").replace(/(<br>)+$/, "");
  }

  document.addEventListener("DOMContentLoaded", () => {
    renderAcct(); initFx();
    $("#burger").onclick = () => $("#links").classList.toggle("open");
  });
  return { $, $$, fmt, esc, store, get, post, request, page, API, toast, debounce, modal, user, uk, favs, toggleFav, compare, rich, countUp, initFx };
})();
