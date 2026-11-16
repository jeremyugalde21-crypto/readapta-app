// Readapta: juegos de readaptación con detección de pose por cámara.
// Juego implementado: Reacción lateral.
import { PoseLandmarker, FilesetResolver, DrawingUtils }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const STORE = "readapta.sessions.v1";

// ---------- Navegación ----------
function show(id) {
  document.querySelectorAll(".screen").forEach((s) => (s.hidden = s.id !== id));
  if (id === "s-config") fillPlayers();
  if (id === "s-history") renderHistory();
  window.scrollTo(0, 0);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-go]");
  if (b) show(b.dataset.go);
});

// ---------- Almacenamiento ----------
const loadSessions = () => { try { return JSON.parse(localStorage.getItem(STORE)) || []; } catch { return []; } };
const saveSessions = (arr) => localStorage.setItem(STORE, JSON.stringify(arr));
function fillPlayers() {
  const names = [...new Set(loadSessions().map((s) => s.player))].sort();
  $("players").innerHTML = names.map((n) => `<option value="${esc(n)}">`).join("");
}
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------- Configuración ----------
const PRESETS = {
  inicial:    { duration: 30, sets: 3, rest: 60, waitMin: 2.0, waitMax: 3.5, maxResp: 3.0, distance: "1.0", stimulus: "arrow", falsePct: "0" },
  intermedia: { duration: 30, sets: 4, rest: 45, waitMin: 1.2, waitMax: 2.5, maxResp: 2.0, distance: "1.8", stimulus: "arrow", falsePct: "0" },
  avanzada:   { duration: 30, sets: 5, rest: 40, waitMin: 0.8, waitMax: 1.8, maxResp: 1.5, distance: "2.6", stimulus: "color", falsePct: "15" },
  vuelta:     { duration: 40, sets: 5, rest: 40, waitMin: 0.6, waitMax: 1.5, maxResp: 1.3, distance: "2.6", stimulus: "inverse", falsePct: "25" },
};
const form = $("cfg");
form.preset.addEventListener("change", () => {
  const p = PRESETS[form.preset.value];
  if (p) for (const [k, v] of Object.entries(p)) form[k].value = v;
});
form.addEventListener("input", (e) => { if (e.target.name !== "preset" && e.target.name !== "player" && e.target.name !== "injured") form.preset.value = ""; });
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const f = new FormData(form);
  const cfg = {
    player: f.get("player").trim(), injured: f.get("injured"), preset: f.get("preset"),
    duration: +f.get("duration"), sets: +f.get("sets"), rest: +f.get("rest"),
    waitMin: +f.get("waitMin"), waitMax: Math.max(+f.get("waitMin"), +f.get("waitMax")), maxResp: +f.get("maxResp"),
    distance: +f.get("distance"), sides: f.get("sides"), stimulus: f.get("stimulus"),
    falsePct: +f.get("falsePct"), sound: !!f.get("sound"), camera: f.get("camera"),
  };
  localStorage.setItem("readapta.lastcfg", JSON.stringify(cfg));
  startGame(cfg);
});
// recordar última configuración
try {
  const last = JSON.parse(localStorage.getItem("readapta.lastcfg"));
  if (last) for (const [k, v] of Object.entries(last)) {
    if (!form[k]) continue;
    if (form[k].type === "checkbox") form[k].checked = v; else form[k].value = v;
  }
} catch {}

// ---------- Sonido ----------
let audio = null;
function beep(freq = 880, ms = 120, vol = 0.25) {
  if (!audio || !game?.cfg.sound) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = freq; o.type = "square"; g.gain.value = vol;
  o.connect(g).connect(audio.destination); o.start(); o.stop(audio.currentTime + ms / 1000);
}

// ---------- Modelo y cámara ----------
let landmarker = null, stream = null, wakeLock = null;
async function loadModel() {
  if (landmarker) return landmarker;
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const opts = (delegate) => ({
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate,
    },
    runningMode: "VIDEO", numPoses: 1,
  });
  try { landmarker = await PoseLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch { landmarker = await PoseLandmarker.createFromOptions(fileset, opts("CPU")); }
  return landmarker;
}
const video = $("video"), canvas = $("canvas"), ctx = canvas.getContext("2d");
async function openCamera(facing) {
  stopCamera();
  stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } } });
  video.srcObject = stream; await video.play();
  const mirror = facing === "user";
  video.classList.toggle("mirror", mirror); canvas.classList.toggle("mirror", mirror);
  fit();
}
function stopCamera() { if (stream) stream.getTracks().forEach((t) => t.stop()); stream = null; }
function fit() {
  if (!video.videoWidth) return;
  canvas.width = video.videoWidth; canvas.height = video.videoHeight;
  const scale = Math.min(innerWidth / video.videoWidth, innerHeight / video.videoHeight);
  for (const el of [video, canvas]) { el.style.width = video.videoWidth * scale + "px"; el.style.height = video.videoHeight * scale + "px"; }
}
addEventListener("resize", fit);

// ---------- Pose ----------
const L = { nose: 0, lSh: 11, rSh: 12, lHip: 23, rHip: 24, lAnk: 27, rAnk: 28 };
const vis = (p) => p && (p.visibility ?? 1) > 0.5 && p.x > 0.01 && p.x < 0.99 && p.y > 0.01 && p.y < 0.99;

// ---------- UI del juego ----------
const stim = $("stim"), overlay = $("overlay");
const hud = { status: $("hud-status"), set: $("hud-set"), time: $("hud-time"), score: $("hud-score") };
function setStatus(t, cls = "") { hud.status.textContent = t; hud.status.className = "pill " + cls; }
function say(big, small = "") { overlay.innerHTML = big ? `<div>${big}</div>${small ? `<small>${small}</small>` : ""}` : ""; }
function showStim(kind, target) {
  // kind: arrow | color | inverse | false ; target: +1 derecha, -1 izquierda (perspectiva del atleta)
  stim.className = "show"; stim.style.background = "transparent"; stim.style.color = "#fff";
  if (kind === "false") { stim.classList.add("full"); stim.style.background = "#e53e3ef0"; stim.textContent = ""; return; }
  if (kind === "color") { stim.classList.add("full"); stim.style.background = target > 0 ? "#ecc94bf0" : "#3182cef0"; stim.textContent = ""; return; }
  const shown = kind === "inverse" ? -target : target;
  stim.style.color = kind === "inverse" ? "#f6ad55" : "#fff";
  stim.textContent = shown > 0 ? "→" : "←";
}
function hideStim() { stim.className = ""; stim.style.background = "transparent"; stim.textContent = ""; }
function flash(color) { stim.className = "show full"; stim.style.background = color; stim.textContent = ""; }

// ---------- Juego: Reacción lateral ----------
let game = null;
const now = () => performance.now();
const rand = (a, b) => a + Math.random() * (b - a);

async function startGame(cfg) {
  show("s-play");
  hideStim(); say("Preparando…", "Cargando detección y cámara");
  setStatus("Cargando…"); hud.set.textContent = hud.time.textContent = hud.score.textContent = "";
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    if (audio.state === "suspended") await audio.resume();
  } catch {}
  game = { cfg, phase: "loading", trials: [], set: 1, center: null, d: 0, dRaw: 0, lastPoseT: now(), anticip: 0 };
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Este navegador no permite usar la cámara. Ábrelo con https en Safari o Chrome.");
    await loadModel();
    await openCamera(cfg.camera);
    try { wakeLock = await navigator.wakeLock?.request("screen"); } catch {}
  } catch (e) {
    say("No se pudo iniciar", e.name === "NotAllowedError" ? "Permite el acceso a la cámara en Ajustes > Safari > Cámara." : esc(e.message || e));
    return;
  }
  enterCalib();
  requestAnimationFrame(tick);
}

function enterCalib() {
  Object.assign(game, { phase: "calib", calibSamples: [], calibStart: null });
  say("Colócate en el centro", "De pie, quieto y de frente a la cámara");
}

function startSet() {
  Object.assign(game, { phase: "toCenter", setStart: now(), setEnd: now() + game.cfg.duration * 1000, phaseT: now(), centerSince: null });
  say(""); hideStim(); beep(660, 200);
}

function nextTrialWait() {
  const c = game.cfg;
  const isFalse = Math.random() * 100 < c.falsePct;
  let target = c.sides === "izq" ? -1 : c.sides === "der" ? 1 : (Math.random() < 0.5 ? -1 : 1);
  Object.assign(game, { phase: "waiting", stimAt: now() + rand(c.waitMin, c.waitMax) * 1000, trial: { set: game.set, type: isFalse ? "false" : "real", target } });
}

function finishTrial(outcome, extra = {}) {
  const t = Object.assign(game.trial, { outcome }, extra);
  game.trials.push(t);
  const good = outcome === "hit" || outcome === "inhibit";
  flash(good ? "#38a169cc" : "#e53e3ecc");
  beep(good ? 1200 : 220, good ? 90 : 250);
  Object.assign(game, { phase: "feedback", phaseT: now() });
}

function endSet() {
  if (game.set >= game.cfg.sets) return endGame();
  game.set++;
  Object.assign(game, { phase: "rest", restEnd: now() + game.cfg.rest * 1000 });
  hideStim();
}

function endGame(aborted = false) {
  if (!game || game.phase === "done") return;
  game.phase = "done";
  stopCamera(); hideStim(); say("");
  try { wakeLock?.release(); } catch {}
  const session = buildSession(game, aborted);
  if (session.summary.realTrials + session.summary.falseTrials > 0) {
    const all = loadSessions(); all.push(session); saveSessions(all);
  }
  renderResults(session);
  show("s-results");
}
$("btn-stop").onclick = () => endGame(true);

function tick() {
  if (!game || game.phase === "done") return;
  const t = now();
  // --- detección ---
  let pose = null;
  if (video.readyState >= 2) {
    const res = landmarker.detectForVideo(video, t);
    pose = res.landmarks?.[0] || null;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let full = false;
  if (pose) {
    game.lastPoseT = t;
    const du = new DrawingUtils(ctx);
    du.drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, { color: "#4fd1c5", lineWidth: 4 });
    full = vis(pose[L.nose]) && vis(pose[L.lAnk]) && vis(pose[L.rAnk]);
    const hipX = (pose[L.lHip].x + pose[L.rHip].x) / 2;
    const shW = Math.abs(pose[L.lSh].x - pose[L.rSh].x);
    game.hipX = hipX; game.shW = shW;
    if (game.center) {
      // desplazamiento en anchos de hombro, positivo = derecha del atleta
      const raw = -(hipX - game.center.x) / game.center.shW;
      game.d = game.d * 0.4 + raw * 0.6;
      drawLanes();
    }
    setStatus(full ? "Cuerpo completo" : "No se ve el cuerpo entero", full ? "ok" : "warn");
  } else {
    setStatus("No se detecta a nadie", "bad");
  }
  const lost = t - game.lastPoseT > 1000;

  // --- máquina de estados ---
  const c = game.cfg, d = game.d;
  switch (game.phase) {
    case "calib": {
      if (pose && full) {
        game.calibStart ??= t;
        game.calibSamples.push({ x: game.hipX, w: game.shW });
        const left = Math.ceil((1500 - (t - game.calibStart)) / 1000);
        say("Quieto…", `Calibrando ${Math.max(left, 0)}`);
        if (t - game.calibStart > 1500) {
          const xs = game.calibSamples, n = xs.length;
          const x = xs.reduce((a, s) => a + s.x, 0) / n, w = xs.reduce((a, s) => a + s.w, 0) / n;
          const spread = Math.max(...xs.map((s) => Math.abs(s.x - x)));
          if (w < 0.03 || spread > 0.4 * w) { game.calibSamples = []; game.calibStart = null; say("Quieto, por favor", "Vuelve a intentarlo"); break; }
          game.center = { x, shW: w }; game.d = 0;
          const reach = c.distance * w;
          game.calibWarn = x - reach < 0.03 || x + reach > 0.97;
          Object.assign(game, { phase: "countdown", countEnd: t + 3000 });
        }
      } else {
        game.calibSamples = []; game.calibStart = null;
        say("Colócate en el centro", pose ? "Aléjate o baja la cámara: deben verse cabeza y pies" : "De pie, de frente a la cámara");
      }
      break;
    }
    case "countdown": {
      const s = Math.ceil((game.countEnd - t) / 1000);
      if (s !== game.lastCount) { game.lastCount = s; if (s > 0) beep(520, 90); }
      say(s > 0 ? String(s) : "¡Ya!", game.calibWarn ? "Aviso: la línea de llegada queda cerca del borde. Aleja la cámara o usa una distancia más corta." : `Serie ${game.set} de ${c.sets}`);
      if (t >= game.countEnd) startSet();
      break;
    }
    case "rest": {
      const s = Math.ceil((game.restEnd - t) / 1000);
      say(`Descanso ${s}`, `Siguiente: serie ${game.set} de ${c.sets}`);
      if (s <= 3 && s !== game.lastCount) { game.lastCount = s; if (s > 0) beep(520, 90); }
      if (t >= game.restEnd) startSet();
      break;
    }
    case "toCenter": {
      if (t >= game.setEnd) { endSet(); break; }
      if (lost) { say("No te veo", "Vuelve al centro"); game.centerSince = null; break; }
      if (Math.abs(d) < 0.5) {
        game.centerSince ??= t;
        say("");
        if (t - game.centerSince > 300) nextTrialWait();
      } else { game.centerSince = null; say("Vuelve al centro"); }
      break;
    }
    case "waiting": {
      if (t >= game.setEnd) { endSet(); break; }
      if (Math.abs(d) > 0.8) { // se movió antes del estímulo
        game.anticip++; game.trials.push({ set: game.set, type: "anticip", outcome: "anticip" });
        say("¡Espera la señal!"); beep(220, 200);
        Object.assign(game, { phase: "toCenter", centerSince: null }); break;
      }
      if (t >= game.stimAt) {
        const kind = game.trial.type === "false" ? "false" : c.stimulus;
        showStim(kind, game.trial.target);
        beep(kind === "false" ? 300 : 980, 110);
        Object.assign(game, { phase: "stim", t0: t, d0: d, moveT: null, moveDir: 0 });
      }
      break;
    }
    case "stim": {
      const el = t - game.t0, tr = game.trial;
      if (game.moveT === null && Math.abs(d - game.d0) > 0.3) { game.moveT = el; game.moveDir = Math.sign(d - game.d0); }
      if (tr.type === "false") {
        if (Math.abs(d) >= c.distance * 0.6) finishTrial("error", { rt: game.moveT });
        else if (el >= c.maxResp * 1000) finishTrial("inhibit");
      } else {
        if (d * tr.target >= c.distance) finishTrial("hit", { rt: game.moveT ?? el, mt: el, corrected: game.moveDir !== 0 && game.moveDir !== tr.target });
        else if (-d * tr.target >= c.distance) finishTrial("error", { rt: game.moveT });
        else if (el >= c.maxResp * 1000) finishTrial("miss", { rt: game.moveT });
      }
      break;
    }
    case "feedback": {
      if (t - game.phaseT > 350) { hideStim(); Object.assign(game, { phase: "toCenter", centerSince: null }); }
      break;
    }
  }
  // --- HUD ---
  if (["toCenter", "waiting", "stim", "feedback"].includes(game.phase)) {
    hud.set.textContent = `Serie ${game.set}/${c.sets}`;
    hud.time.textContent = `${Math.max(0, Math.ceil((game.setEnd - t) / 1000))} s`;
    const st = game.trials.filter((x) => x.set === game.set);
    hud.score.textContent = `✓ ${st.filter((x) => x.outcome === "hit" || x.outcome === "inhibit").length}  ✗ ${st.filter((x) => ["error", "miss", "anticip"].includes(x.outcome)).length}`;
  }
  requestAnimationFrame(tick);
}

function drawLanes() {
  const { x, shW } = game.center, W = canvas.width, H = canvas.height, reach = game.cfg.distance * shW;
  ctx.lineWidth = 4;
  ctx.setLineDash([14, 12]); ctx.strokeStyle = "#ffffff66";
  line(x * W);
  ctx.setLineDash([]); ctx.strokeStyle = "#f6ad55cc";
  line((x - reach) * W); line((x + reach) * W);
  // posición actual de la cadera
  ctx.fillStyle = "#f6e05e";
  ctx.beginPath(); ctx.arc(game.hipX * W, H * 0.92, 14, 0, Math.PI * 2); ctx.fill();
  function line(px) { ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke(); }
}

// ---------- Resultados ----------
const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);
const ms = (v) => (v == null ? "–" : `${Math.round(v)} ms`);
const pct = (a, b) => (b ? Math.round((100 * a) / b) : null);

function summarize(trials) {
  const real = trials.filter((t) => t.type === "real"), fals = trials.filter((t) => t.type === "false");
  const hits = real.filter((t) => t.outcome === "hit");
  const side = (s) => hits.filter((t) => t.target === s);
  const mtL = mean(side(-1).map((t) => t.mt)), mtR = mean(side(1).map((t) => t.mt));
  return {
    realTrials: real.length, hits: hits.length, errors: real.filter((t) => t.outcome === "error").length,
    misses: real.filter((t) => t.outcome === "miss").length, corrected: hits.filter((t) => t.corrected).length,
    hitPct: pct(hits.length, real.length),
    rt: mean(hits.map((t) => t.rt)), mt: mean(hits.map((t) => t.mt)),
    rtL: mean(side(-1).map((t) => t.rt)), rtR: mean(side(1).map((t) => t.rt)), mtL, mtR,
    falseTrials: fals.length, inhibitPct: pct(fals.filter((t) => t.outcome === "inhibit").length, fals.length),
    anticip: trials.filter((t) => t.type === "anticip").length,
  };
}
function asymmetry(s, injured) {
  if (s.mtL == null || s.mtR == null) return null;
  if (injured === "izq") return { v: (100 * (s.mtL - s.mtR)) / s.mtR, label: "Lesionado (izq.) vs sano" };
  if (injured === "der") return { v: (100 * (s.mtR - s.mtL)) / s.mtL, label: "Lesionado (der.) vs sano" };
  return { v: (100 * Math.abs(s.mtL - s.mtR)) / Math.min(s.mtL, s.mtR), label: "Diferencia entre lados" };
}
function buildSession(g, aborted) {
  const perSet = [];
  for (let i = 1; i <= g.set; i++) {
    const tr = g.trials.filter((t) => t.set === i);
    if (tr.length) perSet.push({ set: i, ...summarize(tr) });
  }
  return { id: Date.now(), date: new Date().toISOString(), game: "reaccion-lateral", player: g.cfg.player, injured: g.cfg.injured,
    cfg: g.cfg, aborted, summary: summarize(g.trials), perSet, trials: g.trials };
}
function renderResults(s) {
  const m = s.summary, a = asymmetry(m, s.injured);
  $("results").innerHTML = `
    <p class="muted">${esc(s.player)} · ${new Date(s.date).toLocaleString("es")}${s.aborted ? " · terminada antes" : ""}</p>
    <div class="stats">
      <div class="stat"><b>${m.hitPct ?? "–"}${m.hitPct != null ? " %" : ""}</b><span>Aciertos (${m.hits}/${m.realTrials})</span></div>
      <div class="stat"><b>${ms(m.rt)}</b><span>Tiempo de reacción medio</span></div>
      <div class="stat"><b>${ms(m.mt)}</b><span>Tiempo hasta la línea</span></div>
      <div class="stat"><b>${a ? `${a.v > 0 && s.injured ? "+" : ""}${a.v.toFixed(0)} %` : "–"}</b><span>${a ? a.label : "Asimetría"}</span></div>
    </div>
    <table>
      <tr><th></th><th>Izquierda</th><th>Derecha</th></tr>
      <tr><td>Reacción</td><td>${ms(m.rtL)}</td><td>${ms(m.rtR)}</td></tr>
      <tr><td>Llegada</td><td>${ms(m.mtL)}</td><td>${ms(m.mtR)}</td></tr>
    </table>
    <h3>Por serie</h3>
    <table>
      <tr><th>Serie</th><th>Aciertos</th><th>Reacción</th><th>Llegada</th><th>Err.</th><th>Sin resp.</th></tr>
      ${s.perSet.map((p) => `<tr><td>${p.set}</td><td>${p.hitPct ?? "–"}${p.hitPct != null ? " %" : ""}</td><td>${ms(p.rt)}</td><td>${ms(p.mt)}</td><td>${p.errors}</td><td>${p.misses}</td></tr>`).join("")}
    </table>
    <p class="note">Errores de dirección: ${m.errors} · Sin respuesta a tiempo: ${m.misses} · Salidas anticipadas: ${m.anticip} · Corrigió dirección: ${m.corrected}${m.falseTrials ? ` · Inhibición en falsos: ${m.inhibitPct} % (${m.falseTrials})` : ""}</p>
    <p class="note">Medidas orientativas con una sola cámara. Úsalas para comparar al mismo jugador con el mismo montaje.</p>`;
}

// ---------- Historial ----------
function renderHistory() {
  const all = loadSessions();
  const names = [...new Set(all.map((s) => s.player))].sort();
  const sel = $("hist-player"), cur = sel.value;
  sel.innerHTML = `<option value="">Todos</option>` + names.map((n) => `<option ${n === cur ? "selected" : ""}>${esc(n)}</option>`).join("");
  const rows = all.filter((s) => !sel.value || s.player === sel.value).sort((a, b) => b.id - a.id);
  if (!rows.length) { $("history").innerHTML = `<p class="empty">Aún no hay sesiones guardadas.</p>`; return; }
  $("history").innerHTML = `<table>
    <tr><th>Fecha</th><th>Jugador</th><th>Aciertos</th><th>Reacción</th><th>Llegada I/D</th><th>Asim.</th><th></th></tr>
    ${rows.map((s) => { const m = s.summary, a = asymmetry(m, s.injured); return `<tr>
      <td>${new Date(s.date).toLocaleDateString("es")}</td><td>${esc(s.player)}</td>
      <td>${m.hitPct ?? "–"}${m.hitPct != null ? " %" : ""}</td><td>${ms(m.rt)}</td>
      <td>${m.mtL ? Math.round(m.mtL) : "–"} / ${m.mtR ? Math.round(m.mtR) : "–"}</td>
      <td>${a ? a.v.toFixed(0) + " %" : "–"}</td>
      <td><button class="del" data-del="${s.id}" title="Borrar">×</button></td></tr>`; }).join("")}
  </table>`;
}
$("hist-player").addEventListener("change", renderHistory);
$("history").addEventListener("click", (e) => {
  const id = e.target.closest("[data-del]")?.dataset.del;
  if (id && confirm("¿Borrar esta sesión?")) { saveSessions(loadSessions().filter((s) => String(s.id) !== id)); renderHistory(); }
});
$("btn-csv").onclick = () => {
  const head = ["fecha", "jugador", "lado_lesionado", "juego", "series", "aciertos_pct", "aciertos", "intentos", "errores", "sin_respuesta",
    "anticipaciones", "reaccion_ms", "llegada_ms", "llegada_izq_ms", "llegada_der_ms", "asimetria_pct", "inhibicion_pct", "distancia", "preset"];
  const r = (v) => (v == null ? "" : typeof v === "number" ? Math.round(v * 10) / 10 : `"${String(v).replace(/"/g, '""')}"`);
  const lines = loadSessions().map((s) => { const m = s.summary, a = asymmetry(m, s.injured); return [s.date, s.player, s.injured, s.game, s.perSet.length,
    m.hitPct, m.hits, m.realTrials, m.errors, m.misses, m.anticip, m.rt, m.mt, m.mtL, m.mtR, a?.v, m.inhibitPct, s.cfg.distance, s.cfg.preset].map(r).join(","); });
  const blob = new Blob(["﻿" + [head.join(","), ...lines].join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const aEl = Object.assign(document.createElement("a"), { href: url, download: `readapta-${new Date().toISOString().slice(0, 10)}.csv` });
  document.body.appendChild(aEl); aEl.click(); aEl.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000);
};

// Debug hooks for automated tests
window.__readapta = { get game() { return game; }, summarize, PRESETS };
