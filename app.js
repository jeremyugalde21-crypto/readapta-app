// ReadAPP: juegos de readaptación con detección de pose por cámara.
// Juegos: Reacción lateral, Dianas, Equilibrio.
import { PoseLandmarker, FilesetResolver, DrawingUtils }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const $ = (id) => document.getElementById(id);
const STORE = "readapta.sessions.v1";
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const now = () => performance.now();
const rand = (a, b) => a + Math.random() * (b - a);
const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : null);
const ms = (v) => (v == null ? "–" : `${Math.round(v)} ms`);
const pct = (a, b) => (b ? Math.round((100 * a) / b) : null);
const pctTxt = (v) => (v == null ? "–" : `${Math.round(v)} %`);
const sideName = (s) => (s > 0 ? "der" : "izq");

// Landmarks MediaPipe Pose (izquierda/derecha = del atleta)
const L = { nose: 0, lSh: 11, rSh: 12, lWr: 15, rWr: 16, lHip: 23, rHip: 24, lKnee: 25, rKnee: 26, lAnk: 27, rAnk: 28, lFoot: 31, rFoot: 32 };
const vis = (p) => p && (p.visibility ?? 1) > 0.5 && p.x > 0.01 && p.x < 0.99 && p.y > 0.01 && p.y < 0.99;

// ---------- Navegación ----------
function show(id) {
  document.querySelectorAll(".screen").forEach((s) => (s.hidden = s.id !== id));
  if (id === "s-config" || id === "s-eval") fillPlayers();
  if (id === "s-history") renderHistory();
  window.scrollTo(0, 0);
}
document.addEventListener("click", (e) => {
  const g = e.target.closest("[data-game]:not(fieldset)");
  if (g) return openConfig(g.dataset.game);
  const b = e.target.closest("[data-go]");
  if (b) show(b.dataset.go);
});

// ---------- Almacenamiento ----------
const loadSessions = () => {
  try {
    return (JSON.parse(localStorage.getItem(STORE)) || []).map((s) => (s.game === "reaccion-lateral" ? { ...s, game: "lateral" } : s));
  } catch { return []; }
};
const saveSessions = (arr) => localStorage.setItem(STORE, JSON.stringify(arr));
function fillPlayers() {
  const names = [...new Set(loadSessions().map((s) => s.player))].sort();
  $("players").innerHTML = names.map((n) => `<option value="${esc(n)}">`).join("");
}

// ---------- Sonido ----------
let audio = null;
function beep(freq = 880, dur = 120, vol = 0.25) {
  if (!audio || !game?.cfg.sound) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.frequency.value = freq; o.type = "square"; g.gain.value = vol;
  o.connect(g).connect(audio.destination); o.start(); o.stop(audio.currentTime + dur / 1000);
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
let mirrored = false;
async function openCamera(facing) {
  stopCamera();
  stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 720 } } });
  video.srcObject = stream; await video.play();
  mirrored = facing === "user";
  video.classList.toggle("mirror", mirrored); canvas.classList.toggle("mirror", mirrored);
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
// coordenadas en píxeles del vídeo (para distancias reales sin deformar)
const P = (p) => ({ x: p.x * canvas.width, y: p.y * canvas.height });
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
function text(x, y, txt, size = 40, color = "#fff") {
  ctx.save(); ctx.translate(x, y); if (mirrored) ctx.scale(-1, 1);
  ctx.font = `800 ${size}px -apple-system, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.lineWidth = 6; ctx.strokeStyle = "#000a"; ctx.strokeText(txt, 0, 0); ctx.fillStyle = color; ctx.fillText(txt, 0, 0);
  ctx.restore();
}

// ---------- UI del juego ----------
const stim = $("stim"), overlay = $("overlay");
const hud = { status: $("hud-status"), set: $("hud-set"), time: $("hud-time"), score: $("hud-score") };
function setStatus(t, cls = "") { hud.status.textContent = t; hud.status.className = "pill " + cls; }
function say(big, small = "") {
  const html = big ? `<div>${big}</div>${small ? `<small>${small}</small>` : ""}` : "";
  if (overlay.innerHTML !== html) overlay.innerHTML = html;
}
function hideStim() { stim.className = ""; stim.style.background = "transparent"; stim.textContent = ""; }
function flash(color, band = false) { stim.className = "show full" + (band ? " band" : ""); stim.style.background = color; stim.textContent = ""; }

// ======================================================================
// JUEGO 1: REACCIÓN LATERAL
// ======================================================================
const lateral = {
  name: "Reacción lateral",
  durationLabel: "Duración serie (s)",
  presets: {
    inicial:    { duration: 30, sets: 3, rest: 60, waitMin: 2.0, waitMax: 3.5, maxResp: 3.0, distance: "1.0", stimulus: "arrow", falsePct: "0" },
    intermedia: { duration: 30, sets: 4, rest: 45, waitMin: 1.2, waitMax: 2.5, maxResp: 2.0, distance: "1.8", stimulus: "arrow", falsePct: "0" },
    avanzada:   { duration: 30, sets: 5, rest: 40, waitMin: 0.8, waitMax: 1.8, maxResp: 1.5, distance: "2.6", stimulus: "color", falsePct: "15" },
    vuelta:     { duration: 40, sets: 5, rest: 40, waitMin: 0.6, waitMax: 1.5, maxResp: 1.3, distance: "2.6", stimulus: "inverse", falsePct: "25" },
  },
  presetNames: { inicial: "Inicial: lento, distancia corta", intermedia: "Intermedia: ritmo medio", avanzada: "Avanzada: rápido, distancia larga", vuelta: "Vuelta al juego: decisión y falsos" },
  readCfg: (f) => ({
    waitMin: +f.get("waitMin"), waitMax: Math.max(+f.get("waitMin"), +f.get("waitMax")), maxResp: +f.get("maxResp"),
    distance: +f.get("distance"), sides: f.get("sides"), stimulus: f.get("stimulus"), falsePct: +f.get("falsePct"),
  }),
  calibHint: "De pie, quieto y de frente a la cámara",
  afterCalib(g) {
    const reach = g.cfg.distance * g.center.shW;
    g.calibWarn = g.center.x - reach < 0.03 || g.center.x + reach > 0.97
      ? "Aviso: la línea de llegada queda cerca del borde. Aleja la cámara o usa una distancia más corta." : "";
  },
  startSet(g) { Object.assign(g, { phase: "toCenter", centerSince: null }); },
  update(g, t, pose) {
    const c = g.cfg;
    let d = g.d || 0;
    if (pose) {
      const raw = -(g.hipX - g.center.x) / g.center.shW; // positivo = derecha del atleta
      d = g.d = (g.d || 0) * 0.4 + raw * 0.6;
    }
    const lost = t - g.lastPoseT > 1000;
    switch (g.phase) {
      case "toCenter":
        if (t >= g.setEnd) return endSet();
        if (lost) { say("No te veo", "Vuelve al centro"); g.centerSince = null; break; }
        if (Math.abs(d) < 0.5) {
          g.centerSince ??= t; say("");
          if (t - g.centerSince > 300) {
            const isFalse = Math.random() * 100 < c.falsePct;
            const target = c.sides === "izq" ? -1 : c.sides === "der" ? 1 : (Math.random() < 0.5 ? -1 : 1);
            Object.assign(g, { phase: "waiting", stimAt: t + rand(c.waitMin, c.waitMax) * 1000, trial: { set: g.set, type: isFalse ? "false" : "real", target } });
          }
        } else { g.centerSince = null; say("Vuelve al centro"); }
        break;
      case "waiting":
        if (t >= g.setEnd) return endSet();
        if (Math.abs(d) > 0.8) {
          g.trials.push({ set: g.set, type: "anticip", outcome: "anticip" });
          say("¡Espera la señal!"); beep(220, 200);
          Object.assign(g, { phase: "toCenter", centerSince: null }); break;
        }
        if (t >= g.stimAt) {
          const kind = g.trial.type === "false" ? "false" : c.stimulus;
          lateral.showStim(kind, g.trial.target);
          beep(kind === "false" ? 300 : 980, 110);
          Object.assign(g, { phase: "stim", t0: t, d0: d, moveT: null, moveDir: 0 });
        }
        break;
      case "stim": {
        const el = t - g.t0, tr = g.trial;
        if (g.moveT === null && Math.abs(d - g.d0) > 0.3) { g.moveT = el; g.moveDir = Math.sign(d - g.d0); }
        if (tr.type === "false") {
          if (Math.abs(d) >= c.distance * 0.6) lateral.finish(g, "error", { rt: g.moveT });
          else if (el >= c.maxResp * 1000) lateral.finish(g, "inhibit");
        } else {
          if (d * tr.target >= c.distance) lateral.finish(g, "hit", { rt: g.moveT ?? el, mt: el, corrected: g.moveDir !== 0 && g.moveDir !== tr.target });
          else if (-d * tr.target >= c.distance) lateral.finish(g, "error", { rt: g.moveT });
          else if (el >= c.maxResp * 1000) lateral.finish(g, "miss", { rt: g.moveT });
        }
        break;
      }
      case "feedback":
        if (t - g.phaseT > 350) { hideStim(); Object.assign(g, { phase: "toCenter", centerSince: null }); }
        break;
    }
    if (pose) lateral.draw(g);
    const st = g.trials.filter((x) => x.set === g.set);
    hud.score.textContent = `✓ ${st.filter((x) => x.outcome === "hit" || x.outcome === "inhibit").length}  ✗ ${st.filter((x) => ["error", "miss", "anticip"].includes(x.outcome)).length}`;
  },
  showStim(kind, target) {
    stim.className = "show"; stim.style.background = "transparent"; stim.style.color = "#fff";
    if (kind === "false") { stim.classList.add("full"); stim.style.background = "#e53e3ef0"; stim.textContent = ""; return; }
    if (kind === "color") { stim.classList.add("full"); stim.style.background = target > 0 ? "#ecc94bf0" : "#3182cef0"; stim.textContent = ""; return; }
    const shown = kind === "inverse" ? -target : target;
    stim.style.color = kind === "inverse" ? "#f6ad55" : "#fff";
    stim.textContent = shown > 0 ? "→" : "←";
  },
  finish(g, outcome, extra = {}) {
    g.trials.push(Object.assign(g.trial, { outcome }, extra));
    const good = outcome === "hit" || outcome === "inhibit";
    flash(good ? "#38a169cc" : "#e53e3ecc"); beep(good ? 1200 : 220, good ? 90 : 250);
    Object.assign(g, { phase: "feedback", phaseT: now() });
  },
  draw(g) {
    const { x, shW } = g.center, W = canvas.width, H = canvas.height, reach = g.cfg.distance * shW;
    const line = (px) => { ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke(); };
    ctx.lineWidth = 4; ctx.setLineDash([14, 12]); ctx.strokeStyle = "#ffffff66"; line(x * W);
    ctx.setLineDash([]); ctx.strokeStyle = "#f6ad55cc"; line((x - reach) * W); line((x + reach) * W);
    ctx.fillStyle = "#f6e05e"; ctx.beginPath(); ctx.arc(g.hipX * W, H * 0.92, 14, 0, Math.PI * 2); ctx.fill();
  },
  summarize(trials) {
    const real = trials.filter((t) => t.type === "real"), fals = trials.filter((t) => t.type === "false");
    const hits = real.filter((t) => t.outcome === "hit");
    const side = (s) => hits.filter((t) => t.target === s);
    return {
      realTrials: real.length, hits: hits.length, errors: real.filter((t) => t.outcome === "error").length,
      misses: real.filter((t) => t.outcome === "miss").length, corrected: hits.filter((t) => t.corrected).length,
      hitPct: pct(hits.length, real.length), rt: mean(hits.map((t) => t.rt)), mt: mean(hits.map((t) => t.mt)),
      rtL: mean(side(-1).map((t) => t.rt)), rtR: mean(side(1).map((t) => t.rt)),
      mtL: mean(side(-1).map((t) => t.mt)), mtR: mean(side(1).map((t) => t.mt)),
      falseTrials: fals.length, inhibitPct: pct(fals.filter((t) => t.outcome === "inhibit").length, fals.length),
      anticip: trials.filter((t) => t.type === "anticip").length,
    };
  },
  asym: (m, inj) => timeAsym(m.mtL, m.mtR, inj),
  results(s) {
    const m = s.summary, a = lateral.asym(m, s.injured);
    return `
    <div class="stats">
      ${stat(m.hitPct != null ? m.hitPct + " %" : "–", `Aciertos (${m.hits}/${m.realTrials})`)}
      ${stat(ms(m.rt), "Tiempo de reacción medio")}
      ${stat(ms(m.mt), "Tiempo hasta la línea")}
      ${stat(a ? a.txt : "–", a ? a.label : "Asimetría")}
    </div>
    <table>
      <tr><th></th><th>Izquierda</th><th>Derecha</th></tr>
      <tr><td>Reacción</td><td>${ms(m.rtL)}</td><td>${ms(m.rtR)}</td></tr>
      <tr><td>Llegada</td><td>${ms(m.mtL)}</td><td>${ms(m.mtR)}</td></tr>
    </table>
    <h3>Por serie</h3>
    <table>
      <tr><th>Serie</th><th>Aciertos</th><th>Reacción</th><th>Llegada</th><th>Err.</th><th>Sin resp.</th></tr>
      ${s.perSet.map((p) => `<tr><td>${p.set}</td><td>${pctTxt(p.hitPct)}</td><td>${ms(p.rt)}</td><td>${ms(p.mt)}</td><td>${p.errors}</td><td>${p.misses}</td></tr>`).join("")}
    </table>
    <p class="note">Errores de dirección: ${m.errors} · Sin respuesta a tiempo: ${m.misses} · Salidas anticipadas: ${m.anticip} · Corrigió dirección: ${m.corrected}${m.falseTrials ? ` · Inhibición en falsos: ${m.inhibitPct} % (${m.falseTrials})` : ""}</p>`;
  },
  histMain: (m) => `${pctTxt(m.hitPct)} aciertos · reacción ${ms(m.rt)}`,
  histDetail: (m) => `Llegada I/D: ${m.mtL ? Math.round(m.mtL) : "–"} / ${m.mtR ? Math.round(m.mtR) : "–"} ms`,
};

// ======================================================================
// JUEGO 2: DIANAS
// ======================================================================
const dianas = {
  name: "Dianas",
  durationLabel: "Duración serie (s)",
  presets: {
    inicial:    { duration: 30, sets: 3, rest: 45, dParts: "hand", dHeight: "mid", dReach: "near", dTime: 4, dSize: "0.55", dSeq: "1" },
    intermedia: { duration: 40, sets: 3, rest: 45, dParts: "mix", dHeight: "all", dReach: "near", dTime: 3, dSize: "0.42", dSeq: "1" },
    avanzada:   { duration: 40, sets: 4, rest: 40, dParts: "mix", dHeight: "all", dReach: "far", dTime: 2, dSize: "0.42", dSeq: "1" },
    vuelta:     { duration: 45, sets: 4, rest: 40, dParts: "mix", dHeight: "all", dReach: "far", dTime: 1.6, dSize: "0.3", dSeq: "3" },
  },
  presetNames: { inicial: "Inicial: manos, cerca, dianas grandes", intermedia: "Intermedia: manos y pies", avanzada: "Avanzada: lejos y rápido", vuelta: "Vuelta al juego: secuencias rápidas" },
  readCfg: (f) => ({ parts: f.get("dParts"), side: f.get("dSide"), height: f.get("dHeight"), reach: f.get("dReach"), time: +f.get("dTime"), size: +f.get("dSize"), seq: +f.get("dSeq") }),
  calibHint: "De pie, quieto y de frente a la cámara",
  mirrorView: true,
  afterCalib(g) { g.calibWarn = ""; },
  startSet(g) { Object.assign(g, { phase: "gap", phaseT: now() - 1000 }); },
  spawn(g) {
    const c = g.cfg, C = g.center, n = c.seq;
    const W = canvas.width, H = canvas.height;
    const shWpx = C.shW * W, torso = (C.hipY - C.shY) * H, leg = (C.ankY - C.hipY) * H;
    const r = Math.max(28, c.size * shWpx);
    const targets = [];
    for (let i = 0; i < n; i++) {
      for (let tries = 0; tries < 30; tries++) {
        const part = c.parts === "mix" ? pick(["hand", "foot"]) : c.parts;
        const s = c.side === "izq" ? -1 : c.side === "der" ? 1 : pick([-1, 1]);
        let u, y, zone;
        if (part === "hand") {
          zone = c.height === "all" ? pick(["high", "mid", "low"]) : c.height;
          u = c.reach === "far" ? rand(1.9, 2.7) : rand(0.9, 1.6);
          y = zone === "high" ? C.shY * H - torso * rand(0.4, 0.65) : zone === "mid" ? C.shY * H + torso * rand(-0.15, 0.2) : C.hipY * H + leg * rand(0.25, 0.5);
        } else {
          zone = "floor";
          u = c.reach === "far" ? rand(1.7, 2.5) : rand(0.9, 1.4);
          y = C.ankY * H - leg * rand(0.0, 0.2);
        }
        const x = C.x * W - s * u * shWpx; // positivo = derecha del atleta = izquierda de la imagen
        const tgt = { x, y, r, part, side: s, zone, n: i + 1 };
        const inside = x > r && x < W - r && y > r && y < H - r;
        const overlaps = targets.some((o) => dist(o, tgt) < o.r + r + 10);
        if (inside && !overlaps) { targets.push(tgt); break; }
      }
    }
    if (!targets.length) { g.calibWarn = "No caben las dianas en la imagen: aleja la cámara."; return false; }
    Object.assign(g, { phase: "target", targets, idx: 0, t0: now(), tIdx: now(), timeLimit: c.time * 1000 * targets.length });
    beep(980, 80);
    return true;
  },
  limb(pose, part, side) {
    // side +1 = derecha del atleta
    if (part === "hand") return [pose[side > 0 ? L.rWr : L.lWr]];
    return side > 0 ? [pose[L.rAnk], pose[L.rFoot]] : [pose[L.lAnk], pose[L.lFoot]];
  },
  update(g, t, pose) {
    switch (g.phase) {
      case "gap":
        if (t >= g.setEnd) return endSet();
        if (t - g.phaseT > 400) { say(""); if (!dianas.spawn(g)) say("Aleja la cámara", g.calibWarn); }
        break;
      case "target": {
        const tg = g.targets[g.idx];
        if (pose) {
          const touching = dianas.limb(pose, tg.part, tg.side).some((p) => p && (p.visibility ?? 1) > 0.4 && dist(P(p), tg) < tg.r * 1.1);
          if (touching) {
            g.trials.push({ set: g.set, outcome: "hit", part: tg.part, side: tg.side, zone: tg.zone, t: t - g.tIdx, seq: g.targets.length > 1 });
            beep(1200, 70);
            g.idx++; g.tIdx = t;
            if (g.idx >= g.targets.length) {
              if (g.targets.length > 1) g.seqs.push({ set: g.set, ok: true, t: t - g.t0 });
              flash("#38a16966", true);
              Object.assign(g, { phase: "gap", phaseT: t });
              setTimeout(hideStim, 200);
            }
            break;
          }
        }
        if (t - g.t0 > g.timeLimit) {
          for (let i = g.idx; i < g.targets.length; i++) {
            const m = g.targets[i];
            g.trials.push({ set: g.set, outcome: "miss", part: m.part, side: m.side, zone: m.zone, seq: g.targets.length > 1 });
          }
          if (g.targets.length > 1) g.seqs.push({ set: g.set, ok: false });
          flash("#e53e3e66", true); beep(220, 200);
          setTimeout(hideStim, 250);
          Object.assign(g, { phase: "gap", phaseT: t });
        }
        break;
      }
    }
    dianas.draw(g, t);
    const st = g.trials.filter((x) => x.set === g.set);
    hud.score.textContent = `✓ ${st.filter((x) => x.outcome === "hit").length}  ✗ ${st.filter((x) => x.outcome === "miss").length}`;
  },
  draw(g, t) {
    if (g.phase !== "target") return;
    const left = 1 - (t - g.t0) / g.timeLimit;
    g.targets.forEach((tg, i) => {
      if (i < g.idx) return;
      const col = tg.part === "hand" ? "#3182ce" : "#ed8936";
      const active = i === g.idx;
      ctx.globalAlpha = active ? 1 : 0.55;
      ctx.fillStyle = col + "cc"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 5;
      ctx.beginPath(); ctx.arc(tg.x, tg.y, tg.r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(tg.x, tg.y, tg.r * 0.45, 0, Math.PI * 2); ctx.stroke();
      if (active) { // anillo de tiempo
        ctx.strokeStyle = "#f6e05e"; ctx.lineWidth = 7;
        ctx.beginPath(); ctx.arc(tg.x, tg.y, tg.r + 9, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.max(0, left)); ctx.stroke();
      }
      ctx.globalAlpha = 1;
      text(tg.x, tg.y, g.targets.length > 1 ? String(tg.n) : tg.part === "hand" ? "✋" : "🦶", tg.r * 0.8);
    });
  },
  summarize(trials, seqs = []) {
    const hits = trials.filter((t) => t.outcome === "hit");
    const by = (f) => { const a = trials.filter(f), h = a.filter((t) => t.outcome === "hit"); return { n: a.length, hitPct: pct(h.length, a.length), t: mean(h.filter((x) => !x.seq || true).map((x) => x.t)) }; };
    const zones = {};
    for (const z of ["high", "mid", "low", "floor"]) { const r = by((t) => t.zone === z); if (r.n) zones[z] = r; }
    const L_ = by((t) => t.side < 0), R_ = by((t) => t.side > 0);
    return {
      total: trials.length, hits: hits.length, misses: trials.length - hits.length, hitPct: pct(hits.length, trials.length),
      t: mean(hits.map((x) => x.t)), hitPctL: L_.hitPct, hitPctR: R_.hitPct, tL: L_.t, tR: R_.t,
      zones, seqTotal: seqs.length, seqOk: seqs.filter((s) => s.ok).length, seqT: mean(seqs.filter((s) => s.ok).map((s) => s.t)),
    };
  },
  asym: (m, inj) => timeAsym(m.tL, m.tR, inj),
  results(s) {
    const m = s.summary, a = dianas.asym(m, s.injured);
    const zn = { high: "Altas", mid: "Medias", low: "Bajas", floor: "Pies (suelo)" };
    return `
    <div class="stats">
      ${stat(pctTxt(m.hitPct), `Dianas tocadas (${m.hits}/${m.total})`)}
      ${stat(ms(m.t), "Tiempo medio por diana")}
      ${stat(a ? a.txt : "–", a ? a.label : "Asimetría")}
      ${m.seqTotal ? stat(`${m.seqOk}/${m.seqTotal}`, `Secuencias completas · ${ms(m.seqT)}`) : ""}
    </div>
    <table>
      <tr><th></th><th>Izquierda</th><th>Derecha</th></tr>
      <tr><td>Tocadas</td><td>${pctTxt(m.hitPctL)}</td><td>${pctTxt(m.hitPctR)}</td></tr>
      <tr><td>Tiempo</td><td>${ms(m.tL)}</td><td>${ms(m.tR)}</td></tr>
    </table>
    <h3>Por zona</h3>
    <table>
      <tr><th>Zona</th><th>Dianas</th><th>Tocadas</th><th>Tiempo</th></tr>
      ${Object.entries(m.zones).map(([z, r]) => `<tr><td>${zn[z]}</td><td>${r.n}</td><td>${pctTxt(r.hitPct)}</td><td>${ms(r.t)}</td></tr>`).join("")}
    </table>
    <h3>Por serie</h3>
    <table>
      <tr><th>Serie</th><th>Tocadas</th><th>Tiempo</th><th>Falladas</th></tr>
      ${s.perSet.map((p) => `<tr><td>${p.set}</td><td>${pctTxt(p.hitPct)}</td><td>${ms(p.t)}</td><td>${p.misses}</td></tr>`).join("")}
    </table>`;
  },
  histMain: (m) => `${pctTxt(m.hitPct)} tocadas · ${ms(m.t)}/diana`,
  histDetail: (m) => `Tocadas I/D: ${pctTxt(m.hitPctL)} / ${pctTxt(m.hitPctR)}`,
};

// ======================================================================
// JUEGO 3: EQUILIBRIO
// ======================================================================
const POSTURE = {
  bi: { label: "Bipodal", hint: "De pie sobre los dos pies, quieto" },
  "mono-izq": { label: "Monopodal izquierda", hint: "Apoyado solo en la pierna IZQUIERDA, pie derecho levantado", stance: -1 },
  "mono-der": { label: "Monopodal derecha", hint: "Apoyado solo en la pierna DERECHA, pie izquierdo levantado", stance: 1 },
  squat: { label: "Sentadilla", hint: "Baja a la sentadilla y mantén la posición" },
};
const equilibrio = {
  name: "Equilibrio",
  durationLabel: "Tiempo por serie (s)",
  calibEachSet: true,
  presets: {
    inicial:    { duration: 20, sets: 3, rest: 40, ePosture: "bi", eZone: "0.35", eDual: "none" },
    intermedia: { duration: 30, sets: 4, rest: 40, ePosture: "mono-alt", eZone: "0.35", eDual: "none" },
    avanzada:   { duration: 30, sets: 4, rest: 40, ePosture: "mono-alt", eZone: "0.22", eDual: "none" },
    vuelta:     { duration: 30, sets: 6, rest: 30, ePosture: "mono-alt", eZone: "0.22", eDual: "arms" },
  },
  presetNames: { inicial: "Inicial: bipodal, zona amplia", intermedia: "Intermedia: monopodal alternando", avanzada: "Avanzada: monopodal, zona media", vuelta: "Vuelta al juego: monopodal con doble tarea" },
  readCfg: (f) => ({ posture: f.get("ePosture"), zone: +f.get("eZone"), dual: f.get("eDual") }),
  postureFor(g) {
    const p = g.cfg.posture;
    if (p === "mono-alt") return g.set % 2 === 1 ? "mono-izq" : "mono-der";
    return p;
  },
  calibHintFor(g) { return POSTURE[equilibrio.postureFor(g)].hint; },
  calibCheck(g, pose) {
    const post = POSTURE[equilibrio.postureFor(g)];
    if (!post.stance) return true;
    const st = pose[post.stance > 0 ? L.rAnk : L.lAnk], fr = pose[post.stance > 0 ? L.lAnk : L.rAnk];
    return (st.y - fr.y) * canvas.height > 0.25 * g.shW * canvas.width; // pie libre levantado
  },
  calibFailMsg: "Levanta más el pie libre",
  afterCalib(g, pose) {
    g.calibWarn = "";
    g.posture = equilibrio.postureFor(g);
    const post = POSTURE[g.posture];
    g.ref = { hip: { x: g.center.x * canvas.width, y: g.center.y * canvas.height }, shWpx: g.center.shW * canvas.width };
    if (post.stance) {
      const st = P(pose[post.stance > 0 ? L.rAnk : L.lAnk]), fr = P(pose[post.stance > 0 ? L.lAnk : L.rAnk]);
      g.ref.lift = st.y - fr.y;
    }
  },
  startSet(g) {
    Object.assign(g, { phase: "hold", acc: { tot: 0, inZone: 0, path: 0, dsum: 0, frames: 0, touches: 0, valgus: 0, dualN: 0, dualOk: 0 },
      touching: false, prevOff: null, lastT: now(), nextDual: now() + rand(3000, 5000), dual: null });
  },
  update(g, t, pose) {
    if (g.phase !== "hold") return;
    if (t >= g.setEnd) { equilibrio.closeSet(g); return endSet(); }
    const dt = Math.min(100, t - g.lastT); g.lastT = t;
    const a = g.acc, z = g.cfg.zone, post = POSTURE[g.posture];
    if (pose) {
      const hip = { x: (pose[L.lHip].x + pose[L.rHip].x) / 2 * canvas.width, y: (pose[L.lHip].y + pose[L.rHip].y) / 2 * canvas.height };
      const off = { x: (hip.x - g.ref.hip.x) / g.ref.shWpx, y: (hip.y - g.ref.hip.y) / g.ref.shWpx };
      g.off = g.off ? { x: g.off.x * 0.5 + off.x * 0.5, y: g.off.y * 0.5 + off.y * 0.5 } : off;
      const d = Math.hypot(g.off.x, g.off.y);
      a.tot += dt; a.frames++; a.dsum += d;
      if (d <= z) a.inZone += dt;
      if (g.prevOff) a.path += Math.hypot(g.off.x - g.prevOff.x, g.off.y - g.prevOff.y);
      g.prevOff = { ...g.off };
      // apoyo del pie libre
      if (post.stance && g.ref.lift > 0) {
        const lift = P(pose[post.stance > 0 ? L.rAnk : L.lAnk]).y - P(pose[post.stance > 0 ? L.lAnk : L.rAnk]).y;
        if (!g.touching && lift < 0.3 * g.ref.lift) { g.touching = true; a.touches++; beep(220, 150); }
        else if (g.touching && lift > 0.6 * g.ref.lift) g.touching = false;
      }
      // valgo de rodilla (orientativo, vista frontal)
      const legs = post.stance ? [post.stance] : [-1, 1];
      if (legs.some((s) => equilibrio.valgus(pose, s, g.ref.shWpx) > 0.2)) a.valgus += dt;
      // doble tarea
      if (g.cfg.dual === "arms") {
        if (!g.dual && t >= g.nextDual) {
          g.dual = { side: pick([-1, 1]), t0: t };
          a.dualN++; flash(g.dual.side > 0 ? "#ecc94b" : "#3182ce", true); beep(980, 90);
        } else if (g.dual) {
          const s = g.dual.side;
          const wr = pose[s > 0 ? L.rWr : L.lWr], sh = pose[s > 0 ? L.rSh : L.lSh];
          if (wr.y < sh.y - 0.02) { a.dualOk++; beep(1200, 70); g.dual = null; hideStim(); g.nextDual = t + rand(2500, 4500); }
          else if (t - g.dual.t0 > 2000) { beep(220, 150); g.dual = null; hideStim(); g.nextDual = t + rand(2500, 4500); }
        }
      }
    }
    equilibrio.draw(g);
    say("", "");
    hud.score.textContent = a.tot ? `En zona ${Math.round((100 * a.inZone) / a.tot)} %` : "";
  },
  valgus(pose, s, shWpx) {
    const hip = P(pose[s > 0 ? L.rHip : L.lHip]), knee = P(pose[s > 0 ? L.rKnee : L.lKnee]), ank = P(pose[s > 0 ? L.rAnk : L.lAnk]);
    if (Math.abs(ank.y - hip.y) < 1) return 0;
    const exp = hip.x + (ank.x - hip.x) * ((knee.y - hip.y) / (ank.y - hip.y));
    const mid = (P(pose[L.lHip]).x + P(pose[L.rHip]).x) / 2;
    const inward = Math.sign(mid - hip.x) || 1;
    return ((knee.x - exp) * inward) / shWpx;
  },
  closeSet(g) {
    const a = g.acc, post = POSTURE[g.posture];
    g.trials.push({ set: g.set, posture: g.posture, stance: post.stance || 0,
      inZonePct: a.tot ? (100 * a.inZone) / a.tot : null, sway: a.frames ? (100 * a.dsum) / a.frames : null,
      pathPerS: a.tot ? (100 * a.path) / (a.tot / 1000) : null, touches: a.touches,
      valgusPct: a.tot ? (100 * a.valgus) / a.tot : null, dualN: a.dualN, dualOk: a.dualOk });
  },
  onAbort(g) { if (g.phase === "hold" && g.acc?.tot > 3000) equilibrio.closeSet(g); },
  draw(g) {
    // la zona se dibuja ampliada (radio fijo) para que se vea bien; el punto se escala igual
    const { hip, shWpx } = g.ref, z = 1.1 * shWpx, k = z / g.cfg.zone;
    const off = g.off || { x: 0, y: 0 };
    const d = Math.hypot(off.x, off.y), inZone = d <= g.cfg.zone;
    const sc = Math.min(1.6, d / g.cfg.zone) / (d / g.cfg.zone || 1); // limita el punto a 1,6 veces la zona
    ctx.lineWidth = 6; ctx.strokeStyle = inZone ? "#38a169" : "#e53e3e"; ctx.fillStyle = inZone ? "#38a16933" : "#e53e3e33";
    ctx.beginPath(); ctx.arc(hip.x, hip.y, z, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#f6e05e";
    ctx.beginPath(); ctx.arc(hip.x + off.x * k * sc, hip.y + off.y * k * sc, Math.max(10, z * 0.15), 0, Math.PI * 2); ctx.fill();
    const left = Math.max(0, Math.ceil((g.setEnd - now()) / 1000));
    text(hip.x, hip.y - z - 50, String(left), 70);
    if (g.touching) text(hip.x, hip.y + z + 50, "¡Pie al suelo!", 40, "#feb2b2");
  },
  summarize(trials) {
    const sets = trials.filter((t) => t.inZonePct != null);
    const avg = (f, arr = sets) => mean(arr.map(f).filter((v) => v != null));
    const Ls = sets.filter((t) => t.stance < 0), Rs = sets.filter((t) => t.stance > 0);
    const dn = sets.reduce((s, t) => s + t.dualN, 0), dok = sets.reduce((s, t) => s + t.dualOk, 0);
    return {
      sets: sets.length, inZonePct: avg((t) => t.inZonePct), sway: avg((t) => t.sway), pathPerS: avg((t) => t.pathPerS),
      touches: sets.reduce((s, t) => s + t.touches, 0), valgusPct: avg((t) => t.valgusPct),
      zoneL: Ls.length ? avg((t) => t.inZonePct, Ls) : null, zoneR: Rs.length ? avg((t) => t.inZonePct, Rs) : null,
      touchesL: Ls.reduce((s, t) => s + t.touches, 0), touchesR: Rs.reduce((s, t) => s + t.touches, 0),
      dualN: dn, dualPct: pct(dok, dn),
    };
  },
  asym(m, inj) {
    if (m.zoneL == null || m.zoneR == null) return null;
    if (inj === "izq" || inj === "der") {
      const d = inj === "izq" ? m.zoneL - m.zoneR : m.zoneR - m.zoneL;
      return { v: d, txt: `${d > 0 ? "+" : ""}${Math.round(d)} pts`, label: `En zona: lesionado (${inj}.) vs sano` };
    }
    const d = Math.abs(m.zoneL - m.zoneR);
    return { v: d, txt: `${Math.round(d)} pts`, label: "Diferencia en zona entre piernas" };
  },
  results(s) {
    const m = s.summary, a = equilibrio.asym(m, s.injured), mono = m.zoneL != null || m.zoneR != null;
    return `
    <div class="stats">
      ${stat(pctTxt(m.inZonePct), "Tiempo dentro de la zona")}
      ${stat(m.sway != null ? Math.round(m.sway) : "–", "Oscilación media (% ancho de hombros)")}
      ${mono ? stat(a ? a.txt : "–", a ? a.label : "Asimetría") : stat(pctTxt(m.valgusPct), "Tiempo con rodilla hacia dentro")}
      ${m.dualN ? stat(pctTxt(m.dualPct), `Doble tarea (${m.dualN} estímulos)`) : ""}
    </div>
    ${mono ? `<table>
      <tr><th></th><th>Pierna izq.</th><th>Pierna der.</th></tr>
      <tr><td>En zona</td><td>${pctTxt(m.zoneL)}</td><td>${pctTxt(m.zoneR)}</td></tr>
      <tr><td>Apoyos del pie libre</td><td>${m.zoneL != null ? m.touchesL : "–"}</td><td>${m.zoneR != null ? m.touchesR : "–"}</td></tr>
    </table>` : ""}
    <h3>Por serie</h3>
    <div class="overflow"><table>
      <tr><th>Serie</th><th>Postura</th><th>En zona</th><th>Oscil.</th><th>Apoyos</th><th>Rodilla dentro</th>${m.dualN ? "<th>Doble tarea</th>" : ""}</tr>
      ${s.trials.map((t) => `<tr><td>${t.set}</td><td>${POSTURE[t.posture].label}</td><td>${pctTxt(t.inZonePct)}</td><td>${t.sway != null ? Math.round(t.sway) : "–"}</td><td>${t.stance ? t.touches : "–"}</td><td>${pctTxt(t.valgusPct)}</td>${m.dualN ? `<td>${t.dualOk}/${t.dualN}</td>` : ""}</tr>`).join("")}
    </table></div>
    <p class="note">"Rodilla hacia dentro" es un aviso orientativo desde la vista frontal, no una medición clínica.</p>`;
  },
  histMain: (m) => `${pctTxt(m.inZonePct)} en zona · oscil. ${m.sway != null ? Math.round(m.sway) : "–"}`,
  histDetail: (m) => (m.zoneL != null || m.zoneR != null ? `En zona I/D: ${pctTxt(m.zoneL)} / ${pctTxt(m.zoneR)} · apoyos ${m.touches}` : `Rodilla dentro ${pctTxt(m.valgusPct)}`),
};

// ======================================================================
// JUEGO 4: ATERRIZAJE SUAVE (saltos)
// ======================================================================
const JUMP_POST = {
  bi: { label: "Bipodal", hint: "De pie sobre los dos pies, quieto" },
  "mono-izq": { label: "Pierna izquierda", hint: "Apoyado solo en la pierna IZQUIERDA, pie derecho levantado", stance: -1 },
  "mono-der": { label: "Pierna derecha", hint: "Apoyado solo en la pierna DERECHA, pie izquierdo levantado", stance: 1 },
};
const saltos = {
  name: "Aterrizaje suave",
  durationLabel: "Duración serie (s)",
  calibEachSet: true,
  presets: {
    inicial:    { duration: 30, sets: 3, rest: 60, jPosture: "bi", jMode: "signal", jWaitMin: 2.0, jWaitMax: 3.5 },
    intermedia: { duration: 30, sets: 4, rest: 60, jPosture: "mono-alt", jMode: "signal", jWaitMin: 1.8, jWaitMax: 3.0 },
    avanzada:   { duration: 20, sets: 4, rest: 60, jPosture: "bi", jMode: "cont", jWaitMin: 1.5, jWaitMax: 2.5 },
    vuelta:     { duration: 20, sets: 6, rest: 60, jPosture: "mono-alt", jMode: "cont", jWaitMin: 1.0, jWaitMax: 2.0 },
  },
  presetNames: { inicial: "Inicial: bipodal, salto a la señal", intermedia: "Intermedia: monopodal alternando, a la señal", avanzada: "Avanzada: saltos continuos bipodales", vuelta: "Vuelta al juego: saltos continuos monopodales" },
  readCfg: (f) => ({ posture: f.get("jPosture"), mode: f.get("jMode"), waitMin: +f.get("jWaitMin"), waitMax: Math.max(+f.get("jWaitMin"), +f.get("jWaitMax")) }),
  postureFor(g) { const p = g.cfg.posture; return p === "mono-alt" ? (g.set % 2 === 1 ? "mono-izq" : "mono-der") : p; },
  calibHintFor(g) { return JUMP_POST[saltos.postureFor(g)].hint; },
  calibCheck(g, pose) {
    const post = JUMP_POST[saltos.postureFor(g)];
    if (!post.stance) return true;
    const st = pose[post.stance > 0 ? L.rAnk : L.lAnk], fr = pose[post.stance > 0 ? L.lAnk : L.rAnk];
    return (st.y - fr.y) * canvas.height > 0.25 * g.shW * canvas.width;
  },
  calibFailMsg: "Levanta más el pie libre",
  afterCalib(g, pose) {
    g.calibWarn = "";
    g.posture = saltos.postureFor(g);
    const H = canvas.height, W = canvas.width, post = JUMP_POST[g.posture];
    const ankY = post.stance ? pose[post.stance > 0 ? L.rAnk : L.lAnk].y * H : Math.max(pose[L.lAnk].y, pose[L.rAnk].y) * H;
    g.jref = { floor: ankY, hipY: g.center.y * H, hipX: g.center.x * W, leg: Math.max(20, ankY - g.center.y * H), shWpx: g.center.shW * W };
  },
  footY(g, pose) {
    const post = JUMP_POST[g.posture], H = canvas.height;
    return post.stance ? pose[post.stance > 0 ? L.rAnk : L.lAnk].y * H : Math.max(pose[L.lAnk].y, pose[L.rAnk].y) * H;
  },
  startSet(g) {
    Object.assign(g, { phase: g.cfg.mode === "signal" ? "ready" : "free", air: false, readySince: null, land: null, lastLandT: null, goAt: null });
    if (g.cfg.mode === "cont") say("¡Salta!", "Saltos seguidos, aterriza suave");
  },
  update(g, t, pose) {
    const c = g.cfg, R = g.jref;
    if (t >= g.setEnd && g.phase !== "stick") { saltos.closeLanding(g, t); return endSet(); }
    if (pose) {
      const fy = saltos.footY(g, pose);
      const hip = { x: (pose[L.lHip].x + pose[L.rHip].x) / 2 * canvas.width, y: (pose[L.lHip].y + pose[L.rHip].y) / 2 * canvas.height };
      const up = R.floor - fy;
      // despegue
      if (!g.air && up > 0.07 * R.leg) {
        g.air = true; g.takeoffT = t;
        saltos.closeLanding(g, t); // cierra la medición del aterrizaje anterior (modo continuo)
        if (g.phase === "go") { g.rt = t - g.goAt; hideStim(); }
      }
      // aterrizaje
      else if (g.air && up < 0.03 * R.leg) {
        g.air = false;
        const flight = t - g.takeoffT;
        if (flight >= 120 && flight <= 1200) {
          const contact = g.lastLandT != null && g.lastTakeoffWasFromLand && g.takeoffT - g.lastLandT < 1000 ? g.takeoffT - g.lastLandT : null;
          g.land = { t, flight, h: (9.81 * (flight / 1000) ** 2 / 8) * 100, hipX0: hip.x, maxHipY: hip.y, maxValgus: 0, maxSway: 0, contactPrev: contact, rt: g.rt ?? null };
          g.lastLandT = t; g.lastTakeoffWasFromLand = true; g.rt = null;
          if (c.mode === "signal") Object.assign(g, { phase: "stick", stickEnd: t + 1000 });
        }
      }
      // medición del aterrizaje (hasta 600 ms o hasta el siguiente despegue)
      if (g.land && !g.air) {
        const el = t - g.land.t;
        if (el <= 600) {
          g.land.maxHipY = Math.max(g.land.maxHipY, hip.y);
          const post = JUMP_POST[g.posture];
          const legs = post.stance ? [post.stance] : [-1, 1];
          g.land.maxValgus = Math.max(g.land.maxValgus, ...legs.map((s) => equilibrio.valgus(pose, s, R.shWpx)));
        }
        if (el <= 1000) g.land.maxSway = Math.max(g.land.maxSway, Math.abs(hip.x - g.land.hipX0) / R.shWpx);
        if (c.mode === "cont" && el > 1000) saltos.closeLanding(g, t);
      }
      if (c.mode === "cont" && !g.air && g.lastLandT != null && t - g.lastLandT > 1000) g.lastTakeoffWasFromLand = false;
    }
    switch (g.phase) {
      case "ready": {
        if (pose && !g.air) {
          g.readySince ??= t;
          say("Preparado…", JUMP_POST[g.posture].label);
          if (t - g.readySince > 400 && !g.goAt) { g.goAt = t + rand(c.waitMin, c.waitMax) * 1000; }
          if (g.goAt && t >= g.goAt) {
            stim.className = "show"; stim.style.background = "transparent"; stim.style.color = "#68d391"; stim.textContent = "⬆";
            beep(980, 120); say(""); g.phase = "go";
          }
        } else { g.readySince = null; }
        break;
      }
      case "go":
        if (!g.air && t - g.goAt > 2500) {
          g.trials.push({ set: g.set, posture: g.posture, outcome: "miss" });
          hideStim(); beep(220, 200); Object.assign(g, { phase: "ready", readySince: null, goAt: null });
        }
        break;
      case "stick":
        if (t >= g.stickEnd) { saltos.closeLanding(g, t); Object.assign(g, { phase: "ready", readySince: null, goAt: null }); }
        break;
    }
    const st = g.trials.filter((x) => x.set === g.set && x.outcome === "jump");
    const soft = st.filter((x) => x.soft).length;
    hud.score.textContent = `Saltos ${st.length} · suaves ${soft}`;
    if (pose) saltos.draw(g);
  },
  closeLanding(g) {
    const L0 = g.land; if (!L0) return;
    g.land = null;
    const R = g.jref;
    const absorb = (100 * (L0.maxHipY - R.hipY)) / R.leg; // descenso de cadera tras aterrizar (% pierna)
    const valgus = L0.maxValgus > 0.2, stable = L0.maxSway < 0.5, softDepth = absorb >= 8;
    const soft = !valgus && stable && softDepth;
    g.trials.push({ set: g.set, posture: g.posture, stance: JUMP_POST[g.posture].stance || 0, outcome: "jump",
      h: L0.h, flight: L0.flight, contact: L0.contactPrev, absorb, valgus, sway: L0.maxSway, stable, soft, rt: L0.rt });
    flash(soft ? "#38a16966" : "#ed893666", true); setTimeout(hideStim, 250);
    beep(soft ? 1200 : 400, 90);
    const why = soft ? "¡Aterrizaje suave!" : valgus ? "Rodilla hacia dentro" : !softDepth ? "Aterrizaje rígido: flexiona más" : "Estabiliza el aterrizaje";
    g.lastMsg = { txt: `${Math.round(L0.h)} cm · ${why}`, until: now() + 1500 };
  },
  draw(g) {
    const R = g.jref, W = canvas.width;
    ctx.strokeStyle = "#ffffff55"; ctx.lineWidth = 3; ctx.setLineDash([10, 10]);
    ctx.beginPath(); ctx.moveTo(0, R.floor); ctx.lineTo(W, R.floor); ctx.stroke(); ctx.setLineDash([]);
    if (g.lastMsg && now() < g.lastMsg.until) text(R.hipX, Math.max(60, R.hipY - R.leg * 1.4), g.lastMsg.txt, 44);
  },
  summarize(trials) {
    const j = trials.filter((t) => t.outcome === "jump"), miss = trials.filter((t) => t.outcome === "miss").length;
    const by = (s) => j.filter((t) => t.stance === s);
    const hL = mean(by(-1).map((t) => t.h)), hR = mean(by(1).map((t) => t.h));
    const contacts = j.map((t) => t.contact).filter((v) => v != null);
    const rsi = j.filter((t) => t.contact).map((t) => (t.h / 100) / (t.contact / 1000));
    return {
      jumps: j.length, misses: miss, h: mean(j.map((t) => t.h)), hMax: j.length ? Math.max(...j.map((t) => t.h)) : null,
      softPct: pct(j.filter((t) => t.soft).length, j.length), valgusPct: pct(j.filter((t) => t.valgus).length, j.length),
      absorb: mean(j.map((t) => t.absorb)), stablePct: pct(j.filter((t) => t.stable).length, j.length),
      contact: mean(contacts), rsi: mean(rsi), rt: mean(j.map((t) => t.rt).filter((v) => v != null)),
      hL, hR, softL: pct(by(-1).filter((t) => t.soft).length, by(-1).length), softR: pct(by(1).filter((t) => t.soft).length, by(1).length),
      nL: by(-1).length, nR: by(1).length,
    };
  },
  asym(m, inj) {
    if (m.hL == null || m.hR == null) return null;
    if (inj === "izq" || inj === "der") {
      const v = (100 * (inj === "izq" ? m.hL : m.hR)) / (inj === "izq" ? m.hR : m.hL);
      return { v, txt: `${Math.round(v)} %`, label: "LSI altura (lesionada / sana)" };
    }
    const v = (100 * Math.min(m.hL, m.hR)) / Math.max(m.hL, m.hR);
    return { v, txt: `${Math.round(v)} %`, label: "Simetría de altura entre piernas" };
  },
  results(s) {
    const m = s.summary, a = saltos.asym(m, s.injured), cm = (v) => (v == null ? "–" : `${v.toFixed(1)} cm`);
    const mono = m.nL || m.nR;
    return `
    <div class="stats">
      ${stat(m.jumps, `Saltos${m.misses ? ` (${m.misses} sin respuesta)` : ""}`)}
      ${stat(cm(m.h), `Altura media · máx. ${cm(m.hMax)}`)}
      ${stat(pctTxt(m.softPct), "Aterrizajes suaves")}
      ${mono ? stat(a ? a.txt : "–", a ? a.label : "Simetría") : stat(pctTxt(m.valgusPct), "Aterrizajes con rodilla hacia dentro")}
      ${m.contact != null ? stat(ms(m.contact), `Tiempo de contacto · RSI ${m.rsi != null ? m.rsi.toFixed(2) : "–"}`) : ""}
      ${m.rt != null ? stat(ms(m.rt), "Reacción a la señal") : ""}
    </div>
    <table>
      <tr><th>Calidad del aterrizaje</th><th></th></tr>
      <tr><td>Rodilla hacia dentro</td><td>${pctTxt(m.valgusPct)} de los saltos</td></tr>
      <tr><td>Amortiguación (bajada de cadera)</td><td>${m.absorb != null ? Math.round(m.absorb) + " % de la pierna" : "–"}</td></tr>
      <tr><td>Estable en 1 s</td><td>${pctTxt(m.stablePct)}</td></tr>
    </table>
    ${mono ? `<h3>Por pierna</h3><table>
      <tr><th></th><th>Izquierda</th><th>Derecha</th></tr>
      <tr><td>Saltos</td><td>${m.nL}</td><td>${m.nR}</td></tr>
      <tr><td>Altura</td><td>${cm(m.hL)}</td><td>${cm(m.hR)}</td></tr>
      <tr><td>Suaves</td><td>${pctTxt(m.softL)}</td><td>${pctTxt(m.softR)}</td></tr>
    </table>` : ""}
    <h3>Por serie</h3>
    <table>
      <tr><th>Serie</th><th>Saltos</th><th>Altura</th><th>Suaves</th><th>Rodilla dentro</th></tr>
      ${s.perSet.map((p) => `<tr><td>${p.set}</td><td>${p.jumps}</td><td>${cm(p.h)}</td><td>${pctTxt(p.softPct)}</td><td>${pctTxt(p.valgusPct)}</td></tr>`).join("")}
    </table>
    <p class="note">Altura estimada por el tiempo de vuelo. Suave = sin rodilla hacia dentro, con flexión al aterrizar (bajada de cadera ≥ 8 % de la pierna) y estable durante 1 s. Valores orientativos.</p>`;
  },
  histMain: (m) => `${m.jumps} saltos · ${m.h != null ? m.h.toFixed(1) : "–"} cm`,
  histDetail: (m) => `Suaves ${pctTxt(m.softPct)} · rodilla dentro ${pctTxt(m.valgusPct)}`,
};

// ======================================================================
// JUEGO 5: ESQUIVA
// ======================================================================
const esquiva = {
  name: "Esquiva",
  durationLabel: "Duración serie (s)",
  mirrorView: true,
  presets: {
    inicial:    { duration: 30, sets: 3, rest: 45, xTypes: "side", xSpeed: "2.2", xGapMin: 1.5, xGapMax: 2.5 },
    intermedia: { duration: 40, sets: 3, rest: 45, xTypes: "mix", xSpeed: "1.6", xGapMin: 1.0, xGapMax: 2.0 },
    avanzada:   { duration: 40, sets: 4, rest: 40, xTypes: "mix", xSpeed: "1.2", xGapMin: 0.6, xGapMax: 1.4 },
    vuelta:     { duration: 45, sets: 4, rest: 40, xTypes: "mix", xSpeed: "0.9", xGapMin: 0.4, xGapMax: 1.0 },
  },
  presetNames: { inicial: "Inicial: solo laterales, lento", intermedia: "Intermedia: mixto", avanzada: "Avanzada: mixto rápido", vuelta: "Vuelta al juego: muy rápido" },
  readCfg: (f) => ({ types: f.get("xTypes"), speed: +f.get("xSpeed"), gapMin: +f.get("xGapMin"), gapMax: Math.max(+f.get("xGapMin"), +f.get("xGapMax")) }),
  calibHint: "De pie, quieto y de frente a la cámara",
  afterCalib(g) {
    g.calibWarn = "";
    const W = canvas.width, H = canvas.height, C = g.center;
    g.xref = { noseY: C.noseY * H, hipY: C.hipY * H, shWpx: C.shW * W, head: Math.max(20, (C.hipY - C.noseY) * H) };
  },
  startSet(g) { Object.assign(g, { phase: "gap", nextAt: now() + 1000, obj: null, streak: 0 }); },
  spawn(g, t, pose) {
    const c = g.cfg, R = g.xref, W = canvas.width;
    const type = c.types === "mix" ? pick(["duck", "side", "side"]) : c.types;
    const hipX = (pose[L.lHip].x + pose[L.rHip].x) / 2 * W;
    const nose = pose[L.nose].y * canvas.height;
    g.obj = { type, t0: t, T: c.speed * 1000, x: hipX, startHipX: hipX, startNose: nose, moveT: null };
    beep(type === "duck" ? 500 : 760, 80);
    g.phase = "incoming";
  },
  update(g, t, pose) {
    const R = g.xref, W = canvas.width, H = canvas.height;
    if (g.phase === "gap") {
      if (t >= g.setEnd) return endSet();
      // espera a que esté de pie antes del siguiente objeto
      const upright = pose && (pose[L.nose].y * H - R.noseY) / R.head < 0.15;
      if (!upright) say("", "Ponte de pie");
      else say("");
      if (pose && upright && t >= g.nextAt) esquiva.spawn(g, t, pose);
    } else if (g.phase === "incoming" && pose) {
      const o = g.obj, el = t - o.t0;
      const hipX = (pose[L.lHip].x + pose[L.rHip].x) / 2 * W, nose = pose[L.nose].y * H;
      const duckDepth = (nose - R.noseY) / R.head;
      const sideMove = Math.abs(hipX - o.x) / R.shWpx;
      if (o.moveT === null && ((o.type === "duck" && (nose - o.startNose) / R.head > 0.1) || (o.type === "side" && Math.abs(hipX - o.startHipX) / R.shWpx > 0.3))) {
        o.moveT = el; o.dir = Math.sign(-(hipX - o.startHipX)); // +1 = hacia la derecha del atleta
      }
      if (el >= o.T) {
        const ok = o.type === "duck" ? duckDepth > 0.3 : sideMove > 0.9;
        g.trials.push({ set: g.set, type: o.type, outcome: ok ? "dodge" : "hit", rt: o.moveT, dir: o.type === "side" ? o.dir || 0 : 0 });
        g.streak = ok ? g.streak + 1 : 0;
        flash(ok ? "#38a16966" : "#e53e3e99", true); setTimeout(hideStim, 220);
        beep(ok ? 1200 : 180, ok ? 80 : 300);
        Object.assign(g, { phase: "gap", obj: null, nextAt: t + rand(g.cfg.gapMin, g.cfg.gapMax) * 1000 });
      }
    }
    if (g.phase === "incoming" && t >= g.setEnd + 3000) { g.obj = null; return endSet(); }
    esquiva.draw(g, t);
    const st = g.trials.filter((x) => x.set === g.set);
    hud.score.textContent = `✓ ${st.filter((x) => x.outcome === "dodge").length}  ✗ ${st.filter((x) => x.outcome === "hit").length}${g.streak > 2 ? `  🔥${g.streak}` : ""}`;
  },
  draw(g, t) {
    const o = g.obj; if (!o) return;
    const R = g.xref, W = canvas.width, p = Math.min(1, (t - o.t0) / o.T);
    if (o.type === "duck") {
      // barra a la altura de la cabeza que se acerca (se hace más gruesa y opaca)
      const y = R.noseY, th = 8 + p * R.head * 0.35;
      ctx.fillStyle = `rgba(229,62,62,${0.25 + 0.7 * p})`;
      ctx.fillRect(0, y - th / 2, W, th);
      text(W / 2, y - th / 2 - 30, "¡AGÁCHATE!", 36 + p * 20, "#feb2b2");
    } else {
      // bola que cae sobre la posición del atleta
      const r = R.shWpx * (0.35 + 0.35 * p), y = 20 + (R.noseY - 20) * p;
      ctx.fillStyle = `rgba(237,137,54,${0.5 + 0.5 * p})`; ctx.strokeStyle = "#fff"; ctx.lineWidth = 4;
      ctx.beginPath(); ctx.arc(o.x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = "rgba(237,137,54,.5)"; ctx.setLineDash([10, 10]);
      ctx.beginPath(); ctx.moveTo(o.x - 0.9 * R.shWpx, 0); ctx.lineTo(o.x - 0.9 * R.shWpx, canvas.height); ctx.moveTo(o.x + 0.9 * R.shWpx, 0); ctx.lineTo(o.x + 0.9 * R.shWpx, canvas.height); ctx.stroke();
      ctx.setLineDash([]);
    }
  },
  summarize(trials) {
    const d = trials.filter((t) => t.outcome === "dodge");
    const by = (f) => { const a = trials.filter(f); return { n: a.length, pct: pct(a.filter((t) => t.outcome === "dodge").length, a.length), rt: mean(a.filter((t) => t.outcome === "dodge" && t.rt != null).map((t) => t.rt)) }; };
    const duck = by((t) => t.type === "duck"), side = by((t) => t.type === "side");
    const sl = by((t) => t.type === "side" && t.dir < 0), sr = by((t) => t.type === "side" && t.dir > 0);
    return {
      total: trials.length, dodged: d.length, dodgePct: pct(d.length, trials.length), rt: mean(d.filter((t) => t.rt != null).map((t) => t.rt)),
      duckN: duck.n, duckPct: duck.pct, duckRt: duck.rt, sideN: side.n, sidePct: side.pct, sideRt: side.rt,
      toL: sl.n, toR: sr.n, rtL: sl.rt, rtR: sr.rt,
    };
  },
  asym: (m, inj) => timeAsym(m.rtL, m.rtR, inj),
  results(s) {
    const m = s.summary, a = esquiva.asym(m, s.injured);
    return `
    <div class="stats">
      ${stat(pctTxt(m.dodgePct), `Esquivados (${m.dodged}/${m.total})`)}
      ${stat(ms(m.rt), "Reacción media")}
      ${stat(a ? a.txt : "–", a ? a.label + " (reacción)" : "Asimetría")}
    </div>
    <table>
      <tr><th>Tipo</th><th>Objetos</th><th>Esquivados</th><th>Reacción</th></tr>
      ${m.duckN ? `<tr><td>Agacharse</td><td>${m.duckN}</td><td>${pctTxt(m.duckPct)}</td><td>${ms(m.duckRt)}</td></tr>` : ""}
      ${m.sideN ? `<tr><td>Lateral</td><td>${m.sideN}</td><td>${pctTxt(m.sidePct)}</td><td>${ms(m.sideRt)}</td></tr>` : ""}
    </table>
    ${m.sideN ? `<h3>Esquivas laterales por dirección</h3><table>
      <tr><th></th><th>Hacia la izquierda</th><th>Hacia la derecha</th></tr>
      <tr><td>Veces</td><td>${m.toL}</td><td>${m.toR}</td></tr>
      <tr><td>Reacción</td><td>${ms(m.rtL)}</td><td>${ms(m.rtR)}</td></tr>
    </table><p class="note">Si siempre esquiva hacia el mismo lado, puede estar evitando cargar la pierna lesionada.</p>` : ""}
    <h3>Por serie</h3>
    <table>
      <tr><th>Serie</th><th>Esquivados</th><th>Reacción</th></tr>
      ${s.perSet.map((p) => `<tr><td>${p.set}</td><td>${pctTxt(p.dodgePct)}</td><td>${ms(p.rt)}</td></tr>`).join("")}
    </table>`;
  },
  histMain: (m) => `${pctTxt(m.dodgePct)} esquivados · ${ms(m.rt)}`,
  histDetail: (m) => `Hacia izq./der.: ${m.toL} / ${m.toR}`,
};

const GAMES = { lateral, dianas, equilibrio, saltos, esquiva };

// ---------- Utilidades de resultados ----------
function stat(big, label) { return `<div class="stat"><b>${big}</b><span>${label}</span></div>`; }
function timeAsym(l, r, inj) {
  if (l == null || r == null) return null;
  let v, label;
  if (inj === "izq") { v = (100 * (l - r)) / r; label = "Lesionado (izq.) vs sano"; }
  else if (inj === "der") { v = (100 * (r - l)) / l; label = "Lesionado (der.) vs sano"; }
  else { v = (100 * Math.abs(l - r)) / Math.min(l, r); label = "Diferencia entre lados"; }
  return { v, txt: `${v > 0 && inj ? "+" : ""}${v.toFixed(0)} %`, label };
}

// ---------- Configuración ----------
const form = $("cfg");
let currentGame = "lateral";
function openConfig(key) {
  currentGame = key;
  const def = GAMES[key];
  $("cfg-title").textContent = def.name;
  $("lbl-duration").textContent = def.durationLabel;
  form.querySelectorAll("fieldset[data-game]").forEach((fs) => { const on = fs.dataset.game === key; fs.hidden = !on; fs.disabled = !on; });
  form.preset.innerHTML = `<option value="">Personalizado</option>` + Object.entries(def.presetNames).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  try {
    const last = JSON.parse(localStorage.getItem("readapta.lastform." + key) || "null");
    if (last) for (const [k, v] of Object.entries(last)) {
      const el = form.elements[k];
      if (!el || !el.name || el.disabled) continue;
      if (el.type === "checkbox") el.checked = v; else el.value = v;
    }
  } catch {}
  show("s-config");
}
form.preset.addEventListener("change", () => {
  const p = GAMES[currentGame].presets[form.preset.value];
  if (p) for (const [k, v] of Object.entries(p)) if (form.elements[k]) form.elements[k].value = v;
});
form.addEventListener("input", (e) => { if (!["preset", "player", "injured", "sound", "skeleton", "camera"].includes(e.target.name)) form.preset.value = ""; });
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const f = new FormData(form), def = GAMES[currentGame];
  const cfg = {
    game: currentGame, player: f.get("player").trim(), injured: f.get("injured"), preset: f.get("preset"),
    duration: +f.get("duration"), sets: +f.get("sets"), rest: +f.get("rest"),
    sound: !!f.get("sound"), skeleton: !!f.get("skeleton"), camera: f.get("camera"),
    ...def.readCfg(f),
  };
  const raw = {};
  for (const el of form.elements) if (el.name && !el.disabled) raw[el.name] = el.type === "checkbox" ? el.checked : el.value;
  localStorage.setItem("readapta.lastform." + currentGame, JSON.stringify(raw));
  startGame(cfg);
});
$("btn-repeat").onclick = () => openConfig(lastSession?.game || currentGame);

// ---------- Motor común ----------
let game = null, lastSession = null;

async function startGame(cfg) {
  const def = GAMES[cfg.game];
  show("s-play");
  $("s-play").classList.toggle("mirror-view", !!def.mirrorView);
  hideStim(); say("Preparando…", "Cargando detección y cámara");
  setStatus("Cargando…"); hud.set.textContent = hud.time.textContent = hud.score.textContent = "";
  try {
    audio = audio || new (window.AudioContext || window.webkitAudioContext)();
    if (audio.state === "suspended") await audio.resume();
  } catch {}
  game = { cfg, def, phase: "loading", trials: [], seqs: [], set: 1, center: null, lastPoseT: now() };
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
}

function startSet() {
  Object.assign(game, { setStart: now(), setEnd: now() + game.cfg.duration * 1000 });
  say(""); hideStim(); beep(660, 200);
  game.def.startSet(game);
}

function endSet() {
  if (game.set >= game.cfg.sets) return endGame();
  game.set++;
  Object.assign(game, { phase: "rest", restEnd: now() + game.cfg.rest * 1000 });
  hideStim();
}

function endGame(aborted = false) {
  if (!game || game.phase === "done") return;
  if (aborted) game.def.onAbort?.(game);
  game.phase = "done";
  stopCamera(); hideStim(); say("");
  try { wakeLock?.release(); } catch {}
  const session = buildSession(game, aborted);
  if (game.trials.length > 0) { const all = loadSessions(); all.push(session); saveSessions(all); }
  lastSession = session;
  renderResults(session);
  show("s-results");
}
$("btn-stop").onclick = () => endGame(true);

function tick() {
  if (!game || game.phase === "done") return;
  const t = now(), g = game, def = g.def, c = g.cfg;
  let pose = null;
  if (video.readyState >= 2) {
    const res = landmarker.detectForVideo(video, t);
    pose = res.landmarks?.[0] || null;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let full = false;
  if (pose) {
    g.lastPoseT = t;
    if (c.skeleton) new DrawingUtils(ctx).drawConnectors(pose, PoseLandmarker.POSE_CONNECTIONS, { color: "#4fd1c5", lineWidth: 4 });
    full = vis(pose[L.nose]) && vis(pose[L.lAnk]) && vis(pose[L.rAnk]);
    g.hipX = (pose[L.lHip].x + pose[L.rHip].x) / 2;
    g.hipY = (pose[L.lHip].y + pose[L.rHip].y) / 2;
    g.shW = Math.abs(pose[L.lSh].x - pose[L.rSh].x);
    setStatus(full ? "Cuerpo completo" : "No se ve el cuerpo entero", full ? "ok" : "warn");
  } else setStatus("No se detecta a nadie", "bad");

  switch (g.phase) {
    case "calib": {
      const hint = def.calibHintFor ? def.calibHintFor(g) : def.calibHint;
      const ok = pose && full && (!def.calibCheck || def.calibCheck(g, pose));
      if (ok) {
        g.calibStart ??= t;
        g.calibSamples.push({ x: g.hipX, y: g.hipY, w: g.shW, shY: (pose[L.lSh].y + pose[L.rSh].y) / 2, ankY: (pose[L.lAnk].y + pose[L.rAnk].y) / 2, noseY: pose[L.nose].y });
        say("Quieto…", `Calibrando ${Math.max(Math.ceil((1500 - (t - g.calibStart)) / 1000), 0)}`);
        if (t - g.calibStart > 1500) {
          const xs = g.calibSamples, avg = (k) => mean(xs.map((s) => s[k]));
          const x = avg("x"), w = avg("w");
          const spread = Math.max(...xs.map((s) => Math.abs(s.x - x)));
          if (w < 0.03 || spread > 0.4 * w) { g.calibSamples = []; g.calibStart = null; say("Quieto, por favor", "Vuelve a intentarlo"); break; }
          g.center = { x, y: avg("y"), shW: w, shY: avg("shY"), hipY: avg("y"), ankY: avg("ankY"), noseY: avg("noseY") };
          g.d = 0; g.off = null;
          def.afterCalib(g, pose);
          Object.assign(g, { phase: "countdown", countEnd: t + 3000, lastCount: null });
        }
      } else {
        g.calibSamples = []; g.calibStart = null;
        say(def.calibEachSet ? `Serie ${g.set}: colócate` : "Colócate en el centro",
          !pose ? hint : !full ? "Aléjate o baja la cámara: deben verse cabeza y pies" : def.calibFailMsg || hint);
      }
      break;
    }
    case "countdown": {
      const s = Math.ceil((g.countEnd - t) / 1000);
      if (s !== g.lastCount) { g.lastCount = s; if (s > 0) beep(520, 90); }
      say(s > 0 ? String(s) : "¡Ya!", g.calibWarn || `Serie ${g.set} de ${c.sets}`);
      if (t >= g.countEnd) startSet();
      break;
    }
    case "rest": {
      const s = Math.ceil((g.restEnd - t) / 1000);
      const next = def.calibEachSet && def.calibHintFor ? ` · ${def.calibHintFor(g)}` : "";
      say(`Descanso ${s}`, `Siguiente: serie ${g.set} de ${c.sets}${next}`);
      if (s <= 3 && s !== g.lastCount) { g.lastCount = s; if (s > 0) beep(520, 90); }
      if (t >= g.restEnd) { if (def.calibEachSet) enterCalib(); else startSet(); }
      break;
    }
    default:
      def.update(g, t, pose);
  }
  if (!["calib", "countdown", "rest", "done", "loading"].includes(g.phase)) {
    hud.set.textContent = `Serie ${g.set}/${c.sets}`;
    hud.time.textContent = `${Math.max(0, Math.ceil((g.setEnd - t) / 1000))} s`;
  } else { hud.set.textContent = hud.time.textContent = hud.score.textContent = ""; }
  if (game && game.phase !== "done") requestAnimationFrame(tick);
}

// ---------- Resultados ----------
function buildSession(g, aborted) {
  const def = g.def, perSet = [];
  for (let i = 1; i <= g.set; i++) {
    const tr = g.trials.filter((t) => t.set === i);
    if (tr.length) perSet.push({ set: i, ...def.summarize(tr, g.seqs.filter((s) => s.set === i)) });
  }
  return { id: Date.now(), date: new Date().toISOString(), game: g.cfg.game, player: g.cfg.player, injured: g.cfg.injured,
    cfg: g.cfg, aborted, summary: def.summarize(g.trials, g.seqs), perSet, trials: g.trials };
}
function renderResults(s) {
  const def = GAMES[s.game];
  $("results").innerHTML = `
    <p class="muted">${esc(def.name)} · ${esc(s.player)} · ${new Date(s.date).toLocaleString("es")}${s.aborted ? " · terminada antes" : ""}</p>
    ${def.results(s)}
    <p class="note">Medidas orientativas con una sola cámara. Úsalas para comparar al mismo jugador con el mismo montaje.</p>`;
}

// ---------- Historial ----------
function renderHistory() {
  const all = loadSessions();
  const names = [...new Set(all.map((s) => s.player))].sort();
  const sel = $("hist-player"), cur = sel.value, gsel = $("hist-game").value;
  sel.innerHTML = `<option value="">Todos</option>` + names.map((n) => `<option ${n === cur ? "selected" : ""}>${esc(n)}</option>`).join("");
  const rows = all.filter((s) => (!sel.value || s.player === sel.value) && (!gsel || s.game === gsel) && GAMES[s.game]).sort((a, b) => b.id - a.id);
  if (!rows.length) { $("history").innerHTML = `<p class="empty">Aún no hay sesiones guardadas.</p>`; return; }
  $("history").innerHTML = `<div class="overflow"><table>
    <tr><th>Fecha</th><th>Jugador</th><th>Juego</th><th>Resultado</th><th>Lados</th><th>Asim.</th><th></th></tr>
    ${rows.map((s) => { const d = GAMES[s.game], m = s.summary, a = d.asym(m, s.injured); return `<tr>
      <td>${new Date(s.date).toLocaleDateString("es")}</td><td>${esc(s.player)}</td><td>${d.name}</td>
      <td>${d.histMain(m)}</td><td>${d.histDetail(m)}</td><td>${a ? a.txt : "–"}</td>
      <td><button class="del" data-del="${s.id}" title="Borrar">×</button></td></tr>`; }).join("")}
  </table></div>`;
}
$("hist-player").addEventListener("change", renderHistory);
$("hist-game").addEventListener("change", renderHistory);
$("history").addEventListener("click", (e) => {
  const id = e.target.closest("[data-del]")?.dataset.del;
  if (id && confirm("¿Borrar esta sesión?")) { saveSessions(loadSessions().filter((s) => String(s.id) !== id)); renderHistory(); }
});
$("btn-csv").onclick = () => {
  const sessions = loadSessions().filter((s) => GAMES[s.game]);
  const keys = [...new Set(sessions.flatMap((s) => Object.keys(s.summary).filter((k) => typeof s.summary[k] !== "object")))];
  const head = ["fecha", "jugador", "juego", "lado_lesionado", "series", "preset", "asimetria", ...keys];
  const r = (v) => (v == null ? "" : typeof v === "number" ? Math.round(v * 10) / 10 : `"${String(v).replace(/"/g, '""')}"`);
  const lines = sessions.map((s) => { const a = GAMES[s.game].asym(s.summary, s.injured);
    return [s.date, s.player, GAMES[s.game].name, s.injured, s.perSet?.length || s.trials.length, s.cfg.preset, a?.v, ...keys.map((k) => s.summary[k])].map(r).join(","); });
  const blob = new Blob(["﻿" + [head.join(","), ...lines].join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const aEl = Object.assign(document.createElement("a"), { href: url, download: `readapp-${new Date().toISOString().slice(0, 10)}.csv` });
  document.body.appendChild(aEl); aEl.click(); aEl.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000);
};


// ======================================================================
// EVALUACIÓN DE SALTO (vídeo a cámara lenta, estilo My Jump)
// ======================================================================
const TESTS = { cmj: "CMJ", sj: "SJ", cmj1: "CMJ una pierna", dj: "Drop jump" };
const evalJump = {
  name: "Evaluación de salto",
  asym: () => null,
  histMain: (m) => `${TESTS[m.test]}${m.test === "cmj1" ? ` (${m.side})` : ""}: ${m.h.toFixed(1)} cm`,
  histDetail: (m) => `Vuelo ${Math.round(m.flight)} ms${m.rsi != null ? ` · RSI ${m.rsi.toFixed(2)} · contacto ${Math.round(m.contact)} ms` : ""}`,
};
GAMES.eval = evalJump;

const ev = {
  form: $("ev-form"), work: $("ev-work"), video: $("ev-video"), canvas: $("ev-canvas"),
  slider: $("ev-slider"), frameEl: $("ev-frame"), statusEl: $("ev-status"),
  fileFps: 30, n: 0, i: 0, marks: {}, url: null, ts: 0, floor: null,
};
const evCfg = () => {
  const f = new FormData(ev.form);
  return { player: (f.get("player") || "").trim(), injured: f.get("injured"), test: f.get("test"), side: f.get("side"), box: +f.get("box"), fps: +f.get("fps") };
};
function evToggleFields() {
  const t = ev.form.test.value;
  ev.form.querySelectorAll("[data-ev]").forEach((el) => (el.hidden = el.dataset.ev !== t));
  $("ev-m-contact").hidden = t !== "dj";
  evRender();
}
ev.form.test.addEventListener("change", evToggleFields);
ev.form.addEventListener("change", (e) => { if (e.target.name !== "video") evRender(); });
evToggleFields();

ev.form.video.addEventListener("change", async () => {
  const file = ev.form.video.files[0];
  if (!file) return;
  if (!ev.form.player.value.trim()) { alert("Escribe primero el nombre del jugador."); ev.form.video.value = ""; return; }
  if (ev.url) URL.revokeObjectURL(ev.url);
  ev.url = URL.createObjectURL(file);
  ev.marks = {}; ev.floor = null;
  ev.work.hidden = false;
  ev.statusEl.textContent = "Cargando vídeo…";
  ev.video.src = ev.url;
  await new Promise((r) => ev.video.addEventListener("loadedmetadata", r, { once: true }));
  ev.fileFps = await evDetectFps();
  ev.n = Math.max(1, Math.floor(ev.video.duration * ev.fileFps));
  ev.slider.max = ev.n - 1;
  const capt = evCfg().fps;
  ev.statusEl.textContent = ev.fileFps < capt - 5
    ? `Vídeo a ${Math.round(ev.fileFps)} fps: se toma cada fotograma como 1/${capt} s (cámara lenta)`
    : `Vídeo a ${Math.round(ev.fileFps)} fps`;
  await evSeek(0);
  evRender();
  ev.work.scrollIntoView({ behavior: "smooth" });
});

async function evDetectFps() {
  const v = ev.video;
  if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) return 30;
  const times = [];
  await new Promise((resolve) => {
    let done = false;
    const cb = (_, meta) => {
      times.push(meta.mediaTime);
      if (times.length < 25 && !done) v.requestVideoFrameCallback(cb); else finish();
    };
    const finish = () => { if (done) return; done = true; v.pause(); resolve(); };
    v.requestVideoFrameCallback(cb);
    v.play().catch(finish);
    setTimeout(finish, 2500);
  });
  v.pause(); v.currentTime = 0;
  const d = [];
  for (let k = 1; k < times.length; k++) { const x = times[k] - times[k - 1]; if (x > 0.0005) d.push(x); }
  if (d.length < 3) return 30;
  d.sort((a, b) => a - b);
  const fps = 1 / d[Math.floor(d.length / 2)];
  // redondea a un valor típico
  return [24, 25, 30, 48, 50, 60, 120, 240].reduce((a, b) => (Math.abs(b - fps) < Math.abs(a - fps) ? b : a));
}

function evSeek(i) {
  ev.i = Math.max(0, Math.min(ev.n - 1, i));
  ev.slider.value = ev.i;
  ev.frameEl.textContent = `Fotograma ${ev.i + 1} / ${ev.n}`;
  return new Promise((resolve) => {
    const t = (ev.i + 0.5) / ev.fileFps;
    if (Math.abs(ev.video.currentTime - t) < 1e-4) return resolve();
    ev.video.addEventListener("seeked", () => resolve(), { once: true });
    ev.video.currentTime = t;
  }).then(evDraw);
}
ev.slider.addEventListener("input", () => evSeek(+ev.slider.value));
document.querySelector(".ev-step").addEventListener("click", (e) => { const b = e.target.closest("[data-step]"); if (b) evSeek(ev.i + +b.dataset.step); });
document.querySelector(".ev-marks").addEventListener("click", (e) => {
  const b = e.target.closest("[data-mark]"); if (!b) return;
  ev.marks[b.dataset.mark] = ev.i; evRender(); evDraw();
});
// jump to a mark by long-press-free method: double click on the mark's value
document.querySelector(".ev-marks").addEventListener("dblclick", (e) => {
  const b = e.target.closest("[data-mark]"); if (b && ev.marks[b.dataset.mark] != null) evSeek(ev.marks[b.dataset.mark]);
});

function evCompute() {
  const c = evCfg(), m = ev.marks, dt = 1 / c.fps;
  if (m.takeoff == null || m.landing == null || m.landing <= m.takeoff) return null;
  const flight = (m.landing - m.takeoff) * dt;
  const h = (9.81 * flight * flight) / 8 * 100;
  const r = { test: c.test, side: c.test === "cmj1" ? c.side : null, flight: flight * 1000, h, frames: m.landing - m.takeoff, fpsCapture: c.fps, fileFps: ev.fileFps };
  if (c.test === "dj") {
    if (m.contact == null || m.contact >= m.takeoff) return null;
    r.contact = (m.takeoff - m.contact) * dt * 1000;
    r.rsi = (h / 100) / (r.contact / 1000);
    r.box = c.box;
  }
  return r;
}
function evRender() {
  const set = (id, k) => { const b = $(id); if (b) b.querySelector("b").textContent = ev.marks[k] != null ? `#${ev.marks[k] + 1}` : "–"; };
  set("ev-m-contact", "contact"); set("ev-m-takeoff", "takeoff"); set("ev-m-landing", "landing");
  const r = evCompute();
  $("ev-save").disabled = !r;
  $("ev-result").innerHTML = r ? `
    ${stat(`${r.h.toFixed(1)} cm`, "Altura (tiempo de vuelo)")}
    ${stat(`${Math.round(r.flight)} ms`, `Tiempo de vuelo · ${r.frames} fotogramas`)}
    ${r.rsi != null ? stat(r.rsi.toFixed(2), `RSI · contacto ${Math.round(r.contact)} ms`) : ""}` :
    `<p class="hint">Marca ${evCfg().test === "dj" ? "contacto, despegue y aterrizaje" : "despegue y aterrizaje"} con los botones, o pulsa "Detectar automáticamente" y ajusta.</p>`;
}
function evDraw() {
  const cv = ev.canvas, v = ev.video;
  if (!v.videoWidth) return;
  cv.width = v.videoWidth; cv.height = v.videoHeight;
  const c2 = cv.getContext("2d"); c2.clearRect(0, 0, cv.width, cv.height);
  if (ev.floor != null) {
    c2.strokeStyle = "#f6ad55"; c2.lineWidth = 3; c2.setLineDash([12, 10]);
    c2.beginPath(); c2.moveTo(0, ev.floor * cv.height); c2.lineTo(cv.width, ev.floor * cv.height); c2.stroke();
  }
  const mk = Object.entries(ev.marks).find(([, f]) => f === ev.i);
  if (mk) {
    c2.fillStyle = "#ecc94b"; c2.font = `700 ${Math.round(cv.height / 14)}px sans-serif`;
    c2.fillText({ contact: "CONTACTO", takeoff: "DESPEGUE", landing: "ATERRIZAJE" }[mk[0]], 20, cv.height / 10);
  }
}

// --- detección automática con pose ---
async function evFootY(c) {
  await evSeek(ev.i);
  ev.ts = Math.max(ev.ts + 1, performance.now());
  const res = landmarker.detectForVideo(ev.video, ev.ts);
  const p = res.landmarks?.[0];
  if (!p) return null;
  const pts = c.test === "cmj1" ? (c.side === "izq" ? [27, 29, 31] : [28, 30, 32]) : [27, 28, 29, 30, 31, 32];
  return { y: Math.max(...pts.map((k) => p[k].y)), leg: Math.abs((p[27].y + p[28].y) / 2 - (p[23].y + p[24].y) / 2) };
}
$("ev-auto").onclick = async () => {
  const btn = $("ev-auto"), c = evCfg();
  btn.disabled = true;
  try {
    ev.statusEl.textContent = "Cargando detección…";
    await loadModel();
    const step = Math.max(1, Math.ceil(ev.n / 300));
    const ys = new Map(), legs = [];
    const at = async (i) => {
      if (ys.has(i)) return ys.get(i);
      ev.i = i; const r = await evFootY(c);
      ys.set(i, r ? r.y : null); if (r) legs.push(r.leg);
      return ys.get(i);
    };
    for (let i = 0; i < ev.n; i += step) {
      await at(i);
      if ((i / step) % 10 === 0) ev.statusEl.textContent = `Analizando… ${Math.round((100 * i) / ev.n)} %`;
    }
    const vals = [...ys.values()].filter((v) => v != null).sort((a, b) => a - b);
    if (vals.length < 10) throw new Error("No se detecta bien al atleta en el vídeo.");
    const floor = vals[Math.floor(vals.length * 0.85)];
    const leg = legs.sort((a, b) => a - b)[Math.floor(legs.length / 2)] || 0.3;
    const thr = 0.04 * leg;
    ev.floor = floor;
    const air = async (i) => { const y = await at(i); return y != null && y < floor - thr; };
    // secuencia muestreada
    const idx = [...ys.keys()].sort((a, b) => a - b);
    const states = [];
    for (const i of idx) states.push({ i, a: await air(i) });
    // refina una transición entre dos muestras: devuelve el primer fotograma con estado 'target'
    const refine = async (lo, hi, target) => { for (let i = lo + 1; i <= hi; i++) if ((await air(i)) === target) return i; return hi; };
    let takeoff = null, landing = null, contact = null;
    if (c.test === "dj") {
      // en el cajón (aire) -> suelo (contacto) -> aire (salto) -> suelo (aterrizaje)
      let k = states.findIndex((s) => s.a);
      let kc = states.findIndex((s, j) => j > k && !s.a);
      let kt = states.findIndex((s, j) => j > kc && s.a);
      let kl = states.findIndex((s, j) => j > kt && !s.a);
      if (k < 0 || kc < 0 || kt < 0 || kl < 0) throw new Error("No encuentro caída, contacto y salto. Márcalos a mano.");
      contact = await refine(states[kc - 1].i, states[kc].i, false);
      takeoff = await refine(states[kt - 1].i, states[kt].i, true);
      landing = await refine(states[kl - 1].i, states[kl].i, false);
    } else {
      // el tramo en el aire más largo, con suelo antes y después
      let best = null;
      for (let j = 1; j < states.length; j++) {
        if (states[j].a && !states[j - 1].a) {
          let e = j; while (e < states.length && states[e].a) e++;
          if (e < states.length && (!best || e - j > best.e - best.s)) best = { s: j, e };
        }
      }
      if (!best) throw new Error("No encuentro el salto. Márcalo a mano.");
      takeoff = await refine(states[best.s - 1].i, states[best.s].i, true);
      landing = await refine(states[best.e - 1].i, states[best.e].i, false);
    }
    ev.marks = { takeoff, landing, ...(contact != null ? { contact } : {}) };
    await evSeek(takeoff);
    evRender();
    ev.statusEl.textContent = "Propuesta automática: revisa cada marca fotograma a fotograma";
  } catch (e) {
    ev.statusEl.textContent = e.message || String(e);
  } finally { btn.disabled = false; }
};

$("ev-save").onclick = () => {
  const r = evCompute(), c = evCfg(); if (!r) return;
  const session = { id: Date.now(), date: new Date().toISOString(), game: "eval", player: c.player, injured: c.injured,
    cfg: { ...c, preset: "" }, aborted: false, summary: r, perSet: [], trials: [r] };
  const all = loadSessions(); all.push(session); saveSessions(all);
  $("ev-save").disabled = true;
  ev.statusEl.textContent = "Guardado ✓";
  evRenderToday();
};
$("ev-new").onclick = () => { ev.form.video.value = ""; ev.work.hidden = true; ev.marks = {}; window.scrollTo(0, 0); };

function evRenderToday() {
  const c = evCfg(), today = new Date().toDateString();
  const rows = loadSessions().filter((s) => s.game === "eval" && s.player === c.player && new Date(s.date).toDateString() === today);
  if (!rows.length) { $("ev-today").innerHTML = ""; return; }
  const groups = {};
  for (const s of rows) { const k = s.summary.test + (s.summary.side ? "-" + s.summary.side : ""); (groups[k] ||= []).push(s.summary); }
  const line = (k, a) => {
    const hs = a.map((x) => x.h), mu = mean(hs), best = Math.max(...hs);
    const rsi = a.filter((x) => x.rsi != null).map((x) => x.rsi);
    const [t, side] = k.split("-");
    return `<tr><td>${TESTS[t]}${side ? ` ${side}.` : ""}</td><td>${a.length}</td><td>${mu.toFixed(1)} cm</td><td>${best.toFixed(1)} cm</td><td>${rsi.length ? mean(rsi).toFixed(2) : "–"}</td></tr>`;
  };
  let lsi = "";
  if (groups["cmj1-izq"] && groups["cmj1-der"]) {
    const L_ = mean(groups["cmj1-izq"].map((x) => x.h)), R_ = mean(groups["cmj1-der"].map((x) => x.h));
    const inj = c.injured;
    const v = inj === "izq" ? (100 * L_) / R_ : inj === "der" ? (100 * R_) / L_ : (100 * Math.min(L_, R_)) / Math.max(L_, R_);
    lsi = `<p class="note"><b>LSI ${inj ? `(lesionada / sana)` : "(menor / mayor)"}: ${Math.round(v)} %</b>${v < 90 ? " · por debajo del 90 % habitual para la vuelta al juego" : ""}</p>`;
  }
  $("ev-today").innerHTML = `<h3>Hoy · ${esc(c.player)}</h3><table>
    <tr><th>Test</th><th>Saltos</th><th>Media</th><th>Mejor</th><th>RSI</th></tr>
    ${Object.entries(groups).map(([k, a]) => line(k, a)).join("")}</table>${lsi}
    <p class="note">Recomendado: 3 saltos válidos por test y quedarse con la media.</p>`;
}
ev.form.player.addEventListener("change", evRenderToday);

// Acceso para pruebas automáticas
window.__readapta = { get game() { return game; }, GAMES };
