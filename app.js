// Readapta: juegos de readaptación con detección de pose por cámara.
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
  if (id === "s-config") fillPlayers();
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

const GAMES = { lateral, dianas, equilibrio };

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
        g.calibSamples.push({ x: g.hipX, y: g.hipY, w: g.shW, shY: (pose[L.lSh].y + pose[L.rSh].y) / 2, ankY: (pose[L.lAnk].y + pose[L.rAnk].y) / 2 });
        say("Quieto…", `Calibrando ${Math.max(Math.ceil((1500 - (t - g.calibStart)) / 1000), 0)}`);
        if (t - g.calibStart > 1500) {
          const xs = g.calibSamples, avg = (k) => mean(xs.map((s) => s[k]));
          const x = avg("x"), w = avg("w");
          const spread = Math.max(...xs.map((s) => Math.abs(s.x - x)));
          if (w < 0.03 || spread > 0.4 * w) { g.calibSamples = []; g.calibStart = null; say("Quieto, por favor", "Vuelve a intentarlo"); break; }
          g.center = { x, y: avg("y"), shW: w, shY: avg("shY"), hipY: avg("y"), ankY: avg("ankY") };
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
  const aEl = Object.assign(document.createElement("a"), { href: url, download: `readapta-${new Date().toISOString().slice(0, 10)}.csv` });
  document.body.appendChild(aEl); aEl.click(); aEl.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000);
};

// Acceso para pruebas automáticas
window.__readapta = { get game() { return game; }, GAMES };
