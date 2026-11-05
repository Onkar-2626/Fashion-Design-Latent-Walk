const ZDIM = 100, MODEL_URL = "model/generator.onnx";
const $ = (id) => document.getElementById(id);
const statusEl = $("status");
let session = null, zA, zB, base, busy = false, playing = false, raf = 0;


// ---------- random numbers / latent math
function randn() { let u = 0, v = 0; while (!u) u = Math.random(); v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const newZ = () => Float32Array.from({ length: ZDIM }, () => randn() * 0.8);
const norm = (a) => Math.sqrt(a.reduce((s, x) => s + x * x, 0));
function slerp(a, b, t) {               // spherical interpolation keeps the vector length typical for N(0,I)
  const na = norm(a), nb = norm(b);
  let dot = 0; for (let i = 0; i < ZDIM; i++) dot += (a[i] / na) * (b[i] / nb);
  const om = Math.acos(Math.min(1, Math.max(-1, dot))), s = Math.sin(om), out = new Float32Array(ZDIM);
  for (let i = 0; i < ZDIM; i++) out[i] = s < 1e-4 ? (1 - t) * a[i] + t * b[i] : (Math.sin((1 - t) * om) * a[i] + Math.sin(t * om) * b[i]) / s;
  return out;
}

// ---------- generator
async function generate(z) {
  const out = await session.run({ z: new ort.Tensor("float32", z, [1, ZDIM, 1, 1]) });
  return out.image.data;                // 784 floats in [-1, 1]
}

// ---------- colour palettes: the generator outputs grey values, we map each grey level to a colour gradient
const PALETTES = {
  sunset:  { label: "Sunset",   stops: [[0, "#12002b"], [.35, "#d6246e"], [.7, "#ff8a3d"], [1, "#fff2a8"]], ui: ["#e0307a", "#ff9a3d"], hue: -15 },
  ocean:   { label: "Ocean",    stops: [[0, "#001533"], [.4, "#0a6cc4"], [.75, "#2fd4e6"], [1, "#e8fffd"]], ui: ["#1d8fe8", "#25d3c4"], hue: 150 },
  neon:    { label: "Neon",     stops: [[0, "#0b0020"], [.4, "#7a2cff"], [.75, "#ff3ed0"], [1, "#9dfff0"]], ui: ["#8b3dff", "#ff3ed0"], hue: 245 },
  forest:  { label: "Forest",   stops: [[0, "#021a0e"], [.4, "#1c8a4a"], [.75, "#9ad43b"], [1, "#fbffd1"]], ui: ["#22a35a", "#a6d93a"], hue: 85 },
  gold:    { label: "Royal gold", stops: [[0, "#1a0b2e"], [.45, "#6a2fb8"], [.8, "#f5b932"], [1, "#fffbe0"]], ui: ["#7b3fd0", "#f2b21f"], hue: 330 },
  candy:   { label: "Candy",    stops: [[0, "#1e0a3c"], [.3, "#ff5fa2"], [.65, "#ffd23f"], [1, "#d4fff5"]], ui: ["#ff4f98", "#ffc933"], hue: -25 },
  original:{ label: "Original grey", stops: [[0, "#000000"], [1, "#ffffff"]], ui: ["#4f46e5", "#8b5cf6"], hue: null }
};
const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
function buildLUT(stops) {               // 256 x RGB lookup table, linearly blended between the stops
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255; let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k], [t1, c1] = stops[k + 1], f = Math.min(1, Math.max(0, (t - t0) / (t1 - t0)));
    const a = hex2rgb(c0), b = hex2rgb(c1);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = a[c] + (b[c] - a[c]) * f;
  }
  return lut;
}
let palKey = "sunset", lut = buildLUT(PALETTES.sunset.stops);
try { const s = localStorage.getItem("palette"); if (s && PALETTES[s]) { palKey = s; lut = buildLUT(PALETTES[s].stops); } } catch (e) {}

function applyPalette(key, remember = true) {
  const p = PALETTES[key]; palKey = key; lut = buildLUT(p.stops);
  const root = document.documentElement.style;
  root.setProperty("--accent", p.ui[0]); root.setProperty("--accent2", p.ui[1]);
  root.setProperty("--glow", p.ui[0] + "88");
  // pre-rendered grey PNG/GIF images get a matching colour tint via CSS filter
  root.setProperty("--tint", p.hue === null ? "none" : `sepia(1) saturate(3.2) hue-rotate(${p.hue}deg) brightness(1.15)`);
  if ($("palette")) $("palette").value = key;
  if (remember) { try { localStorage.setItem("palette", key); } catch (e) {} }
  repaint();
}
function repaint() { ["cA", "cM", "cB"].forEach((id) => { const c = $(id); if (c && c._data) draw(c, c._data); }); }

// Bicubic (Catmull-Rom) upscale of the 28x28 values in [-1,1] to size x size, then colourise
function cubic(p0, p1, p2, p3, t) {
  return p1 + 0.5 * t * (p2 - p0 + t * (2*p0 - 5*p1 + 4*p2 - p3 + t * (3*(p1 - p2) + p3 - p0)));
}
function upscale(data, size) {
  const out = new Float32Array(size * size), at = (x, y) =>
    data[Math.min(27, Math.max(0, y)) * 28 + Math.min(27, Math.max(0, x))];
  for (let y = 0; y < size; y++) {
    const fy = (y + 0.5) * 28 / size - 0.5, y0 = Math.floor(fy), ty = fy - y0;
    for (let x = 0; x < size; x++) {
      const fx = (x + 0.5) * 28 / size - 0.5, x0 = Math.floor(fx), tx = fx - x0, r = [];
      for (let j = -1; j <= 2; j++)
        r.push(cubic(at(x0-1, y0+j), at(x0, y0+j), at(x0+1, y0+j), at(x0+2, y0+j), tx));
      out[y * size + x] = cubic(r[0], r[1], r[2], r[3], ty);
    }
  }
  return out;
}
function draw(canvas, data, size = 448, contrast = 1.15) {
  canvas.width = canvas.height = size;
  const g = upscale(data, size), ctx = canvas.getContext("2d"), img = ctx.createImageData(size, size);
  for (let i = 0; i < g.length; i++) {
    let v = (g[i] * 0.5 + 0.5);                  // 0..1
    v = Math.min(1, Math.max(0, (v - 0.5) * contrast + 0.5));   // mild contrast boost
    const k = Math.round(v * 255) * 3;
    img.data[i*4] = lut[k]; img.data[i*4+1] = lut[k+1]; img.data[i*4+2] = lut[k+2]; img.data[i*4+3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

async function showEnds() { draw($("cA"), await generate(zA)); draw($("cB"), await generate(zB)); }
async function showMid() {
  const t = $("t").value / 100;
  $("tLabel").textContent = `t = ${t.toFixed(2)}`;
  draw($("cM"), await generate(slerp(zA, zB, t)));
}
async function showSliders() { draw($("cM"), await generate(base)); $("tLabel").textContent = "slider mix"; }

// ---------- UI wiring
function buildSliders() {
  const box = $("sliders"); box.innerHTML = "";
  for (let i = 0; i < 8; i++) {
    const l = document.createElement("label"), r = document.createElement("input");
    r.type = "range"; r.min = -3; r.max = 3; r.step = 0.1; r.value = base[i];
    r.oninput = async () => { base[i] = +r.value; if (!busy) { busy = true; await showSliders(); busy = false; } };
    l.append(`z[${i}]`, r); box.append(l);
  }
}
function wire() {
  $("t").oninput = async () => { if (!busy) { busy = true; await showMid(); busy = false; } };
  $("newA").onclick = async () => { zA = newZ(); await showEnds(); await showMid(); };
  $("newB").onclick = async () => { zB = newZ(); await showEnds(); await showMid(); };
  $("rand").onclick = async () => { base = newZ(); buildSliders(); await showSliders(); };
  $("reset").onclick = async () => { for (let i = 0; i < 8; i++) base[i] = 0; buildSliders(); await showSliders(); };
  $("download").onclick = () => {       // save an upscaled (448x448) coloured copy instead of the tiny 28x28 canvas
    const big = document.createElement("canvas"); big.width = big.height = 448;
    const g = big.getContext("2d"); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high"; g.drawImage($("cM"), 0, 0, 448, 448);
    const a = document.createElement("a"); a.download = "generated.png"; a.href = big.toDataURL("image/png"); a.click();
  };
  const sel = $("palette");
  sel.innerHTML = Object.entries(PALETTES).map(([k, p]) => `<option value="${k}">${p.label}</option>`).join("");
  sel.value = palKey; sel.onchange = () => applyPalette(sel.value);
  $("surprise").onclick = () => { const keys = Object.keys(PALETTES).filter((k) => k !== palKey && k !== "original"); applyPalette(keys[Math.floor(Math.random() * keys.length)]); };
  $("play").onclick = () => {
    playing = !playing; $("play").innerHTML = playing ? "&#10074;&#10074; Pause" : "&#9654; Animate";
    if (!playing) return cancelAnimationFrame(raf);
    let dir = 1;
    const step = async () => {
      if (!playing) return;
      let v = +$("t").value + dir * 2; if (v >= 100 || v <= 0) dir = -dir; $("t").value = Math.max(0, Math.min(100, v));
      await showMid(); raf = requestAnimationFrame(step);
    };
    step();
  };
}

// ---------- static data panels (work even without the model)
async function loadJSON(p) { try { const r = await fetch(p); return r.ok ? await r.json() : null; } catch { return null; } }
async function panels() {
  const meta = await loadJSON("data/meta.json");
  if (meta) $("metaLine").textContent = `DCGAN ${meta.gan_epochs} epochs (final D loss ${meta.final_d_loss}, G loss ${meta.final_g_loss}); VAE ${meta.vae_epochs} epochs (loss ${meta.final_vae_loss}); CNN checker test accuracy ${(meta.classifier_test_accuracy * 100).toFixed(1)}%.`;
  const cap = await loadJSON("data/captions.json"), gal = $("gallery");
  if (cap) {
    $("capSummary").textContent = `Caption category matched the image label for ${Math.round(cap.agreement * 100)}% of ${cap.n} samples.`;
    cap.items.forEach((it) => {
      const d = document.createElement("div"); d.className = "gi";
      const img = document.createElement("img"); img.src = it.file; img.alt = it.cnn_label; img.className = "tint";
      const p = document.createElement("div");
      p.innerHTML = `<b></b><br><span></span><br><span class="tag ${it.match ? "ok" : "bad"}"></span>`;
      p.querySelector("b").textContent = `Image: ${it.cnn_label}`;
      p.querySelector("span").textContent = `"${it.caption}"`;
      p.querySelector(".tag").textContent = `BERT: ${it.bert_label} (${Math.round(it.bert_conf * 100)}%) ${it.match ? "✓" : "✗"}`;
      d.append(img, p); gal.append(d);
    });
  } else gal.append(missing("data/captions.json"));
  const bal = await loadJSON("data/balance.json"), bb = $("balance");
  if (bal) {
    const max = Math.max(...bal.percent, 10) * 1.1;
    bb.innerHTML = `<div class="bars">` + bal.classes.map((c, i) =>
      `<div class="bar"><span>${c}</span><div class="t" style="--m:${10 / max * 100}%"><i style="width:${bal.percent[i] / max * 100}%"></i></div><span>${bal.percent[i]}%</span></div>`).join("") + `</div>
      <p><b>Result:</b> chi-square = ${bal.chi_square} (critical value ${bal.critical_value_p05}), so the generator is <b>${bal.balanced ? "balanced" : "NOT balanced"}</b>. Largest/smallest class ratio: ${bal.max_min_ratio}. Mean classifier confidence: ${bal.mean_classifier_confidence}.
      <span class="hint">An unconditional GAN has no class control, so some imbalance is expected (mode bias). The checker CNN is also imperfect, especially on blurry samples.</span></p>`;
  } else bb.append(missing("data/balance.json"));
}

// ---------- start
(async function init() {
  applyPalette(palKey, false);
  panels();
  if (window.ortFailed || typeof ort === "undefined") { statusEl.textContent = "Could not load the ONNX Runtime library (check your internet connection)."; statusEl.className = "status err"; return; }
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/";
  try {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
  } catch (e) {
    statusEl.textContent = "Generator model not found (public/model/generator.onnx). Run training/train.py first, see README.";
    statusEl.className = "status err"; document.querySelectorAll(".card button, #t").forEach((b) => (b.disabled = true)); return;
  }
  statusEl.textContent = "Generator ready. Everything below runs in your browser.";
  zA = newZ(); zB = newZ(); base = newZ(); for (let i = 0; i < 8; i++) base[i] = 0;
  buildSliders(); wire(); await showEnds(); await showMid();
})();
