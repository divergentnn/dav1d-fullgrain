// dav1d-fullgrain in the browser: a <canvas> that behaves like a (muted)
// <video> for a whole AV1-in-MP4 clip held in memory, decoded by dav1d in
// WebAssembly (dav1d-worker.mjs) and drawn with WebGL2.
//
//   import { wasmSupport, createWasmClip } from './wasm-clip.mjs';
//   if (!wasmSupport().length) {
//     const c = await createWasmClip(arrayBuffer, {fps: 24000 / 1001, mode: 'full'});
//     if (c) document.body.append(c);        // null: not AV1 (or no grain with requireGrain)
//   }
//
// The canvas gets the parts of the HTMLVideoElement API a frame-accurate
// player needs: currentTime (+ 'seeking'/'seeked' events), duration,
// videoWidth/videoHeight, readyState, paused, play(), pause(),
// requestVideoFrameCallback(cb) with metadata.mediaTime, plus
// setGrainMode(mode), dispose() and stats.
//
// Needs WebAssembly SIMD, WebGL2 and a cross-origin isolated page
// (COOP: same-origin, COEP: require-corp) for the decoder's threads.
//
// Copyright © 2026, divergentnn (dav1d-fullgrain). SPDX-License-Identifier: BSD-2-Clause

const SIMD_TEST = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10,
  10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);

/** [] when the WebAssembly player can run here, otherwise the reasons */
export function wasmSupport() {
  const why = [];
  if (typeof WebAssembly !== 'object') why.push('no WebAssembly');
  else if (!WebAssembly.validate(SIMD_TEST)) why.push('no WebAssembly SIMD');
  if (!self.crossOriginIsolated || typeof SharedArrayBuffer !== 'function')
    why.push('page is not cross-origin isolated (COOP/COEP headers)');
  try { if (!document.createElement('canvas').getContext('webgl2')) why.push('no WebGL2'); }
  catch (e) { why.push('no WebGL2'); }
  return why;
}

const VS = `#version 300 es
void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;
const FS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform usampler2D ty, tu, tv;
uniform vec2 ss;            // chroma subsampling factors (1 or 2)
uniform float yoff, ysc, coff, csc;
uniform vec3 k;             // (2(1-Kr), 2(1-Kb), unused); green from kg below
uniform vec2 kg;            // green: -kg.x * pb - kg.y * pr
uniform int mono;
out vec4 o;
float cfetch(usampler2D t, vec2 p) {
  ivec2 sz = textureSize(t, 0);
  vec2 q = p - 0.5, fl = floor(q), fr = q - fl;
  ivec2 a = clamp(ivec2(fl), ivec2(0), sz - 1), b = clamp(ivec2(fl) + 1, ivec2(0), sz - 1);
  float v00 = float(texelFetch(t, a, 0).r), v10 = float(texelFetch(t, ivec2(b.x, a.y), 0).r);
  float v01 = float(texelFetch(t, ivec2(a.x, b.y), 0).r), v11 = float(texelFetch(t, b, 0).r);
  return mix(mix(v00, v10, fr.x), mix(v01, v11, fr.x), fr.y);
}
void main() {
  ivec2 sz = textureSize(ty, 0);
  ivec2 p = ivec2(gl_FragCoord.xy); p.y = sz.y - 1 - p.y;
  float y = (float(texelFetch(ty, p, 0).r) - yoff) * ysc, pb = 0.0, pr = 0.0;
  if (mono == 0) {
    vec2 c = (vec2(p) + 0.5) / ss;      // chroma sited at the centre of its luma pixels
    pb = (cfetch(tu, c) - coff) * csc; pr = (cfetch(tv, c) - coff) * csc;
  }
  o = vec4(clamp(vec3(y + k.x * pr, y - kg.x * pb - kg.y * pr, y + k.y * pb), 0.0, 1.0), 1.0);
}`;

function renderer(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: false,
                                           premultipliedAlpha: false, preserveDrawingBuffer: true });
  if (!gl) throw new Error('no WebGL2');
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const U = n => gl.getUniformLocation(prog, n);
  const tex = [0, 1, 2].map(i => {
    const t = gl.createTexture(); gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, t);
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, p, gl.CLAMP_TO_EDGE);
    gl.uniform1i(U(['ty', 'tu', 'tv'][i]), i);
    return t;
  });
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  return f => {
    if (canvas.width !== f.w || canvas.height !== f.h) { canvas.width = f.w; canvas.height = f.h; }
    gl.viewport(0, 0, f.w, f.h);
    const hbd = f.bpc > 8, fmt = hbd ? gl.R16UI : gl.R8UI, typ = hbd ? gl.UNSIGNED_SHORT : gl.UNSIGNED_BYTE;
    const ssx = f.layout !== 3 ? 2 : 1, ssy = f.layout === 1 ? 2 : 1;
    const cw = Math.ceil(f.w / ssx), ch = Math.ceil(f.h / ssy);
    const up = (i, d, w, h) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, tex[i]);
      gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, gl.RED_INTEGER, typ, d); };
    up(0, f.y, f.w, f.h);
    if (f.u) { up(1, f.u, cw, ch); up(2, f.v, cw, ch); }
    const s = 1 << (f.bpc - 8), max = (1 << f.bpc) - 1;
    // matrix coefficients: 1 = BT.709, 5/6 = BT.601, 9/10 = BT.2020; others: BT.709
    const [kr, kb] = f.mtrx === 5 || f.mtrx === 6 ? [0.299, 0.114] : f.mtrx === 9 || f.mtrx === 10 ? [0.2627, 0.0593] : [0.2126, 0.0722];
    const kgv = 1 - kr - kb;
    if (f.full_range) { gl.uniform1f(U('yoff'), 0); gl.uniform1f(U('ysc'), 1 / max);
                        gl.uniform1f(U('coff'), 128 * s); gl.uniform1f(U('csc'), 1 / max); }
    else { gl.uniform1f(U('yoff'), 16 * s); gl.uniform1f(U('ysc'), 1 / (219 * s));
           gl.uniform1f(U('coff'), 128 * s); gl.uniform1f(U('csc'), 1 / (224 * s)); }
    gl.uniform3f(U('k'), 2 * (1 - kr), 2 * (1 - kb), 0);
    gl.uniform2f(U('kg'), 2 * kb * (1 - kb) / kgv, 2 * kr * (1 - kr) / kgv);
    gl.uniform2f(U('ss'), ssx, ssy);
    gl.uniform1i(U('mono'), f.u ? 0 : 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
}

/**
 * buffer: ArrayBuffer of an MP4 file (transferred to the worker)
 * opts: {fps, mode: 'full'|'standard'|..., threads, requireGrain, cacheBytes, workerUrl}
 * -> a canvas with the video-like API, or null if the clip is not AV1 (or has no film grain with requireGrain)
 */
export async function createWasmClip(buffer, opts = {}) {
  const fps = opts.fps || 24000 / 1001;
  const threads = opts.threads || Math.max(2, Math.min(6, (navigator.hardwareConcurrency || 4) - 1));
  const worker = new Worker(opts.workerUrl || new URL('./dav1d-worker.mjs', import.meta.url), { type: 'module' });
  let firstFrame = null;
  const info = await new Promise((res, rej) => {
    let opened = null;
    worker.onmessage = e => {
      const m = e.data;
      if (m.type === 'opened') { opened = m; if (!m.frames) res(m); }
      else if (m.type === 'frame' && m.id === -1) { firstFrame = m; res(opened); }
      else if (m.type === 'error') rej(new Error(m.message));
    };
    worker.onerror = e => rej(new Error(e.message || 'worker failed to start'));
    worker.postMessage({ type: 'open', buffer, threads, mode: opts.mode || 'full' }, [buffer]);
  }).catch(e => { worker.terminate(); throw e; });
  if (!info.frames || (opts.requireGrain && !info.grain) || (info.trc === 16 || info.trc === 18)) {
    worker.terminate();
    return null;                                 // not AV1 / no grain / HDR: leave it to <video>
  }

  const c = document.createElement('canvas');
  const draw = renderer(c);
  const frames = info.frames, frameBytes = firstFrame.y.byteLength * (firstFrame.u ? 1.5 : 1);
  const cacheMax = Math.max(4, Math.floor((opts.cacheBytes || 160e6) / frameBytes));
  const cache = new Map(), pending = new Map(), rvfc = [];
  let cur = 0, time = 0.5 / fps, nextId = 1, seekTok = 0, mode = opts.mode || 'full';
  let playing = false, raf = 0, t0 = 0, f0 = 0, seeking = false, modeSwitch = false;
  const stats = { decoded: 0, shown: 0, dropped: 0, decodeMs: 0 };

  const put = f => {
    cache.set(f.n, f);
    if (cache.size > cacheMax) {                  // evict the frame farthest from the one on screen
      let far = -1, dist = -1;
      for (const n of cache.keys()) { const d = n < cur ? (cur - n) * 2 : n - cur; if (d > dist && n !== cur) { dist = d; far = n; } }
      cache.delete(far);
    }
  };
  worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'mode') { modeSwitch = false; return; }
    if (m.type === 'frame') {
      if (modeSwitch) return;                      // decoded with the previous grain mode
      stats.decoded++; put(m);
      const p = pending.get(m.n);
      if (p) { pending.delete(m.n); p.res(m); }
    } else if (m.type === 'error') {
      for (const [n, p] of pending) if (m.id === undefined || p.id === m.id) { pending.delete(n); p.rej(new Error(m.message)); }
    }
  };
  const want = n => {
    if (cache.has(n)) return Promise.resolve(cache.get(n));
    if (pending.has(n)) return pending.get(n).promise;
    const id = nextId++;
    let res, rej; const promise = new Promise((a, b) => { res = a; rej = b; });
    pending.set(n, { id, res, rej, promise, t: performance.now() });
    worker.postMessage({ type: 'frame', n, id, keep: Math.min(8, cacheMax >> 1) });
    return promise;
  };
  const present = (f, now) => {
    draw(f); cur = f.n; stats.shown++;
    const cbs = rvfc.splice(0);
    const md = { mediaTime: f.n / fps, presentedFrames: stats.shown, width: f.w, height: f.h };
    for (const cb of cbs) { try { cb(now || performance.now(), md); } catch (e) { console.error(e); } }
  };
  present(firstFrame); cache.set(0, firstFrame);

  const loop = now => {
    if (!playing) return;
    const target = Math.min(frames - 1, f0 + Math.floor((now - t0) * fps / 1000));
    for (let n = cur + 1; n <= Math.min(frames - 1, cur + 6); n++) want(n);
    let best = -1;
    for (let n = target; n > cur; n--) if (cache.has(n)) { best = n; break; }
    if (best > cur) { stats.dropped += best - cur - 1; present(cache.get(best), now); time = (best + 0.5) / fps; }
    raf = requestAnimationFrame(loop);
  };
  const startClock = () => { t0 = performance.now(); f0 = cur; };

  Object.defineProperties(c, {
    wasm: { value: true },
    videoWidth: { get: () => info.width }, videoHeight: { get: () => info.height },
    duration: { get: () => frames / fps },
    readyState: { get: () => 4 },
    paused: { get: () => !playing },
    seeking: { get: () => seeking },
    grainMode: { get: () => mode },
    hasGrain: { get: () => !!info.grain },
    frameCount: { get: () => frames },
    stats: { get: () => stats },
    currentTime: {
      get: () => (playing ? (cur + 0.5) / fps : time),
      set: t => {
        const n = Math.min(frames - 1, Math.max(0, Math.floor(t * fps + 1e-6)));
        time = t;
        const tok = ++seekTok;
        seeking = true;
        c.dispatchEvent(new Event('seeking'));
        want(n).then(f => {
          if (tok !== seekTok) return;
          present(f);
          if (playing) startClock();
          seeking = false;
          c.dispatchEvent(new Event('seeked'));
        }, () => { if (tok === seekTok) { seeking = false; c.dispatchEvent(new Event('seeked')); } });
      },
    },
  });
  c.play = async () => { if (playing) return; playing = true; startClock(); raf = requestAnimationFrame(loop); };
  c.pause = () => { playing = false; cancelAnimationFrame(raf); time = (cur + 0.5) / fps; };
  c.requestVideoFrameCallback = cb => { rvfc.push(cb); return rvfc.length; };
  c.cancelVideoFrameCallback = () => {};
  c.load = () => {};
  /** switch grain synthesis; resolves when the frame on screen has been redrawn with it */
  c.setGrainMode = async m => {
    if (m === mode) return;
    mode = m; modeSwitch = true;
    cache.clear();
    for (const [, p] of pending) p.rej(new Error('grain mode changed'));
    pending.clear();
    worker.postMessage({ type: 'mode', mode: m });
    const f = await want(cur);
    present(f);
    c.dispatchEvent(new Event('seeked'));
  };
  /** the decoded planes of frame n (for tests / analysis): {n, w, h, bpc, layout, y, u, v, grain, ...} */
  c.getFrame = n => want(n);
  c.dispose = () => { playing = false; cancelAnimationFrame(raf); worker.terminate(); cache.clear(); };
  return c;
}
