// av1-fullgrain in the browser: a <canvas> that behaves like a (muted) <video>
// for an AV1 clip with film grain. The browser's own AV1 decoder (WebCodecs,
// hardware where the device has it) decodes a grain-free copy of the clip,
// and the grain is drawn from its PARAMS.json:
//
//   tier 1  WebGPU compute shaders (gpu-grain.mjs) + a WebGPU canvas
//   tier 2  without WebGPU: the fork's grain code in WebAssembly
//           (fg-grain.wasm in grain-worker.mjs, loaded only then) + WebGL2
//
// Both give pictures bit-identical to the dav1d fork (full-frame grain) and to
// dav1d (standard AV1 grain). Browsers that cannot decode AV1 are not
// supported: createGrainClip() throws, and av1Support() says so first.
//
//   import { av1Support, createGrainClip } from './grain-clip.mjs';
//   const s = await av1Support(codec);              // {decode, engine, why}
//   const c = await createGrainClip(mp4ArrayBuffer, paramsJson, {fps: 24000 / 1001, mode: 'full'});
//   document.body.append(c); c.play();
//
// The canvas offers what a frame-accurate player needs of HTMLVideoElement:
// currentTime (+ 'seeking' / 'seeked' events), duration, videoWidth,
// videoHeight, readyState, paused, seeking, play(), pause(),
// requestVideoFrameCallback(cb) (metadata.mediaTime), and setGrainMode(mode),
// getFrame(n) (the grained planes, for tests), engine, stats, dispose().
//
// Copyright © 2026, divergentnn (av1-fullgrain). SPDX-License-Identifier: BSD-2-Clause

import { demuxAv1 } from './mp4-demux.mjs';
import { loadParams } from './fg-params.mjs';
import { createGpuGrain, gpuSupport } from './gpu-grain.mjs';
import { GpuRenderer, glRenderer } from './yuv-render.mjs';

const US = 1000;                    // chunk timestamps: frame index * 1000 µs (only the order matters)

/** can this browser decode the codec (WebCodecs)? -> {decode: bool, why} */
export async function av1DecodeSupport(codec = 'av01.0.08M.10', width = 1920, height = 1080) {
  if (typeof VideoDecoder !== 'function') return { decode: false, why: 'no WebCodecs' };
  try {
    const r = await VideoDecoder.isConfigSupported({ codec, codedWidth: width, codedHeight: height });
    return { decode: !!r.supported, why: r.supported ? '' : `no AV1 decoder for ${codec}` };
  } catch (e) { return { decode: false, why: 'WebCodecs: ' + e.message }; }
}

let wasmMod = null;
const loadWasm = () => wasmMod || (wasmMod = import('./wasm-grain.mjs'));

/** the grain engine this browser gets: {engine: 'webgpu' | 'wasm' | null, why: [...]} */
export async function grainEngineSupport(prefer = 'auto') {
  const why = [];
  if (prefer !== 'wasm') {
    const g = await gpuSupport();
    if (!g.length) return { engine: 'webgpu', why };
    why.push(...g);
  }
  const w = (await loadWasm()).wasmGrainSupport();
  if (!w.length) return { engine: 'wasm', why };
  return { engine: null, why: why.concat(w) };
}

/** both at once: {decode, engine, why: string} */
export async function av1Support(codec, width, height, prefer) {
  const [d, e] = await Promise.all([av1DecodeSupport(codec, width, height), grainEngineSupport(prefer)]);
  const why = [d.why, ...e.why].filter(Boolean).join(', ');
  return { decode: d.decode, engine: e.engine, why };
}


/** a VideoFrame's planes as copyTo() layout: {bps, semi, size, planes: [{offset, stride}]}, or null */
export function planarLayout(frame) {
  const fmt = frame.format, w = frame.visibleRect.width, h = frame.visibleRect.height;
  const m = /^(I420|I422|I444)(P10|P12)?$/.exec(fmt || '');
  if (m) {
    const bps = m[2] ? 2 : 1, ssx = m[1] === 'I444' ? 0 : 1, ssy = m[1] === 'I420' ? 1 : 0;
    const cw = (w + ssx) >> ssx, ch = (h + ssy) >> ssy, ysz = w * h * bps, csz = cw * ch * bps;
    return { bps, semi: false, size: ysz + 2 * csz,
             planes: [{ offset: 0, stride: w * bps }, { offset: ysz, stride: cw * bps }, { offset: ysz + csz, stride: cw * bps }] };
  }
  if (fmt === 'NV12') {
    const cw = (w + 1) >> 1, ch = (h + 1) >> 1;
    return { bps: 1, semi: true, size: w * h + 2 * cw * ch,
             planes: [{ offset: 0, stride: w }, { offset: w * h, stride: 2 * cw }] };
  }
  return null;
}

/** NV12 -> planar I420 (the CPU grain path wants planar planes) */
function deinterleave(bytes, L, w, h) {
  const cw = (w + 1) >> 1, ch = (h + 1) >> 1, src = new Uint8Array(bytes), out = new Uint8Array(w * h + 2 * cw * ch);
  out.set(src.subarray(0, w * h));
  const u = w * h, v = u + cw * ch, uv = L.planes[1].offset, st = L.planes[1].stride;
  for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
    out[u + y * cw + x] = src[uv + y * st + 2 * x]; out[v + y * cw + x] = src[uv + y * st + 2 * x + 1];
  }
  return { bytes: out.buffer, bps: 1, semi: false, planes: [{ offset: 0, stride: w }, { offset: u, stride: cw }, { offset: v, stride: cw }] };
}

/**
 * buffer: ArrayBuffer of the grain-free MP4; params: its PARAMS.json (text or object)
 * opts: {fps, mode: 'full'|'standard', engine: 'auto'|'webgpu'|'wasm', hw: WebCodecs hardwareAcceleration,
 *        cacheBytes, threads}
 * -> the canvas; throws when the clip or this browser cannot be played (the message says why)
 */
export async function createGrainClip(buffer, params, opts = {}) {
  const fps = opts.fps || 24000 / 1001;
  const d = demuxAv1(buffer);
  if (!d) throw new Error('no AV1 track');
  const P = loadParams(params);
  if (P.frames !== d.samples.length) throw new Error(`the grain parameters list ${P.frames} pictures, the clip has ${d.samples.length}`);
  const fmt = { width: P.width, height: P.height, bitdepth: P.bitdepth, layout: P.layout, mtrx: P.mtrx,
                trc: P.trc, pri: P.pri, range: P.color_range };
  const dcfg = { codec: d.codec, codedWidth: d.width, codedHeight: d.height,
                 hardwareAcceleration: opts.hw || 'no-preference', optimizeForLatency: true };
  if (typeof VideoDecoder !== 'function') throw new Error('this browser cannot decode AV1 (no WebCodecs)');
  if (!(await VideoDecoder.isConfigSupported(dcfg)).supported) throw new Error(`this browser cannot decode AV1 (${d.codec})`);
  const eng = opts.engine && opts.engine !== 'auto' ? { engine: opts.engine } : await grainEngineSupport();
  if (!eng.engine) throw new Error('no way to draw the grain here: ' + eng.why.join(', '));

  const c = document.createElement('canvas');
  c.width = fmt.width; c.height = fmt.height;
  let gg = null, ren = null, wg = null, gl = null;
  if (eng.engine === 'webgpu') {
    gg = await createGpuGrain();                // its own buffers, the page's shared device
    gg.configure(fmt);
    ren = new GpuRenderer(gg, c, { hdr: opts.hdr });
  } else {
    wg = await (await loadWasm()).createWasmGrain({ threads: opts.threads });
    gl = glRenderer(c);
  }

  const N = d.samples.length, frameBytes = (fmt.width * fmt.height * 1.5 + 64) * (fmt.bitdepth > 8 ? 2 : 1);
  const cacheMax = Math.max(6, Math.floor((opts.cacheBytes || 256e6) / frameBytes));
  const cache = new Map(), pending = new Map(), rvfc = [];
  const stats = { decoded: 0, shown: 0, dropped: 0, grainMs: 0, grained: 0, copyMs: 0, format: null, engine: eng.engine };
  let mode = opts.mode || 'standard', cur = -1, time = 0.5 / fps, seekTok = 0;
  let playing = false, raf = 0, t0 = 0, f0 = 0, seeking = false, disposed = false, err = null;
  let dec = null, gen = 0, feedPos = 0, outPos = 0, keepFrom = 0, target = -1;

  const put = e => {
    cache.set(e.n, e);
    while (cache.size > cacheMax) {             // evict the frame farthest from the one on screen (behind counts double)
      let far = -1, dist = -1;
      for (const n of cache.keys()) {
        const dd = n < cur ? (cur - n) * 2 : n - cur;
        if (dd > dist && n !== cur && !pending.has(n)) { dist = dd; far = n; }
      }
      if (far < 0) break;
      const old = cache.get(far);
      cache.delete(far);
      if (old.gpu) { const b = old.gpu; drawing.then(() => gg.device.queue.onSubmittedWorkDone()).then(() => b.destroy()); }
    }
  };

  const fail = e => {
    err = e;
    for (const [, p] of pending) p.rej(e);
    pending.clear();
    c.dispatchEvent(new Event('error'));
  };

  function restart(n) {
    let k = n;
    while (k > 0 && !d.samples[k].sync) k--;
    for (const [m, p] of pending)                 // requests this decode will not reach (a seek superseded them)
      if (m !== n && (m < Math.max(k, n - 8) || m > n + 48)) { pending.delete(m); p.rej(new Error('superseded')); }
    if (dec && dec.state !== 'closed') dec.close();
    const g = ++gen;
    dec = new VideoDecoder({
      output: f => onOutput(f, g),
      error: e => { if (g === gen) fail(new Error('AV1 decoder: ' + (e.message || e))); },
    });
    dec.configure(dcfg);
    feedPos = outPos = k;
    keepFrom = Math.max(k, n - 8);              // a few frames before n are kept for stepping back
    target = n;
    flushed = false;
  }

  async function onOutput(frame, g) {
    const n = Math.round(frame.timestamp / US);
    try {
      if (g !== gen || disposed) return;
      outPos = Math.max(outPos, n + 1);
      if (n < keepFrom && !pending.has(n)) return;
      if (!stats.format) stats.format = frame.format;
      let L = planarLayout(frame);
      if (!L) {
        // a hardware decoder's frames that copyTo() cannot read (format null / unknown): decode in software
        const sw = { ...dcfg, hardwareAcceleration: 'prefer-software' };
        if (dcfg.hardwareAcceleration !== 'prefer-software' && (await VideoDecoder.isConfigSupported(sw)).supported) {
          if (g === gen) {
            stats.fallback = `decoder output ${frame.format} is not readable; decoding in software`;
            dcfg.hardwareAcceleration = 'prefer-software';
            restart(Math.min(...pending.keys(), n));
            pump();
          }
          return;
        }
        fail(new Error(`unsupported decoder output format ${frame.format}`));
        return;
      }
      const t = performance.now();
      let e;
      if (gg) {                                   // straight into a GPU buffer: the cache lives on the GPU
        const { buffer, range } = gg.sourceBuffer(L.size);
        try { await frame.copyTo(range, { layout: L.planes }); } finally { buffer.unmap(); }
        e = { n, gpu: buffer, planes: L.planes, bps: L.bps, semi: L.semi };
      } else {
        let bytes = new ArrayBuffer(L.size);
        await frame.copyTo(bytes, { layout: L.planes });
        if (L.semi) { const r = deinterleave(bytes, L, fmt.width, fmt.height); bytes = r.bytes; L = r; }
        e = { n, bytes, planes: L.planes, bps: L.bps, semi: L.semi };
      }
      stats.copyMs += performance.now() - t;
      stats.decoded++;
      if (g !== gen) { if (e.gpu) e.gpu.destroy(); return; }
      put(e);
      const p = pending.get(n);
      if (p) { pending.delete(n); p.res(e); }
    } catch (e) {
      if (g === gen) fail(e);
    } finally {
      frame.close();
      if (g === gen && !disposed) pump();
    }
  }

  // feed the decoder up to the target (+ a few frames while playing); a decoder that holds frames back gets
  // more input, at the end of the clip a flush
  let lastOut = -1, watchdog = 0;
  function pump() {
    if (!dec || dec.state !== 'configured') return;
    for (const m of pending.keys()) if (m < outPos + 48) target = Math.max(target, m);
    const ahead = playing ? 6 : 1;
    while (feedPos < N && feedPos <= target + ahead && dec.decodeQueueSize < 3 && feedPos - outPos < 8) {
      const s = d.samples[feedPos];
      dec.decode(new EncodedVideoChunk({ type: s.sync ? 'key' : 'delta', timestamp: feedPos * US,
                                         data: d.bytes.subarray(s.offset, s.offset + s.size) }));
      feedPos++;
    }
    if (feedPos >= N && outPos < N && !flushed) { flushed = true; dec.flush().catch(() => {}); }
    clearTimeout(watchdog);
    if (pending.size) {
      const at = outPos;
      watchdog = setTimeout(() => {
        if (disposed || !pending.size || outPos !== at || lastOut === at) return;
        lastOut = at;
        target += 2;                               // no output for a while: give the decoder more input
        pump();
      }, 150);
    }
  }
  let flushed = false;

  function want(n) {
    if (err) return Promise.reject(err);
    if (cache.has(n)) return Promise.resolve(cache.get(n));
    if (pending.has(n)) return pending.get(n).promise;
    let res, rej;
    const promise = new Promise((a, b) => { res = a; rej = b; });
    pending.set(n, { res, rej, promise });
    if (!dec || n < outPos || n - outPos > 48 || n < keepFrom) restart(n);
    else target = Math.max(target, n);
    pump();
    return promise;
  }

  // drawing: grain + render; one at a time
  let drawing = Promise.resolve();
  const draw = e => (drawing = drawing.catch(() => {}).then(async () => {
    if (disposed) return;
    const t = performance.now();
    const pic = P.picture(e.n);
    if (gg) {
      gg.configure(fmt);
      ren.configure(fmt);
      gg.setSource(e.gpu, e.planes, e.bps, e.semi);
      gg.setGrain(pic, mode);
      const enc = gg.device.createCommandEncoder();
      gg.encode(enc);
      ren.encode(enc);
      gg.device.queue.submit([enc.finish()]);
    } else {
      let r;
      try { r = await wg.apply(e.bytes, e.planes, e.bps, fmt, pic, mode); } catch (x) { if (disposed) return; throw x; }
      gl(r.bytes, fmt);
    }
    stats.grainMs += performance.now() - t; stats.grained++;
  }));

  const present = async (e, now) => {
    await draw(e);
    cur = e.n; stats.shown++;
    const cbs = rvfc.splice(0);
    const md = { mediaTime: e.n / fps, presentedFrames: stats.shown, width: fmt.width, height: fmt.height };
    for (const cb of cbs) { try { cb(now || performance.now(), md); } catch (x) { console.error(x); } }
  };

  const first = await want(0);
  await present(first);

  let busy = false;
  const loop = async now => {
    if (!playing || disposed) return;
    raf = requestAnimationFrame(loop);
    if (busy) return;
    const tgt = Math.min(N - 1, f0 + Math.floor((now - t0) * fps / 1000));
    for (let n = cur + 1; n <= Math.min(N - 1, cur + 6); n++) want(n).catch(() => {});
    let best = -1;
    for (let n = tgt; n > cur; n--) if (cache.has(n)) { best = n; break; }
    if (best > cur) {
      stats.dropped += best - cur - 1;
      busy = true;
      try { await present(cache.get(best), now); } finally { busy = false; }
      time = (best + 0.5) / fps;
    }
    if (cur >= N - 1) { playing = false; cancelAnimationFrame(raf); c.dispatchEvent(new Event('ended')); }
  };
  const startClock = () => { t0 = performance.now(); f0 = cur; };

  Object.defineProperties(c, {
    player: { value: true },
    engine: { get: () => eng.engine },
    decoder: { get: () => ({ codec: d.codec, hardwareAcceleration: dcfg.hardwareAcceleration, format: stats.format }) },
    videoWidth: { get: () => fmt.width }, videoHeight: { get: () => fmt.height },
    duration: { get: () => N / fps },
    readyState: { get: () => 4 },
    paused: { get: () => !playing },
    seeking: { get: () => seeking },
    grainMode: { get: () => mode },
    hasGrain: { get: () => P.sets.length > 0 },
    hdr: { get: () => fmt.trc === 16 || fmt.trc === 18 },
    hdrOutput: { get: () => !!(ren && ren.extended) },
    frameCount: { get: () => N },
    stats: { get: () => stats },
    error: { get: () => err },
    currentTime: {
      get: () => (playing ? (cur + 0.5) / fps : time),
      set: t => {
        const n = Math.min(N - 1, Math.max(0, Math.floor(t * fps + 1e-6)));
        time = t;
        const tok = ++seekTok;
        seeking = true;
        c.dispatchEvent(new Event('seeking'));
        want(n).then(async e => {
          if (tok !== seekTok) return;
          await present(e);
          if (playing) startClock();
          seeking = false;
          c.dispatchEvent(new Event('seeked'));
        }, () => { if (tok === seekTok) { seeking = false; c.dispatchEvent(new Event('seeked')); } });
      },
    },
  });
  c.play = async () => { if (playing || err) return; if (cur >= N - 1) { await present(await want(0)); } playing = true; startClock(); raf = requestAnimationFrame(loop); };
  c.pause = () => { playing = false; cancelAnimationFrame(raf); time = (cur + 0.5) / fps; };
  c.requestVideoFrameCallback = cb => { rvfc.push(cb); return rvfc.length; };
  c.cancelVideoFrameCallback = () => {};
  c.load = () => {};
  /** 'full' | 'standard'; resolves when the frame on screen has been redrawn with it */
  c.setGrainMode = async m => {
    if (m !== 'full' && m !== 'standard') throw new Error('grain mode: full or standard');
    if (m === mode) return;
    mode = m;
    if (cur >= 0) await draw(await want(cur));
    c.dispatchEvent(new Event('seeked'));
  };
  /** frame n with the current grain, as packed planes (Y, U, V; 16-bit little-endian above 8 bits) */
  c.getFrame = async n => {
    const e = await want(n), pic = P.picture(n);
    if (gg) {
      await drawing;
      gg.setSource(e.gpu, e.planes, e.bps, e.semi);
      gg.setGrain(pic, mode);
      await gg.run();
      return gg.readPacked();
    }
    return (await wg.apply(e.bytes, e.planes, e.bps, fmt, pic, mode)).bytes;
  };
  c.dispose = () => {
    disposed = true; playing = false; cancelAnimationFrame(raf);
    if (dec && dec.state !== 'closed') dec.close();
    if (wg) wg.dispose();
    for (const e of cache.values()) if (e.gpu) e.gpu.destroy();
    cache.clear(); pending.clear();
  };
  return c;
}
