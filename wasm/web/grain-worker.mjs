// av1-fullgrain: the CPU grain worker (module Worker). Applies full-frame or
// standard AV1 grain with fg-grain.wasm (the fork's grain code, WebAssembly
// SIMD + threads; wasm/fgjs.c) to pictures the browser decoded without grain.
// Used where WebGPU is missing. Needs a cross-origin isolated page (threads).
//
// in:  {type: 'open', threads}
//      {type: 'apply', id, mode: 'full'|'standard', fmt: {width, height, bitdepth, layout, mtrx},
//       bytes: ArrayBuffer, planes: [{offset, stride}] x3 (bytes), bps, params: Int32Array | null}
// out: {type: 'opened', threads} | {type: 'done', id, bytes (packed Y, U, V), ms} | {type: 'error', id, message}
//
// Copyright © 2026, divergentnn (av1-fullgrain). SPDX-License-Identifier: BSD-2-Clause
import createGrain from './fg-grain.mjs';

let M = null, f = 0, cap = 0, base = 0, P = 0;

self.onmessage = async ev => {
  const m = ev.data;
  try {
    if (m.type === 'open') {
      M = await createGrain();
      f = M._fgjs_open(m.threads || 4);
      P = M._fgjs_alloc(f, 165 * 4);
      postMessage({ type: 'opened', threads: M._fgjs_threads(f) });
      return;
    }
    if (m.type !== 'apply') return;
    const t0 = performance.now();
    const { width: w, height: h, bitdepth: bd, layout, mtrx } = m.fmt;
    const bps = bd > 8 ? 2 : 1, mono = layout === 0;
    const ssx = layout === 1 || layout === 2 ? 1 : 0, ssy = layout === 1 ? 1 : 0;
    const cw = mono ? 0 : (w + ssx) >> ssx, ch = mono ? 0 : (h + ssy) >> ssy;
    const sy = Math.ceil(w * bps / 64) * 64 + 64, sc = Math.max(64, Math.ceil(cw * bps / 64) * 64 + 64);
    const plane = [sy * h, sc * ch, sc * ch], total = 2 * (plane[0] + plane[1] + plane[2]) + 4096;
    if (cap < total) {
      if (base) M._fgjs_free(base);
      base = M._fgjs_alloc(f, total); cap = total;
      if (!base) throw new Error('out of memory');
    }
    const inp = [base, base + plane[0], base + plane[0] + plane[1]];
    const out = inp.map(a => a + plane[0] + plane[1] + plane[2]);
    // copy the planes in, row by row
    let heap = new Uint8Array(M.wasmMemory.buffer);
    const src = new Uint8Array(m.bytes);
    const dims = [[w, h, sy], [cw, ch, sc], [cw, ch, sc]];
    for (let p = 0; p < (mono ? 1 : 3); p++) {
      const [pw, ph, st] = dims[p], { offset, stride } = m.planes[p];
      for (let y = 0; y < ph; y++) heap.set(src.subarray(offset + y * stride, offset + y * stride + pw * bps), inp[p] + y * st);
    }
    if (m.params) {
      new Int32Array(M.wasmMemory.buffer, P, 165).set(m.params);
      const r = M._fgjs_apply(f, m.mode === 'full' ? 1 : 0, w, h, bd, layout, mtrx, P,
                              inp[0], inp[1], inp[2], out[0], out[1], out[2], sy, sc);
      if (r < 0) throw new Error('fgjs_apply: bad parameters');
    }
    // copy the result out, packed (Y, U, V)
    heap = new Uint8Array(M.wasmMemory.buffer);
    const res = new Uint8Array((w * h + 2 * cw * ch) * bps);
    const from = m.params ? out : inp;
    let o = 0;
    for (let p = 0; p < (mono ? 1 : 3); p++) {
      const [pw, ph, st] = dims[p];
      for (let y = 0; y < ph; y++) { res.set(heap.subarray(from[p] + y * st, from[p] + y * st + pw * bps), o); o += pw * bps; }
    }
    postMessage({ type: 'done', id: m.id, bytes: res.buffer, ms: performance.now() - t0 }, [res.buffer]);
  } catch (e) {
    postMessage({ type: 'error', id: m.id, message: String(e && e.message || e) });
  }
};
