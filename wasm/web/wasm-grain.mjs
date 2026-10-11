// av1-fullgrain: CPU grain (fg-grain.wasm in grain-worker.mjs), for browsers without WebGPU.
//
//   const wg = await createWasmGrain({threads});
//   const packed = await wg.apply(bytes, planes, bps, fmt, pic, 'full');   // Uint8Array: Y, U, V planes
//
// Copyright © 2026, divergentnn (av1-fullgrain). SPDX-License-Identifier: BSD-2-Clause
import { paramsInt32 } from './fg-params.mjs';

/** [] when the WebAssembly grain can run here, otherwise the reasons */
export function wasmGrainSupport() {
  const why = [];
  if (typeof WebAssembly !== 'object') why.push('no WebAssembly');
  else if (!WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10,
    10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]))) why.push('no WebAssembly SIMD');
  if (!self.crossOriginIsolated || typeof SharedArrayBuffer !== 'function') why.push('page is not cross-origin isolated');
  return why;
}

export async function createWasmGrain(opts = {}) {
  const threads = opts.threads || Math.max(2, Math.min(8, (navigator.hardwareConcurrency || 4) - 1));
  const worker = new Worker(opts.workerUrl || new URL('./grain-worker.mjs', import.meta.url), { type: 'module' });
  const pending = new Map();
  let nextId = 1;
  const opened = await new Promise((res, rej) => {
    worker.onmessage = e => (e.data.type === 'opened' ? res(e.data) : rej(new Error(e.data.message)));
    worker.onerror = e => rej(new Error(e.message || 'grain worker failed to start'));
    worker.postMessage({ type: 'open', threads });
  });
  worker.onmessage = e => {
    const m = e.data, p = pending.get(m.id);
    if (!p) return;
    pending.delete(m.id);
    if (m.type === 'done') p.res({ bytes: new Uint8Array(m.bytes), ms: m.ms });
    else p.rej(new Error(m.message));
  };
  return {
    threads: opened.threads,
    /** grain one picture: bytes + planes [{offset, stride}] x3, bps; fmt {width, height, bitdepth, layout, mtrx};
     *  pic: loadParams(...).picture(n) or null -> {bytes: packed planes, ms} */
    apply(bytes, planes, bps, fmt, pic, mode = 'full') {
      const id = nextId++;
      return new Promise((res, rej) => {
        pending.set(id, { res, rej });
        worker.postMessage({ type: 'apply', id, mode, fmt, bytes, planes, bps, params: pic ? paramsInt32(pic) : null });
      });
    },
    dispose() { worker.terminate(); for (const [, p] of pending) p.rej(new Error('disposed')); pending.clear(); },
  };
}
