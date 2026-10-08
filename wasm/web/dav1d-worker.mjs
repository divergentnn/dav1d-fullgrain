// dav1d-fullgrain decode worker (module Worker). One instance per clip.
//
// Messages in:
//   {type: 'open', buffer: ArrayBuffer (MP4), threads, mode}
//   {type: 'frame', n, id, keep}   decode and return frame n; also returns up to
//                                  `keep` frames just before n when it had to
//                                  decode them anyway (for backward stepping)
//   {type: 'mode', mode}           switch the grain mode (reopens the decoder)
// Messages out:
//   {type: 'opened', width, height, frames, bpc, layout, grain, mtrx, trc, full_range, codec}
//   {type: 'frame', n, id, w, h, bpc, layout, y, u, v, grain, mtrx, full_range}  (planes transferred)
//   {type: 'error', message}
//
// Copyright © 2026, divergentnn (dav1d-fullgrain). SPDX-License-Identifier: BSD-2-Clause
import createDav1d from './dav1d-fg.mjs';
import { createFile, MP4BoxBuffer } from './mp4box.all.mjs';   // vendor/mp4box, copied next to this file by build.sh

let M = null, dec = 0, mode = 'full', threads = 4;
let bytes = null, samples = [], config = null;
let nextIn = 0, nextOut = 0, needConfig = true;

function demux(buffer) {
  const f = createFile();
  let info = null, err = null;
  f.onReady = i => { info = i; };
  f.onError = e => { err = e; };
  f.appendBuffer(MP4BoxBuffer.fromArrayBuffer(buffer, 0));
  f.flush();
  if (!info) throw new Error('not an MP4 file' + (err ? ': ' + err : ''));
  const vt = info.videoTracks[0];
  if (!vt) throw new Error('no video track');
  if (!/^av01/.test(vt.codec)) return { codec: vt.codec };
  const trak = f.getTrackById(vt.id);
  const entry = trak.mdia.minf.stbl.stsd.entries[0];
  return {
    codec: vt.codec,
    config: entry.av1C && entry.av1C.configOBUs ? new Uint8Array(entry.av1C.configOBUs) : null,
    samples: trak.samples.slice().sort((a, b) => a.dts - b.dts)
      .map(s => ({ offset: s.offset, size: s.size, sync: !!s.is_sync })),
  };
}

function open() {
  if (dec) M._djs_close(dec);
  const ms = M.stringToNewUTF8(mode);
  dec = M._djs_open(threads, 0, ms);
  M._free(ms);
  if (!dec) throw new Error('dav1d_open failed');
  nextIn = nextOut = 0;
  needConfig = true;
}

function send(i) {
  const s = samples[i];
  let data = bytes.subarray(s.offset, s.offset + s.size);
  if (needConfig && config) {                  // sequence header from av1C before the first sample
    const d = new Uint8Array(config.length + data.length);
    d.set(config); d.set(data, config.length); data = d;
  }
  needConfig = false;
  const p = M._malloc(data.length);
  new Uint8Array(M.wasmMemory.buffer).set(data, p);
  const r = M._djs_send(dec, p, data.length);
  M._free(p);
  if (r < 0) throw new Error('decode error ' + r + ' at frame ' + i);
}

function grab(n) {
  const buf = M.wasmMemory.buffer;
  const w = M._djs_w(dec), h = M._djs_h(dec), bpc = M._djs_bpc(dec), layout = M._djs_layout(dec);
  const ssx = layout !== 3 ? 1 : 0, ssy = layout === 1 ? 1 : 0;   // DAV1D_PIXEL_LAYOUT_I420 = 1, I444 = 3
  const cw = (w + ssx) >> ssx, ch = (h + ssy) >> ssy, hbd = bpc > 8;
  const T = hbd ? Uint16Array : Uint8Array, bs = hbd ? 2 : 1;
  const plane = (i, pw, ph) => {
    const out = new T(pw * ph), stride = M._djs_stride(dec, i) / bs, base = M._djs_plane(dec, i) / bs;
    const src = new T(buf);
    for (let y = 0; y < ph; y++) out.set(src.subarray(base + y * stride, base + y * stride + pw), y * pw);
    return out;
  };
  const f = { n, w, h, bpc, layout, y: plane(0, w, h), grain: M._djs_grain(dec),
              mtrx: M._djs_mtrx(dec), trc: M._djs_trc(dec), full_range: M._djs_full_range(dec) };
  if (layout) { f.u = plane(1, cw, ch); f.v = plane(2, cw, ch); }
  return f;
}

// decode up to frame n; frames in [n - keep, n) that pass by are returned too
function decodeTo(n, keep) {
  if (n < 0 || n >= samples.length) throw new Error('frame ' + n + ' outside the clip');
  if (n < nextOut || n - nextOut > 32) {        // restart from the last keyframe at or before n
    let k = n;
    while (k > 0 && !samples[k].sync) k--;
    if (!(k < nextOut && n >= nextOut)) {
      M._djs_flush(dec);
      nextIn = nextOut = k; needConfig = true;
    }
  }
  const extra = [];
  for (;;) {
    while (!M._djs_queued(dec)) {
      if (nextIn < samples.length) send(nextIn++);
      else { M._djs_drain(dec); if (!M._djs_queued(dec)) throw new Error('decoder produced no frame ' + n); }
    }
    M._djs_pop(dec);
    const i = nextOut++;
    if (i === n) return { frame: grab(i), extra };
    if (i >= n - keep) extra.push(grab(i));
  }
}

const planes = f => [f.y.buffer, ...(f.u ? [f.u.buffer, f.v.buffer] : [])];

self.onmessage = async ev => {
  const m = ev.data;
  try {
    if (m.type === 'open') {
      threads = m.threads || 4; mode = m.mode || 'full';
      const d = demux(m.buffer);
      if (!d.samples) { postMessage({ type: 'opened', codec: d.codec, frames: 0 }); return; }
      if (!M) M = await createDav1d();
      bytes = new Uint8Array(m.buffer); samples = d.samples; config = d.config;
      open();
      const { frame } = decodeTo(0, 0);
      postMessage({ type: 'opened', codec: d.codec, frames: samples.length, width: frame.w, height: frame.h,
                    bpc: frame.bpc, layout: frame.layout, grain: frame.grain, mtrx: frame.mtrx,
                    trc: frame.trc, full_range: frame.full_range });
      postMessage(Object.assign({ type: 'frame', id: -1 }, frame), planes(frame));
    } else if (m.type === 'mode') {
      mode = m.mode; open();
      postMessage({ type: 'mode', mode });
    } else if (m.type === 'frame') {
      const { frame, extra } = decodeTo(m.n, m.keep || 0);
      for (const f of extra) postMessage(Object.assign({ type: 'frame', id: -2 }, f), planes(f));
      postMessage(Object.assign({ type: 'frame', id: m.id }, frame), planes(frame));
    }
  } catch (e) {
    postMessage({ type: 'error', message: String(e && e.message || e), id: m.id });
  }
};
