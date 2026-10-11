// A minimal MP4 demuxer for the AV1 player: the first AV1 track's samples
// (offset, size, timestamps, sync flag), its av1C, colour box and the
// WebCodecs codec string. Non-fragmented MP4 only (what ffmpeg -c copy and
// tools/av1grain_mp4.py write).
//
// Copyright © 2026, divergentnn (av1-fullgrain). SPDX-License-Identifier: BSD-2-Clause

function boxes(dv, off, end) {
  const out = [];
  while (off + 8 <= end) {
    let size = dv.getUint32(off), hdr = 8;
    const type = String.fromCharCode(dv.getUint8(off + 4), dv.getUint8(off + 5), dv.getUint8(off + 6), dv.getUint8(off + 7));
    if (size === 1) { size = Number(dv.getBigUint64(off + 8)); hdr = 16; } else if (size === 0) size = end - off;
    if (size < hdr || off + size > end) throw new Error(`MP4: broken ${type} box`);
    out.push({ type, off, size, hdr, body: off + hdr, end: off + size });
    off += size;
  }
  return out;
}
const child = (dv, b, type) => b && boxes(dv, b.body, b.end).find(x => x.type === type);
const path = (dv, b, ...types) => types.reduce((x, t) => child(dv, x, t), b);

/** -> {codec, width, height, bitdepth, timescale, duration, samples: [{offset, size, dts, cts, dur, sync}],
 *      av1C (Uint8Array, the record), colr: {primaries, transfer, matrix, fullRange} | null}, or null without AV1 */
export function demuxAv1(buffer) {
  const dv = new DataView(buffer instanceof ArrayBuffer ? buffer : buffer.buffer, buffer.byteOffset || 0, buffer.byteLength);
  const top = boxes(dv, 0, dv.byteLength);
  if (top.some(b => b.type === 'moof')) throw new Error('fragmented MP4 is not supported');
  const moov = top.find(b => b.type === 'moov');
  if (!moov) throw new Error('not an MP4 file (no moov)');
  for (const trak of boxes(dv, moov.body, moov.end).filter(b => b.type === 'trak')) {
    const stbl = path(dv, trak, 'mdia', 'minf', 'stbl');
    const stsd = child(dv, stbl, 'stsd');
    if (!stsd) continue;
    const entry = boxes(dv, stsd.body + 8, stsd.end)[0];
    if (!entry || entry.type !== 'av01') continue;
    const width = dv.getUint16(entry.body + 24), height = dv.getUint16(entry.body + 26);
    const sub = boxes(dv, entry.body + 78, entry.end);
    const av1c = sub.find(b => b.type === 'av1C'), colrBox = sub.find(b => b.type === 'colr');
    const av1C = av1c ? new Uint8Array(dv.buffer, dv.byteOffset + av1c.body, av1c.end - av1c.body) : null;
    let codec = 'av01.0.08M.08', bitdepth = 8;
    if (av1C && av1C.length >= 4) {
      const profile = av1C[1] >> 5, level = av1C[1] & 31, tier = av1C[2] >> 7;
      bitdepth = av1C[2] & 0x20 ? 12 : av1C[2] & 0x40 ? 10 : 8;
      codec = `av01.${profile}.${String(level).padStart(2, '0')}${tier ? 'H' : 'M'}.${String(bitdepth).padStart(2, '0')}`;
    }
    let colr = null;
    if (colrBox && dv.getUint32(colrBox.body) === 0x6e636c78) {          // 'nclx'
      colr = { primaries: dv.getUint16(colrBox.body + 4), transfer: dv.getUint16(colrBox.body + 6),
               matrix: dv.getUint16(colrBox.body + 8), fullRange: !!(dv.getUint8(colrBox.body + 10) & 0x80) };
    }
    const mdhd = path(dv, trak, 'mdia', 'mdhd');
    const v1 = dv.getUint8(mdhd.body) === 1;
    const timescale = dv.getUint32(mdhd.body + (v1 ? 20 : 12));
    // sample table
    const u32s = (b, at, n) => Array.from({ length: n }, (_, i) => dv.getUint32(at + 4 * i));
    const stsz = child(dv, stbl, 'stsz');
    const constSize = dv.getUint32(stsz.body + 4), n = dv.getUint32(stsz.body + 8);
    const sizes = constSize ? new Array(n).fill(constSize) : u32s(stsz, stsz.body + 12, n);
    const stco = child(dv, stbl, 'stco'), co64 = child(dv, stbl, 'co64');
    const nc = dv.getUint32((stco || co64).body + 4);
    const chunks = stco ? u32s(stco, stco.body + 8, nc)
                        : Array.from({ length: nc }, (_, i) => Number(dv.getBigUint64(co64.body + 8 + 8 * i)));
    const stsc = child(dv, stbl, 'stsc'), ne = dv.getUint32(stsc.body + 4);
    const ent = Array.from({ length: ne }, (_, i) => [dv.getUint32(stsc.body + 8 + 12 * i), dv.getUint32(stsc.body + 12 + 12 * i)]);
    const samples = [];
    for (let c = 0, e = 0; c < nc; c++) {
      while (e + 1 < ne && ent[e + 1][0] - 1 <= c) e++;
      let off = chunks[c];
      for (let k = 0; k < ent[e][1] && samples.length < n; k++) {
        samples.push({ offset: off, size: sizes[samples.length], dts: 0, cts: 0, dur: 0, sync: false });
        off += sizes[samples.length - 1];
      }
    }
    const stts = child(dv, stbl, 'stts');
    for (let i = 0, k = 0, t = 0, nn = dv.getUint32(stts.body + 4); i < nn; i++) {
      const cnt = dv.getUint32(stts.body + 8 + 8 * i), d = dv.getUint32(stts.body + 12 + 8 * i);
      for (let j = 0; j < cnt && k < n; j++, k++) { samples[k].dts = samples[k].cts = t; samples[k].dur = d; t += d; }
    }
    const ctts = child(dv, stbl, 'ctts');
    if (ctts) {
      const ver = dv.getUint8(ctts.body);
      for (let i = 0, k = 0, nn = dv.getUint32(ctts.body + 4); i < nn; i++) {
        const cnt = dv.getUint32(ctts.body + 8 + 8 * i);
        const o = ver ? dv.getInt32(ctts.body + 12 + 8 * i) : dv.getUint32(ctts.body + 12 + 8 * i);
        for (let j = 0; j < cnt && k < n; j++, k++) samples[k].cts = samples[k].dts + o;
      }
    }
    const stss = child(dv, stbl, 'stss');
    if (stss) for (const s of u32s(stss, stss.body + 8, dv.getUint32(stss.body + 4))) { if (samples[s - 1]) samples[s - 1].sync = true; }
    else for (const s of samples) s.sync = true;
    const duration = samples.length ? samples[n - 1].dts + samples[n - 1].dur : 0;
    return { codec, width, height, bitdepth, timescale, duration, samples, av1C, colr,
             bytes: new Uint8Array(dv.buffer, dv.byteOffset, dv.byteLength) };
  }
  return null;
}
