// av1-fullgrain on the GPU: AV1 film grain drawn with WebGPU compute shaders
// onto pictures decoded without grain (by the browser's own decoder, see
// native-clip.mjs), bit-identical to the dav1d fork's C code.
//
//   full      full-frame grain (src/fg_variant_tmpl.c, fgv_full_row): hashed
//             white noise, the signalled AR filter over the whole frame,
//             restarted for each group of 4 x 32 rows 16 rows above it.
//   standard  AV1 grain as the specification defines it (src/filmgrain_tmpl.c):
//             an 82 x 73 template per plane (LFSR noise + AR filter, run as a
//             wavefront in workgroup memory), 32 x 32 blocks copied from it at
//             per-block LFSR offsets, overlap blending at the block seams.
//
// Full-frame AR: one workgroup per band group and plane, one thread per row.
// A row's AR filter depends on its own previous samples, so rows run as a
// wavefront: in step t, row r filters columns [(t - K r) S, (t - K r + 1) S).
// The rows above are read from a ring of the last 8 steps' results in
// workgroup memory (the taps reach back at most 7 steps), into a 3 x 7
// register window that slides along the row. Global memory only receives the
// finished field (row-major, two 16-bit samples per word) for the next passes:
// storageBarrier() did not make one thread's global writes visible to the
// others reliably (Chromium / Vulkan / NVIDIA), workgroupBarrier() on
// workgroup memory does.
//
// Copyright © 2026, divergentnn (av1-fullgrain). SPDX-License-Identifier: BSD-2-Clause

import { gaussianTable, generateScaling, fullKeys } from './fg-params.mjs';

export const ROWS = 144;           // rows of one band group: 16 warm-up + 4 x 32
const S = 4, K = 2;                // columns per wavefront step; steps of skew per row (K = 2 needs S >= 3)
const RING = 8;                    // steps kept in workgroup memory (the oldest tap is 7 steps back)

// parameter block (a uniform array of 36 x vec4<i32>); indices:
const I = {
  W: 0, H: 1, SSX: 2, SSY: 3, BD: 4, CW: 5, CH: 6, NG: 7,
  LAG: 8, ARSH: 9, GMIN: 10, GMAX: 11, HASY: 12, HASU: 13, HASV: 14, KEYY: 15, KEYU: 16, KEYV: 17,
  SCSH: 18, YMIN: 19, YMAX: 20, CMIN: 21, CMAX: 22, CSFL: 23,
  ULM: 24, UM: 25, UOFF: 26, VLM: 27, VM: 28, VOFF: 29, LCU: 30, LCV: 31,
  RSL: 32, LPITCH: 33, LBW: 34, NCHL: 35, RSC: 36, CPITCH: 37, CBW: 38, NCHC: 39,
  NTC: 57, SOY: 40, SSY_: 41, SOU: 42, SSU: 43, SOV: 44, SSV: 45, BPS: 46,
  DOY: 47, DOU: 48, DOV: 49, DSY: 50, DSC: 51, MONO: 52, SCPU: 53, SCPV: 54, PMAX: 55, CPLANE: 56,
  MODE: 58, SEED: 59, OVL: 60, NBR: 61, NBC: 62, STEPY: 63, STEPC: 136,
  GRID: 64,                        // per plane: 21 previous-row taps (3 x 7, zero outside the lag), c1, c2, c3
};
const TPL = 74 * 82;               // a standard-mode template (GRAIN_HEIGHT + 1) x GRAIN_WIDTH, as dav1d allocates it
const GSTRIDE = 24;
const NPARAM = 144;

const WGSL = /* wgsl */`
struct Params { v: array<vec4<i32>, ${NPARAM / 4}> }
@group(0) @binding(0) var<uniform> U: Params;
@group(0) @binding(1) var<storage, read> tabs: array<i32>;        // Gaussian table (2048), then 3 scaling LUTs (4096 each)
@group(0) @binding(2) var<storage, read_write> FL: array<u32>;
@group(0) @binding(3) var<storage, read_write> FC: array<u32>;
@group(0) @binding(4) var<storage, read_write> SB: array<i32>;   // standard mode: 3 templates, then the block offsets
@group(0) @binding(5) var<storage, read> src: array<u32>;
@group(0) @binding(6) var<storage, read_write> dst: array<u32>;
@group(0) @binding(7) var<storage, read_write> LT: array<i32>;

const S = ${S};
const K = ${K};

fn p(i: i32) -> i32 { return U.v[i >> 2][i & 3]; }

fn hash(x0: u32) -> u32 {
  var x = x0;
  x = x ^ (x >> 16u); x = x * 0x7feb352du;
  x = x ^ (x >> 15u); x = x * 0x846ca68bu;
  return x ^ (x >> 16u);
}

fn round2(x: i32, sh: i32) -> i32 { return (x + ((1 << u32(sh)) >> 1u)) >> u32(sh); }

// the finished fields: row-major per band group, two 16-bit samples per word
fn lword(g: i32, r: i32, c: i32) -> i32 { return ((g * p(${I.RSL}) + r) * p(${I.LPITCH}) + c) >> 1u; }
fn cword(pl: i32, g: i32, r: i32, c: i32) -> i32 {
  let n = p(${I.NG}) * p(${I.RSC}) * p(${I.CPITCH});
  return ((pl - 1) * n + (g * p(${I.RSC}) + r) * p(${I.CPITCH}) + c) >> 1u;
}
fn half(w: u32, c: i32) -> i32 {
  if ((c & 1) == 0) { return bitcast<i32>(w << 16u) >> 16u; }
  return bitcast<i32>(w) >> 16u;
}
fn lfield(g: i32, r: i32, c: i32) -> i32 { return half(FL[lword(g, r, c)], c); }
fn cfield(pl: i32, g: i32, r: i32, c: i32) -> i32 { return half(FC[cword(pl, g, r, c)], c); }

// the chroma AR's luma term, round2(sum of the 1, 2 or 4 luma grain samples, ssx + ssy), for chroma (r, c) of
// group g; stored in the order the AR reads it (step-major: the rows of one step are adjacent)
fn ltidx(g: i32, r: i32, c: i32) -> i32 {
  return ((g * p(${I.NTC}) + c / S + K * r) * S + c % S) * p(${I.RSC}) + r;
}

@compute @workgroup_size(64)
fn lumaterm(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = i32(id.x); let r = i32(id.y); let g = i32(id.z);
  let ssx = p(${I.SSX}); let ssy = p(${I.SSY});
  let bhc = (min(p(${I.H}) - g * 128, 128) + ssy) >> u32(ssy);
  if (c >= p(${I.CBW}) || r >= (16 >> u32(ssy)) + bhc) { return; }
  let rl = r << u32(ssy); let cl = c << u32(ssx);
  var l = lfield(g, rl, cl);
  if (ssx != 0) { l += lfield(g, rl, cl + 1); }
  if (ssy != 0) {
    l += lfield(g, rl + 1, cl);
    if (ssx != 0) { l += lfield(g, rl + 1, cl + 1); }
  }
  let ssum = u32(ssx + ssy);
  LT[ltidx(g, r, c)] = (l + ((1 << ssum) >> 1u)) >> ssum;
}

// the last ${RING} steps' results of every row: [step % ${RING}][column % S / 2][row], 2 samples per word
var<workgroup> ring: array<u32, ${RING * (S / 2) * ROWS}>;

fn rread(r: i32, c: i32) -> i32 {
  let slot = (c / S + K * r) & ${RING - 1};
  return half(ring[(slot * ${S / 2} + ((c % S) >> 1u)) * ${ROWS} + r], c);
}

// one band group of one plane: the AR recursion of fgv_full_row / ar_band
fn ar(pl: i32, g: i32, r: i32) {
  let H = p(${I.H}); let ssx = p(${I.SSX}); let ssy = p(${I.SSY});
  let Y0 = g * 128;
  let bh = min(H - Y0, 128);
  let bhc = (bh + ssy) >> u32(ssy);
  let hasC = p(${I.HASU}) != 0 || p(${I.HASV}) != 0;
  var R: i32; var bw: i32; var y0: i32; var x0: i32; var nch: i32; var key: u32; var lc = 0;
  if (pl == 0) {
    R = 16 + max(bh, select(0, bhc << u32(ssy), hasC));
    bw = p(${I.LBW}); y0 = Y0 - 16; x0 = -16; nch = p(${I.NCHL});
    key = bitcast<u32>(p(${I.KEYY}));
  } else {
    let Pc = 16 >> u32(ssy);
    R = Pc + bhc;
    bw = p(${I.CBW}); y0 = (Y0 >> u32(ssy)) - Pc; x0 = -(16 >> u32(ssx)); nch = p(${I.NCHC});
    key = bitcast<u32>(p(${I.KEYY} + pl));
    lc = p(${I.LCU} + pl - 1);
  }
  let G = ${I.GRID} + ${GSTRIDE} * pl;
  ${[0, 1, 2].map(d => [0, 1, 2, 3, 4, 5, 6].map(k => `let k${d}${k} = p(G + ${d * 7 + k});`).join(' ')).join('\n  ')}
  let c1 = p(G + 21); let c2 = p(G + 22); let c3 = p(G + 23);
  let sh = u32(p(${I.ARSH}));
  let gmin = p(${I.GMIN}); let gmax = p(${I.GMAX});
  let rnd = (1 << sh) >> 1u;
  let rk = hash(key ^ hash(bitcast<u32>(y0 + r) * 0x9E3779B9u + 0x7f4a7c15u));
  let ar_row = r >= 3 && r < R;
  // window: a = row r - 1, b = r - 2, d = r - 3; position j holds column c - 3 + j
  ${['a', 'b', 'd'].map(w => [0, 1, 2, 3, 4, 5, 6].map(k => `var ${w}${k} = 0;`).join(' ')).join('\n  ')}
  var p1 = 0; var p2 = 0; var p3 = 0;
  var pend = 0u;
  let steps = nch + K * (R - 1);
  for (var t = 0; t < steps; t++) {
    let q = t - K * r;
    if (r < R && q >= 0 && q < nch) {
      if (q == 0 && ar_row) {
        a4 = rread(r - 1, 0); a5 = rread(r - 1, 1); a6 = rread(r - 1, 2);
        b4 = rread(r - 2, 0); b5 = rread(r - 2, 1); b6 = rread(r - 2, 2);
        d4 = rread(r - 3, 0); d5 = rread(r - 3, 1); d6 = rread(r - 3, 2);
      }
      var n = 0;
      for (var i = 0; i < S; i++) {
        let c = q * S + i;
        if (c >= bw) { break; }
        ${['a', 'b', 'd'].map(w => [0, 1, 2, 3, 4, 5].map(k => `${w}${k} = ${w}${k + 1};`).join(' ')).join('\n        ')}
        if (ar_row && c + 3 < bw) {
          a6 = rread(r - 1, c + 3); b6 = rread(r - 2, c + 3); d6 = rread(r - 3, c + 3);
        }
        let x = x0 + c;
        let h = hash(rk + bitcast<u32>(x >> 1u) * 0x85ebca6bu);
        var v = tabs[select((h >> 5u) & 0x7ffu, h >> 21u, (x & 1) != 0)];
        if (ar_row && c >= 3 && c < bw - 3) {
          var sum = ${[['a', 0], ['b', 1], ['d', 2]].map(([w, d]) => [0, 1, 2, 3, 4, 5, 6].map(k => `k${2 - d}${k} * ${w}${k}`).join(' + ')).join('\n            + ')};
          if (lc != 0) { sum += LT[ltidx(g, r, c)] * lc; }
          sum += c3 * p3 + c2 * p2 + c1 * p1 + rnd;
          v = clamp(v + (sum >> sh), gmin, gmax);
        }
        p3 = p2; p2 = p1; p1 = v;
        if ((i & 1) == 0) {
          pend = u32(v) & 0xffffu;
        } else {
          let w = pend | (u32(v) << 16u);
          ring[((t & ${RING - 1}) * ${S / 2} + (i >> 1u)) * ${ROWS} + r] = w;
          if (pl == 0) { FL[lword(g, r, c - 1)] = w; } else { FC[cword(pl, g, r, c - 1)] = w; }
        }
        n++;
      }
      if ((n & 1) != 0) {            // the row's last column, alone in its word
        let c = q * S + n - 1;
        ring[((t & ${RING - 1}) * ${S / 2} + ((n - 1) >> 1u)) * ${ROWS} + r] = pend;
        if (pl == 0) { FL[lword(g, r, c)] = pend; } else { FC[cword(pl, g, r, c)] = pend; }
      }
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(${ROWS})
fn ar_luma(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  ar(0, i32(wg.x), i32(li));
}

@compute @workgroup_size(${ROWS})
fn ar_chroma(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  ar(p(${I.CPLANE}) + i32(wg.y), i32(wg.x), i32(li));
}

// a sample of the input picture (8 or 16 bits, any offsets / strides in bytes)
fn rsrc(pl: i32, x: i32, y: i32) -> i32 {
  let bps = p(${I.BPS});
  let b = p(${I.SOY} + 2 * pl) + y * p(${I.SSY_} + 2 * pl) + x * select(p(${I.STEPC}), p(${I.STEPY}), pl == 0);
  let w = src[b >> 2u];
  if (bps == 1) { return i32((w >> (u32(b & 3) * 8u)) & 0xffu); }
  return i32((w >> (u32(b & 2) * 8u)) & 0xffffu);
}

fn noise(sc: i32, lut: i32, gr: i32) -> i32 {
  return round2(tabs[2048 + lut * 4096 + sc] * gr, p(${I.SCSH}));
}

// ---- standard AV1 grain (src/filmgrain_tmpl.c)

fn lfsr(st: ptr<function, u32>, bits: u32) -> u32 {
  let r = *st;
  let bit = (r ^ (r >> 1u) ^ (r >> 3u) ^ (r >> 12u)) & 1u;
  *st = (r >> 1u) | (bit << 15u);
  return (*st >> (16u - bits)) & ((1u << bits) - 1u);
}

// the template of one plane, packed two samples per word (41 words per row), in workgroup memory
var<workgroup> tw: array<u32, ${74 * 41}>;

fn std_tpl(pl: i32, r: i32) {
  let ssx = select(0, p(${I.SSX}), pl > 0); let ssy = select(0, p(${I.SSY}), pl > 0);
  let TW = select(82, 44, ssx != 0); let TH = select(73, 38, ssy != 0);
  if (r == 0) {                                          // LFSR noise, raster order (one thread: it is serial)
    var st = u32(p(${I.SEED})) ^ select(select(0u, 0xb524u, pl == 1), 0x49d8u, pl == 2);
    for (var y = 0; y < TH; y++) {
      for (var x = 0; x < TW; x += 2) {
        let v0 = tabs[lfsr(&st, 11u)]; let v1 = tabs[lfsr(&st, 11u)];
        tw[y * 41 + (x >> 1u)] = (u32(v0) & 0xffffu) | (u32(v1) << 16u);
      }
    }
  }
  workgroupBarrier();
  // AR filter: rows [3, TH) x columns [3, TW - 3); row r handles column t - 5 r in step t. A thread writes a word
  // when both its samples are done, so a row's sample is readable 1 step late: skew 5 (lag 3 needs 4) covers it.
  let G = ${I.GRID} + ${GSTRIDE} * pl;
  ${[0, 1, 2].map(d => [0, 1, 2, 3, 4, 5, 6].map(k => `let k${d}${k} = p(G + ${d * 7 + k});`).join(' ')).join('\n  ')}
  let c1 = p(G + 21); let c2 = p(G + 22); let c3 = p(G + 23);
  let lc = select(0, p(${I.LCU} + pl - 1), pl > 0);
  let sh = u32(p(${I.ARSH})); let rnd = (1 << sh) >> 1u;
  let gmin = p(${I.GMIN}); let gmax = p(${I.GMAX});
  var p1 = 0; var p2 = 0; var p3 = 0; var hold = 0u;
  let steps = TW + 5 * (TH - 1);
  for (var t = 0; t < steps; t++) {
    let c = t - 5 * r;
    if (r >= 3 && r < TH && c >= 0 && c < TW) {
      let w0 = tw[r * 41 + (c >> 1u)];
      var v = half(w0, c);
      if (c >= 3 && c < TW - 3) {
        var sum = 0;
        ${[0, 1, 2].map(d => [0, 1, 2, 3, 4, 5, 6].map(k => `sum += k${d}${k} * half(tw[(r - ${3 - d}) * 41 + ((c + ${k - 3}) >> 1u)], c + ${k - 3});`).join('\n        ')).join('\n        ')}
        if (lc != 0) {
          let lx = ((c - 3) << u32(ssx)) + 3; let ly = ((r - 3) << u32(ssy)) + 3;
          var l = 0;
          for (var i = 0; i <= ssy; i++) { for (var j = 0; j <= ssx; j++) { l += SB[(ly + i) * 82 + lx + j]; } }
          sum += round2(l, ssx + ssy) * lc;
        }
        sum += c3 * p3 + c2 * p2 + c1 * p1 + rnd;
        v = clamp(v + (sum >> sh), gmin, gmax);
      }
      p3 = p2; p2 = p1; p1 = v;
      if ((c & 1) == 0) { hold = u32(v) & 0xffffu; } else { tw[r * 41 + (c >> 1u)] = hold | (u32(v) << 16u); }
    }
    workgroupBarrier();
  }
  for (var i = r; i < TH * TW; i += ${128}) {
    let y = i / TW; let x = i % TW;
    SB[pl * ${TPL} + y * 82 + x] = half(tw[y * 41 + (x >> 1u)], x);
  }
}

@compute @workgroup_size(128)
fn std_tpl_luma(@builtin(local_invocation_index) li: u32) { std_tpl(0, i32(li)); }

@compute @workgroup_size(128)
fn std_tpl_chroma(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  std_tpl(p(${I.CPLANE}) + i32(wg.x), i32(li));
}

// the 8-bit block offsets of every 32 x 32 block: one LFSR per block row, one number per block
@compute @workgroup_size(64)
fn std_offsets(@builtin(global_invocation_id) id: vec3<u32>) {
  let R = i32(id.x);
  if (R >= p(${I.NBR})) { return; }
  let s = u32(p(${I.SEED}));
  var st = s ^ ((u32(R * 37 + 178) & 0xffu) << 8u) ^ (u32(R * 173 + 105) & 0xffu);
  let n = p(${I.NBC});
  for (var c = 0; c < n; c++) { SB[${3 * TPL} + R * n + c] = i32(lfsr(&st, 8u)); }
}

// sample_lut(): block (C - bxo, R - byo)'s template at (x, y) of the current block, + its overlap extension
fn slut(pl: i32, ssx: i32, ssy: i32, R: i32, C: i32, bxo: i32, byo: i32, x: i32, y: i32) -> i32 {
  let off = SB[${3 * TPL} + (R - byo) * p(${I.NBC}) + C - bxo];
  let ox = 3 + (2 >> u32(ssx)) * (3 + (off >> 4u));
  let oy = 3 + (2 >> u32(ssy)) * (3 + (off & 15));
  return SB[pl * ${TPL} + (oy + y + (32 >> u32(ssy)) * byo) * 82 + ox + x + (32 >> u32(ssx)) * bxo];
}

fn blend(old: i32, cur: i32, w0: i32, w1: i32) -> i32 {
  return clamp(round2(old * w0 + cur * w1, 5), p(${I.GMIN}), p(${I.GMAX}));
}

// overlap weights: w[sub][pos] -> (old, new)
fn wold(sub: i32, i: i32) -> i32 { return select(select(17, 27, i == 0), 23, sub != 0); }
fn wnew(sub: i32, i: i32) -> i32 { return select(select(27, 17, i == 0), 22, sub != 0); }

// the grain of plane pl at (x, y): full-frame field or standard blocks
fn grain_at(pl: i32, x: i32, y: i32) -> i32 {
  let ssx = select(0, p(${I.SSX}), pl > 0); let ssy = select(0, p(${I.SSY}), pl > 0);
  if (p(${I.MODE}) == 0) {
    if (pl == 0) { let g = y / 128; return lfield(g, 16 + y - g * 128, 16 + x); }
    let g = (y << u32(ssy)) / 128;
    return cfield(pl, g, (16 >> u32(ssy)) + y - ((g * 128) >> u32(ssy)), (16 >> u32(ssx)) + x);
  }
  let bw = 32 >> u32(ssx); let bh = 32 >> u32(ssy);
  let R = y / bh; let C = x / bw; let lx = x - C * bw; let ly = y - R * bh;
  let ov = p(${I.OVL}) != 0;
  let vx = ov && C > 0 && lx < (2 >> u32(ssx));
  let vy = ov && R > 0 && ly < (2 >> u32(ssy));
  var g = slut(pl, ssx, ssy, R, C, 0, 0, lx, ly);
  if (vx) { g = blend(slut(pl, ssx, ssy, R, C, 1, 0, lx, ly), g, wold(ssx, lx), wnew(ssx, lx)); }
  if (vy) {
    var top = slut(pl, ssx, ssy, R, C, 0, 1, lx, ly);
    if (vx) { top = blend(slut(pl, ssx, ssy, R, C, 1, 1, lx, ly), top, wold(ssx, lx), wnew(ssx, lx)); }
    g = blend(top, g, wold(ssy, ly), wnew(ssy, ly));
  }
  return g;
}

@compute @workgroup_size(16, 8)
fn apply_y(@builtin(global_invocation_id) id: vec3<u32>) {
  let W = p(${I.W}); let H = p(${I.H});
  let x0 = i32(id.x) * 2; let y = i32(id.y);
  if (x0 >= W || y >= H) { return; }
  var word = 0u;
  for (var k = 0; k < 2; k++) {
    let x = x0 + k;
    if (x >= W) { break; }
    let s = rsrc(0, x, y);
    var o = s;
    if (p(${I.HASY}) != 0) {
      o = clamp(s + noise(s, 0, grain_at(0, x, y)), p(${I.YMIN}), p(${I.YMAX}));
    }
    word |= u32(o) << (16u * u32(k));
  }
  dst[(p(${I.DOY}) + y * p(${I.DSY}) + x0) >> 1u] = word;
}

@compute @workgroup_size(16, 8)
fn apply_c(@builtin(global_invocation_id) id: vec3<u32>) {
  let pl = 1 + i32(id.z);
  let CW = p(${I.CW}); let CH = p(${I.CH}); let W = p(${I.W});
  let ssx = p(${I.SSX}); let ssy = p(${I.SSY});
  let x0 = i32(id.x) * 2; let y = i32(id.y);
  if (x0 >= CW || y >= CH) { return; }
  let has = p(${I.HASY} + pl) != 0;
  let lm = p(${I.ULM} + 3 * (pl - 1)); let m = p(${I.UM} + 3 * (pl - 1)); let off = p(${I.UOFF} + 3 * (pl - 1));
  let lut = p(${I.SCPU} + pl - 1);
  var word = 0u;
  for (var k = 0; k < 2; k++) {
    let x = x0 + k;
    if (x >= CW) { break; }
    let s = rsrc(pl, x, y);
    var o = s;
    if (has) {
      let lx = x << u32(ssx); let ly = y << u32(ssy);
      var avg = rsrc(0, lx, ly);
      if (ssx != 0) { avg = (avg + rsrc(0, min(lx + 1, W - 1), ly) + 1) >> 1u; }
      var val = avg;
      if (p(${I.CSFL}) == 0) { val = clamp(((avg * lm + s * m) >> 6u) + off, 0, p(${I.PMAX})); }
      let gr = grain_at(pl, x, y);
      o = clamp(s + noise(val, lut, gr), p(${I.CMIN}), p(${I.CMAX}));
    }
    word |= u32(o) << (16u * u32(k));
  }
  let dof = select(p(${I.DOV}), p(${I.DOU}), pl == 1);
  dst[(dof + y * p(${I.DSC}) + x0) >> 1u] = word;
}
`;

const adapters = new Map();
/** the WebGPU adapter (cached; the first request can return null while the GPU process starts, so ask twice) */
export async function gpuAdapter(powerPreference = 'high-performance') {
  if (!adapters.has(powerPreference)) {
    adapters.set(powerPreference, (async () => {
      for (let i = 0; i < 2; i++) {
        const a = await navigator.gpu.requestAdapter({ powerPreference });
        if (a) return a;
      }
      return null;
    })());
  }
  return adapters.get(powerPreference);
}

/** [] when WebGPU is usable here, otherwise the reasons */
export async function gpuSupport(powerPreference = 'high-performance') {
  if (!self.isSecureContext) return ['not a secure context'];
  if (!navigator.gpu) return ['no WebGPU'];
  try {
    if (!await gpuAdapter(powerPreference)) return ['no WebGPU adapter'];
  } catch (e) { return ['WebGPU adapter: ' + e.message]; }
  return [];
}

const devices = new Map();
/** the page's WebGPU device for this power preference (one for every GpuGrain): {adapter, device} */
export function gpuDevice(powerPreference = 'high-performance') {
  if (!devices.has(powerPreference)) {
    devices.set(powerPreference, (async () => {
      let adapter = null;                       // a fresh adapter: an adapter makes one device only
      for (let i = 0; i < 2 && !adapter; i++) adapter = await navigator.gpu.requestAdapter({ powerPreference });
      if (!adapter) throw new Error('no WebGPU adapter');
      const features = adapter.features.has('timestamp-query') ? ['timestamp-query'] : [];
      const device = await adapter.requestDevice({
        requiredFeatures: features,
        requiredLimits: {
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxBufferSize: adapter.limits.maxBufferSize,
          maxComputeInvocationsPerWorkgroup: Math.max(256, ROWS),
          maxComputeWorkgroupSizeX: Math.max(256, ROWS),
        },
      });
      device.lost.then(() => devices.delete(powerPreference));
      return { adapter, device };
    })());
    devices.get(powerPreference).catch(() => devices.delete(powerPreference));
  }
  return devices.get(powerPreference);
}

/** a grain renderer with its own buffers on the shared device (or opts.device + opts.adapter) */
export async function createGpuGrain(opts = {}) {
  const { adapter, device } = opts.device ? opts : await gpuDevice(opts.powerPreference || 'high-performance');
  return new GpuGrain(adapter, device, device.features.has('timestamp-query'));
}

const align = (x, a) => Math.ceil(x / a) * a;
const pipeCache = new WeakMap();

// AR coefficients (AV1 order: rows -lag..-1 x columns -lag..lag, then -lag..-1 in the row) as a fixed 3 x 7 grid
// of previous-row taps (row -3 first) and the in-row taps c1, c2, c3 (columns -1, -2, -3)
function grid(P, at, co, lag) {
  P.fill(0, at, at + GSTRIDE);
  let k = 0;
  for (let dy = -lag; dy < 0; dy++)
    for (let dx = -lag; dx <= lag; dx++) P[at + (dy + 3) * 7 + dx + 3] = co[k++];
  for (let dx = -lag; dx < 0; dx++) P[at + 20 - dx] = co[k++];
}

export class GpuGrain {
  constructor(adapter, device, timestamps) {
    this.I = I;
    this.adapter = adapter;
    this.device = device;
    this.info = adapter.info || {};
    this.timestamps = timestamps;
    if (!pipeCache.has(device)) {               // shader + pipelines once per device
      const module = device.createShaderModule({ code: WGSL });
      const B = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
      const bgl = device.createBindGroupLayout({ entries: [
        B(0, 'uniform'), B(1, 'read-only-storage'), B(2, 'storage'), B(3, 'storage'),
        B(4, 'storage'), B(5, 'read-only-storage'), B(6, 'storage'), B(7, 'storage'),
      ] });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });
      const pipe = entryPoint => device.createComputePipeline({ layout, compute: { module, entryPoint } });
      pipeCache.set(device, { bgl, pipes: {
        ar_luma: pipe('ar_luma'), ar_chroma: pipe('ar_chroma'), lumaterm: pipe('lumaterm'),
        std_tpl_luma: pipe('std_tpl_luma'), std_tpl_chroma: pipe('std_tpl_chroma'), std_offsets: pipe('std_offsets'),
        apply_y: pipe('apply_y'), apply_c: pipe('apply_c') } });
    }
    ({ bgl: this.bgl, pipes: this.pipes } = pipeCache.get(device));
    this.params = new Int32Array(NPARAM);
    this.uni = device.createBuffer({ size: NPARAM * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.tabs = device.createBuffer({ size: (2048 + 3 * 4096) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.fmt = null;
    this.lastSet = null;
    if (timestamps) {
      this.qs = device.createQuerySet({ type: 'timestamp', count: 8 });
      this.qbuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.qread = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
  }

  /** picture format: {width, height, bitdepth, layout (dav1d: 0 I400, 1 I420, 2 I422, 3 I444), mtrx} */
  configure(f) {
    const k = JSON.stringify([f.width, f.height, f.bitdepth, f.layout, f.mtrx]);
    if (this.fmtKey === k) return;
    this.fmtKey = k;
    this.fmt = f;
    const P = this.params, W = f.width, H = f.height, mono = f.layout === 0;
    const ssx = f.layout === 1 || f.layout === 2 ? 1 : 0, ssy = f.layout === 1 ? 1 : 0;
    const CW = mono ? 0 : (W + ssx) >> ssx, CH = mono ? 0 : (H + ssy) >> ssy;
    const NG = Math.ceil(H / 128);
    const lbw = 16 + W + 8, nchl = Math.ceil(lbw / S), lpitch = align(lbw, 4);
    const rsc = ROWS >> ssy, cbw = (16 >> ssx) + CW + (8 >> ssx), nchc = Math.ceil(cbw / S), cpitch = align(cbw, 4);
    Object.assign(this, { W, H, CW, CH, NG, ssx, ssy, mono });
    P.fill(0);
    P[I.W] = W; P[I.H] = H; P[I.SSX] = ssx; P[I.SSY] = ssy; P[I.BD] = f.bitdepth; P[I.CW] = CW; P[I.CH] = CH; P[I.NG] = NG;
    P[I.RSL] = ROWS; P[I.LPITCH] = lpitch; P[I.LBW] = lbw; P[I.NCHL] = nchl;
    P[I.RSC] = rsc; P[I.CPITCH] = cpitch; P[I.CBW] = cbw; P[I.NCHC] = nchc;
    const ntc = nchc + K * (rsc - 1);
    P[I.NTC] = ntc;
    P[I.MONO] = mono ? 1 : 0; P[I.PMAX] = (1 << f.bitdepth) - 1;
    // output: 16-bit samples, rows padded to an even count
    const dsy = align(W, 2), dsc = align(CW, 2);
    P[I.DOY] = 0; P[I.DSY] = dsy; P[I.DOU] = dsy * H; P[I.DOV] = dsy * H + dsc * CH; P[I.DSC] = dsc;
    this.dstSamples = dsy * H + 2 * dsc * CH;
    const dev = this.device;
    if (this.srcBuf && !this.ownSrc) this.srcBuf = null;      // not ours to destroy
    for (const b of ['fl', 'fc', 'lt', 'sb', 'dst', 'srcBuf', 'readBuf']) if (this[b]) { this[b].destroy(); this[b] = null; }
    P[I.NBR] = Math.ceil(H / 32); P[I.NBC] = Math.ceil(W / 32);
    this.sb = dev.createBuffer({ size: (3 * TPL + P[I.NBR] * P[I.NBC]) * 4, usage: GPUBufferUsage.STORAGE });
    this.fl = dev.createBuffer({ size: NG * ROWS * lpitch * 2, usage: GPUBufferUsage.STORAGE });
    this.fc = dev.createBuffer({ size: Math.max(16, mono ? 16 : 2 * NG * rsc * cpitch * 2), usage: GPUBufferUsage.STORAGE });
    this.lt = dev.createBuffer({ size: Math.max(16, mono ? 16 : NG * ntc * S * rsc * 4), usage: GPUBufferUsage.STORAGE });
    this.dst = dev.createBuffer({ size: this.dstSamples * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.srcSize = 0;
    this.bind = null;
    this.lastSet = null;
    this.fieldBytes = this.fl.size + this.fc.size + this.lt.size + this.sb.size;
  }

  /** a GPU buffer (STORAGE usage) that already holds the input picture, e.g. filled by VideoFrame.copyTo() into
   *  its mapped range (sourceBuffer()); planes / bps / semi as for upload() */
  setSource(buffer, planes, bps, semi = false) {
    if (this.srcBuf !== buffer) { this.srcBuf = buffer; this.ownSrc = false; this.bind = null; }
    this.layoutPlanes(planes, bps, semi);
  }

  /** a decoded VideoFrame -> a GPU buffer holding its planes (for setSource()), through a pool of mappable staging
   *  buffers and a GPU copy: a new mappedAtCreation buffer per frame is 3x slower in Chrome (fresh shared memory).
   *  Give the buffer back with releaseFrame() when it is no longer drawn. */
  async frameBuffer(frame, layout) {
    const dev = this.device, size = align(layout.size, 4);
    if (this.poolSize !== size) {                  // a new frame size: start new pools
      for (const b of this.stageFree || []) b.destroy();
      for (const b of this.frameFree || []) b.destroy();
      this.stageFree = []; this.frameFree = []; this.poolSize = size;
    }
    // staging buffers are mapped again as soon as their copy is done, so one is usually ready (no mapAsync wait
    // while the decoder holds the frame)
    const st = this.stageFree.pop() ||
               dev.createBuffer({ size, usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation: true });
    try { await frame.copyTo(st.getMappedRange(), { layout: layout.planes }); } finally { st.unmap(); }
    const fb = this.frameFree.pop() || dev.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(st, 0, fb, 0, size);
    dev.queue.submit([enc.finish()]);
    const pools = this.stageFree;
    st.mapAsync(GPUMapMode.WRITE).then(() => {
      if (this.stageFree === pools && pools.length < 6) pools.push(st); else st.destroy();
    }, () => st.destroy());
    return fb;
  }

  /** a frameBuffer() that is no longer needed: reused once the GPU is done with it */
  releaseFrame(fb) {
    const pools = this.frameFree;
    this.device.queue.onSubmittedWorkDone().then(() => {
      if (this.frameFree === pools && fb.size === this.poolSize && pools.length < 8) pools.push(fb); else fb.destroy();
    });
  }

  layoutPlanes(planes, bps, semi) {
    const P = this.params;
    for (let i = 0; i < 3; i++) {
      const pl = planes[Math.min(i, planes.length - 1)];
      P[I.SOY + 2 * i] = pl.offset; P[I.SSY_ + 2 * i] = pl.stride;
    }
    P[I.BPS] = bps; P[I.STEPY] = bps; P[I.STEPC] = semi ? 2 * bps : bps;
    if (semi) { P[I.SOV] = P[I.SOU] + bps; P[I.SSV] = P[I.SSU]; }
  }

  /** the input picture: bytes (ArrayBuffer / view) with per-plane {offset, stride} in bytes; bps 1 or 2;
   *  semi = true: planes[1] is interleaved U V (NV12), planes[2] is ignored */
  upload(bytes, planes, bps, semi = false) {
    const P = this.params, dev = this.device;
    const u8 = bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const size = align(u8.byteLength, 4);
    if (!this.srcBuf || !this.ownSrc || this.srcSize < size) {
      if (this.srcBuf && this.ownSrc) this.srcBuf.destroy();
      this.ownSrc = true;
      this.srcBuf = dev.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      this.srcSize = size;
      this.bind = null;
    }
    if (u8.byteLength % 4) {                  // writeBuffer wants a multiple of 4 bytes
      const t = new Uint8Array(size); t.set(u8); dev.queue.writeBuffer(this.srcBuf, 0, t);
    } else dev.queue.writeBuffer(this.srcBuf, 0, u8);
    this.layoutPlanes(planes, bps, semi);
  }

  /** the grain of this picture: null (none) or loadParams(...).picture(n); mode 'full' or 'standard' */
  setGrain(pic, mode = 'full') {
    const P = this.params, f = this.fmt, bd = f.bitdepth;
    if (mode !== 'full' && mode !== 'standard') throw new Error('grain mode: full or standard');
    this.mode = pic ? mode : 'none';
    P[I.MODE] = mode === 'standard' ? 1 : 0;
    if (!pic) { P[I.HASY] = P[I.HASU] = P[I.HASV] = 0; return; }
    const mono = this.mono, csfl = pic.chroma_scaling_from_luma;
    P[I.HASY] = pic.num_y_points ? 1 : 0;
    P[I.HASU] = !mono && (csfl || pic.num_uv_points[0]) ? 1 : 0;
    P[I.HASV] = !mono && (csfl || pic.num_uv_points[1]) ? 1 : 0;
    P[I.LAG] = pic.ar_coeff_lag; P[I.ARSH] = pic.ar_coeff_shift;
    P[I.GMIN] = -(128 << (bd - 8)); P[I.GMAX] = (128 << (bd - 8)) - 1;
    const keys = fullKeys(pic.seed);
    P[I.KEYY] = keys[0] | 0; P[I.KEYU] = keys[1] | 0; P[I.KEYV] = keys[2] | 0;
    P[I.SCSH] = pic.scaling_shift; P[I.CSFL] = csfl;
    P[I.SEED] = pic.seed; P[I.OVL] = pic.overlap_flag;
    const b8 = bd - 8;
    if (pic.clip_to_restricted_range) {
      P[I.YMIN] = 16 << b8; P[I.YMAX] = 235 << b8; P[I.CMIN] = 16 << b8; P[I.CMAX] = (f.mtrx === 0 ? 235 : 240) << b8;
    } else { P[I.YMIN] = P[I.CMIN] = 0; P[I.YMAX] = P[I.CMAX] = (1 << bd) - 1; }
    for (let pl = 0; pl < 2; pl++) {
      P[I.ULM + 3 * pl] = pic.uv_luma_mult[pl]; P[I.UM + 3 * pl] = pic.uv_mult[pl];
      P[I.UOFF + 3 * pl] = pic.uv_offset[pl] * (1 << b8);
      const npos = 2 * pic.ar_coeff_lag * (pic.ar_coeff_lag + 1);
      P[I.LCU + pl] = pic.num_y_points ? pic.ar_coeffs_uv[pl][npos] : 0;
      grid(P, I.GRID + GSTRIDE * (1 + pl), pic.ar_coeffs_uv[pl], pic.ar_coeff_lag);
    }
    grid(P, I.GRID, pic.ar_coeffs_y, pic.ar_coeff_lag);
    P[I.SCPU] = csfl ? 0 : 1; P[I.SCPV] = csfl ? 0 : 2;
    // tables: once per parameter set
    if (this.lastSet !== pic.table || this.lastBd !== bd) {
      this.lastSet = pic.table; this.lastBd = bd;
      const t = new Int32Array(2048 + 3 * 4096);
      t.set(gaussianTable(bd, pic.grain_scale_shift), 0);
      if (pic.num_y_points || csfl) t.set(generateScaling(bd, pic.y_points), 2048);
      for (let pl = 0; pl < 2; pl++) if (pic.num_uv_points[pl]) t.set(generateScaling(bd, pic.uv_points[pl]), 2048 + 4096 * (1 + pl));
      this.device.queue.writeBuffer(this.tabs, 0, t);
    }
  }

  bindGroup() {
    if (!this.bind) {
      this.bind = this.device.createBindGroup({ layout: this.bgl, entries: [
        { binding: 0, resource: { buffer: this.uni } }, { binding: 1, resource: { buffer: this.tabs } },
        { binding: 2, resource: { buffer: this.fl } }, { binding: 3, resource: { buffer: this.fc } },
        { binding: 4, resource: { buffer: this.sb } }, { binding: 5, resource: { buffer: this.srcBuf } },
        { binding: 6, resource: { buffer: this.dst } }, { binding: 7, resource: { buffer: this.lt } },
      ] });
    }
    return this.bind;
  }

  /** record the grain passes into enc (after upload + setGrain) */
  encode(enc) {
    const P = this.params, dev = this.device;
    P[I.CPLANE] = P[I.HASU] ? 1 : 2;           // ar_chroma: first chroma plane with grain
    dev.queue.writeBuffer(this.uni, 0, P);
    const bg = this.bindGroup(), ts = this.timestamps && this.timing;
    const pass = (pipe, x, y, z, q) => {
      const d = { };
      if (ts && q !== undefined) d.timestampWrites = { querySet: this.qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 };
      const cp = enc.beginComputePass(d);
      cp.setPipeline(this.pipes[pipe]); cp.setBindGroup(0, bg); cp.dispatchWorkgroups(x, y, z); cp.end();
    };
    if (this.mode === 'full') {
      if (P[I.HASY]) pass('ar_luma', this.NG, 1, 1, 0);
      if (P[I.HASU] || P[I.HASV]) {
        if (P[I.LCU] || P[I.LCV]) pass('lumaterm', Math.ceil(P[I.CBW] / 64), P[I.RSC], this.NG);
        pass('ar_chroma', this.NG, P[I.HASU] && P[I.HASV] ? 2 : 1, 1, 2);   // both planes in one dispatch
      }
    }
    if (this.mode === 'standard') {
      if (P[I.HASY] || P[I.LCU] || P[I.LCV]) pass('std_tpl_luma', 1, 1, 1, 0);
      if (P[I.HASU] || P[I.HASV]) pass('std_tpl_chroma', P[I.HASU] && P[I.HASV] ? 2 : 1, 1, 1, 2);
      pass('std_offsets', Math.ceil(P[I.NBR] / 64), 1, 1);
    }
    pass('apply_y', Math.ceil(this.W / 32), Math.ceil(this.H / 8), 1, 4);
    if (!this.mono) pass('apply_c', Math.ceil(this.CW / 32), Math.ceil(this.CH / 8), 2, 6);
  }

  /** upload + grain + wait; -> GPU ms per pass when timestamps are available */
  async run() {
    const enc = this.device.createCommandEncoder();
    this.timing = true;
    this.encode(enc);
    if (this.timestamps) {
      enc.resolveQuerySet(this.qs, 0, 8, this.qbuf, 0);
      enc.copyBufferToBuffer(this.qbuf, 0, this.qread, 0, 64);
    }
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    if (!this.timestamps) return null;
    await this.qread.mapAsync(GPUMapMode.READ);
    const t = new BigUint64Array(this.qread.getMappedRange().slice(0));
    this.qread.unmap();
    const ms = i => (t[i + 1] > t[i] ? Number(t[i + 1] - t[i]) / 1e6 : 0);
    return { arLuma: ms(0), arChroma: ms(2), applyY: ms(4), applyC: ms(6) };
  }

  /** the grained picture as packed planes (Y, U, V; 8-bit bytes or 16-bit little-endian), like av1grain decode */
  async readPacked() {
    const dev = this.device, bytes = this.dstSamples * 2;
    if (!this.readBuf || this.readBuf.size < bytes) {
      if (this.readBuf) this.readBuf.destroy();
      this.readBuf = dev.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.dst, 0, this.readBuf, 0, bytes);
    dev.queue.submit([enc.finish()]);
    await this.readBuf.mapAsync(GPUMapMode.READ);
    const s16 = new Uint16Array(this.readBuf.getMappedRange(0, bytes).slice(0));
    this.readBuf.unmap();
    const P = this.params, hbd = this.fmt.bitdepth > 8;
    const planes = [[I.DOY, I.DSY, this.W, this.H]];
    if (!this.mono) planes.push([I.DOU, I.DSC, this.CW, this.CH], [I.DOV, I.DSC, this.CW, this.CH]);
    const total = planes.reduce((a, [, , w, h]) => a + w * h, 0);
    const out = hbd ? new Uint16Array(total) : new Uint8Array(total);
    let o = 0;
    for (const [off, st, w, h] of planes)
      for (let y = 0; y < h; y++) { out.set(s16.subarray(P[off] + y * P[st], P[off] + y * P[st] + w), o); o += w; }
    return new Uint8Array(out.buffer);
  }
}
