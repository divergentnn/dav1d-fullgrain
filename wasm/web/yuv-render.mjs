// av1-fullgrain: drawing the grained YUV pictures, with WebGPU (from the
// GpuGrain output buffer) or WebGL2 (from packed planes, the CPU grain path).
//
// Colour: matrix (BT.709 / 601 / 2020), limited or full range and the
// transfer come from the AV1 sequence header (PARAMS.json). SDR video is
// shown as its R'G'B' (as browsers do). PQ and HLG are converted to linear
// light; with WebGPU on a high dynamic range display they go to an
// extended-range canvas (1.0 = SDR white = 203 cd/m², BT.2408), otherwise
// they are tone mapped to SDR. Chroma is upsampled bilinearly.
//
// Copyright © 2026, divergentnn (av1-fullgrain). SPDX-License-Identifier: BSD-2-Clause

/** the colour constants of a format {bitdepth, mtrx, trc, pri, range} -> plain numbers for the shaders */
export function colour(f) {
  const s = 1 << (f.bitdepth - 8), max = (1 << f.bitdepth) - 1;
  const [kr, kb] = f.mtrx === 5 || f.mtrx === 6 ? [0.299, 0.114] : f.mtrx === 9 || f.mtrx === 10 ? [0.2627, 0.0593] : [0.2126, 0.0722];
  const kg = 1 - kr - kb;
  const full = !!f.range;
  return {
    yoff: full ? 0 : 16 * s, ysc: full ? 1 / max : 1 / (219 * s),
    coff: 128 * s, csc: full ? 1 / max : 1 / (224 * s),
    kr2: 2 * (1 - kr), kb2: 2 * (1 - kb), kgb: 2 * kb * (1 - kb) / kg, kgr: 2 * kr * (1 - kr) / kg,
    transfer: f.trc === 16 ? 1 : f.trc === 18 ? 2 : 0,              // 0 SDR, 1 PQ, 2 HLG
    bt2020: f.pri === 9,
  };
}

export const isHdr = f => f.trc === 16 || f.trc === 18;

// shared colour code (WGSL); GLSL below mirrors it
const WGSL_COLOUR = /* wgsl */`
fn pq_eotf(e: vec3<f32>) -> vec3<f32> {        // -> cd/m2
  let m1 = 2610.0 / 16384.0; let m2 = 2523.0 / 4096.0 * 128.0;
  let c1 = 3424.0 / 4096.0; let c2 = 2413.0 / 4096.0 * 32.0; let c3 = 2392.0 / 4096.0 * 32.0;
  let p = pow(clamp(e, vec3(0.0), vec3(1.0)), vec3(1.0 / m2));
  return 10000.0 * pow(max(p - c1, vec3(0.0)) / (c2 - c3 * p), vec3(1.0 / m1));
}
fn hlg_eotf(e: vec3<f32>) -> vec3<f32> {       // -> cd/m2, 1000 cd/m2 display, system gamma 1.2
  let a = 0.17883277; let b = 0.28466892; let c = 0.55991073;
  let x = clamp(e, vec3(0.0), vec3(1.0));
  let lin = select((exp((x - c) / a) + b) / 12.0, x * x / 3.0, x <= vec3(0.5));
  let ys = dot(lin, vec3(0.2627, 0.6780, 0.0593));
  return 1000.0 * pow(max(ys, 1e-6), 0.2) * lin;
}
fn srgb_oetf(v: vec3<f32>) -> vec3<f32> {      // extended: odd-symmetric, no upper clip
  let a = abs(v);
  return sign(v) * select(1.055 * pow(a, vec3(1.0 / 2.4)) - 0.055, 12.92 * a, a <= vec3(0.0031308));
}
fn to_output(rgb: vec3<f32>, transfer: i32, bt2020: bool, extended: bool) -> vec3<f32> {
  if (transfer == 0) { return clamp(rgb, vec3(0.0), vec3(1.0)); }
  var lin = select(hlg_eotf(rgb), pq_eotf(rgb), transfer == 1) / 203.0;
  if (bt2020) {
    lin = vec3(dot(lin, vec3(1.6605, -0.5876, -0.0728)), dot(lin, vec3(-0.1246, 1.1329, -0.0083)),
               dot(lin, vec3(-0.0182, -0.1006, 1.1187)));
  }
  if (extended) { return srgb_oetf(lin); }
  let m = max(max(lin.r, lin.g), max(lin.b, 1e-6));          // extended Reinhard on the largest channel
  let w = 1000.0 / 203.0;
  let mt = m * (1.0 + m / (w * w)) / (1.0 + m);
  return srgb_oetf(clamp(lin * (mt / m), vec3(0.0), vec3(1.0)));
}
`;

const RENDER = /* wgsl */`
struct RU {
  size: vec4<i32>,      // W, H, CW, CH
  off: vec4<i32>,       // Y, U, V offsets (samples), mono
  stride: vec4<i32>,    // Y stride, C stride, ssx, ssy
  cs: vec4<f32>,        // yoff, ysc, coff, csc
  k: vec4<f32>,         // kr2, kb2, kgb, kgr
  mode: vec4<i32>,      // transfer, bt2020, extended
}
@group(0) @binding(0) var<uniform> R: RU;
@group(0) @binding(1) var<storage, read> img: array<u32>;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn smp(off: i32, stride: i32, x: i32, y: i32) -> f32 {
  let i = off + y * stride + x;
  return f32((img[i >> 1u] >> (u32(i & 1) * 16u)) & 0xffffu);
}
fn csmp(off: i32, c: vec2<f32>) -> f32 {
  let sz = R.size.zw;
  let q = c - 0.5; let fl = floor(q); let fr = q - fl;
  let a = clamp(vec2<i32>(fl), vec2<i32>(0), sz - 1); let b = clamp(vec2<i32>(fl) + 1, vec2<i32>(0), sz - 1);
  let st = R.stride.y;
  return mix(mix(smp(off, st, a.x, a.y), smp(off, st, b.x, a.y), fr.x), mix(smp(off, st, a.x, b.y), smp(off, st, b.x, b.y), fr.x), fr.y);
}
${WGSL_COLOUR}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let p = vec2<i32>(pos.xy);
  let y = (smp(R.off.x, R.stride.x, p.x, p.y) - R.cs.x) * R.cs.y;
  var pb = 0.0; var pr = 0.0;
  if (R.off.w == 0) {
    let c = (vec2<f32>(p) + 0.5) / vec2<f32>(f32(1 << u32(R.stride.z)), f32(1 << u32(R.stride.w)));
    pb = (csmp(R.off.y, c) - R.cs.z) * R.cs.w; pr = (csmp(R.off.z, c) - R.cs.z) * R.cs.w;
  }
  let rgb = vec3<f32>(y + R.k.x * pr, y - R.k.z * pb - R.k.w * pr, y + R.k.y * pb);
  return vec4<f32>(to_output(rgb, R.mode.x, R.mode.y != 0, R.mode.z != 0), 1.0);
}
`;

/**
 * Draws a GpuGrain's output buffer into a canvas (WebGPU). hdr: try an extended-range canvas for PQ / HLG
 * when the display has a high dynamic range.
 */
export class GpuRenderer {
  constructor(gg, canvas, opts = {}) {
    this.gg = gg;
    this.canvas = canvas;
    this.ctx = canvas.getContext('webgpu');
    if (!this.ctx) throw new Error('no WebGPU canvas context');
    const dev = gg.device;
    const module = dev.createShaderModule({ code: RENDER });
    this.bgl = dev.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'read-only-storage' } },
    ] });
    this.module = module;
    this.uni = dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.wantHdr = opts.hdr !== false;
    this.pipes = new Map();
    this.extended = false;
  }

  pipeline(format) {
    if (!this.pipes.has(format)) {
      const dev = this.gg.device;
      this.pipes.set(format, dev.createRenderPipeline({
        layout: dev.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
        vertex: { module: this.module, entryPoint: 'vs' },
        fragment: { module: this.module, entryPoint: 'fs', targets: [{ format }] },
        primitive: { topology: 'triangle-list' },
      }));
    }
    return this.pipes.get(format);
  }

  /** (re)configure for a format {width, height, bitdepth, layout, mtrx, trc, pri, range} */
  configure(f) {
    const key = JSON.stringify(f);
    if (key === this.key) return;
    this.key = key;
    const dev = this.gg.device;
    this.canvas.width = f.width; this.canvas.height = f.height;
    const hdrDisplay = typeof matchMedia === 'function' && matchMedia('(dynamic-range: high)').matches;
    this.extended = false;
    if (isHdr(f) && this.wantHdr && hdrDisplay) {
      try {
        this.ctx.configure({ device: dev, format: 'rgba16float', colorSpace: 'srgb', toneMapping: { mode: 'extended' }, alphaMode: 'opaque' });
        const c = this.ctx.getConfiguration ? this.ctx.getConfiguration() : null;
        this.extended = !c || !c.toneMapping || c.toneMapping.mode === 'extended';
        this.format = 'rgba16float';
      } catch (e) { this.extended = false; }
    }
    if (!this.extended) {
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.ctx.configure({ device: dev, format: this.format, colorSpace: 'srgb', alphaMode: 'opaque' });
    }
    const c = colour(f), P = this.gg.params, I = this.gg.I;
    const u = new ArrayBuffer(96), i32 = new Int32Array(u), f32 = new Float32Array(u);
    const ssx = f.layout === 1 || f.layout === 2 ? 1 : 0, ssy = f.layout === 1 ? 1 : 0;
    i32.set([f.width, f.height, this.gg.CW, this.gg.CH, P[I.DOY], P[I.DOU], P[I.DOV], f.layout === 0 ? 1 : 0,
             P[I.DSY], P[I.DSC], ssx, ssy], 0);
    f32.set([c.yoff, c.ysc, c.coff, c.csc, c.kr2, c.kb2, c.kgb, c.kgr], 12);
    i32.set([c.transfer, c.bt2020 ? 1 : 0, this.extended ? 1 : 0, 0], 20);
    dev.queue.writeBuffer(this.uni, 0, u);
    this.bind = dev.createBindGroup({ layout: this.bgl, entries: [
      { binding: 0, resource: { buffer: this.uni } }, { binding: 1, resource: { buffer: this.gg.dst } }] });
    this.dstBuf = this.gg.dst;
  }

  /** record the draw (after the grain passes) */
  encode(enc) {
    if (this.dstBuf !== this.gg.dst) { this.key = null; throw new Error('renderer: configure() after GpuGrain.configure()'); }
    const pass = enc.beginRenderPass({ colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(),
      loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }] });
    pass.setPipeline(this.pipeline(this.format));
    pass.setBindGroup(0, this.bind);
    pass.draw(3);
    pass.end();
  }
}

// ---- WebGL2 (CPU grain path): packed planes -> canvas, SDR output

const GL_VS = `#version 300 es
void main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`;
const GL_FS = `#version 300 es
precision highp float; precision highp int; precision highp usampler2D;
uniform usampler2D ty, tu, tv;
uniform vec2 ss;
uniform vec4 cs, k;
uniform ivec4 mode;           // transfer, bt2020, mono
out vec4 o;
float cfetch(usampler2D t, vec2 p) {
  ivec2 sz = textureSize(t, 0);
  vec2 q = p - 0.5, fl = floor(q), fr = q - fl;
  ivec2 a = clamp(ivec2(fl), ivec2(0), sz - 1), b = clamp(ivec2(fl) + 1, ivec2(0), sz - 1);
  float v00 = float(texelFetch(t, a, 0).r), v10 = float(texelFetch(t, ivec2(b.x, a.y), 0).r);
  float v01 = float(texelFetch(t, ivec2(a.x, b.y), 0).r), v11 = float(texelFetch(t, b, 0).r);
  return mix(mix(v00, v10, fr.x), mix(v01, v11, fr.x), fr.y);
}
vec3 pq_eotf(vec3 e) {
  float m1 = 2610.0 / 16384.0, m2 = 2523.0 / 4096.0 * 128.0, c1 = 3424.0 / 4096.0, c2 = 2413.0 / 4096.0 * 32.0, c3 = 2392.0 / 4096.0 * 32.0;
  vec3 p = pow(clamp(e, 0.0, 1.0), vec3(1.0 / m2));
  return 10000.0 * pow(max(p - c1, 0.0) / (c2 - c3 * p), vec3(1.0 / m1));
}
vec3 hlg_eotf(vec3 e) {
  float a = 0.17883277, b = 0.28466892, c = 0.55991073;
  vec3 x = clamp(e, 0.0, 1.0);
  vec3 lin = mix((exp((x - c) / a) + b) / 12.0, x * x / 3.0, vec3(lessThanEqual(x, vec3(0.5))));
  float ys = dot(lin, vec3(0.2627, 0.6780, 0.0593));
  return 1000.0 * pow(max(ys, 1e-6), 0.2) * lin;
}
vec3 srgb_oetf(vec3 v) { vec3 a = abs(v); return sign(v) * mix(1.055 * pow(a, vec3(1.0 / 2.4)) - 0.055, 12.92 * a, vec3(lessThanEqual(a, vec3(0.0031308)))); }
void main() {
  ivec2 sz = textureSize(ty, 0);
  ivec2 p = ivec2(gl_FragCoord.xy); p.y = sz.y - 1 - p.y;
  float y = (float(texelFetch(ty, p, 0).r) - cs.x) * cs.y, pb = 0.0, pr = 0.0;
  if (mode.z == 0) {
    vec2 c = (vec2(p) + 0.5) / ss;
    pb = (cfetch(tu, c) - cs.z) * cs.w; pr = (cfetch(tv, c) - cs.z) * cs.w;
  }
  vec3 rgb = vec3(y + k.x * pr, y - k.z * pb - k.w * pr, y + k.y * pb);
  if (mode.x == 0) { o = vec4(clamp(rgb, 0.0, 1.0), 1.0); return; }
  vec3 lin = (mode.x == 1 ? pq_eotf(rgb) : hlg_eotf(rgb)) / 203.0;
  if (mode.y != 0) lin = vec3(dot(lin, vec3(1.6605, -0.5876, -0.0728)), dot(lin, vec3(-0.1246, 1.1329, -0.0083)), dot(lin, vec3(-0.0182, -0.1006, 1.1187)));
  float m = max(max(lin.r, lin.g), max(lin.b, 1e-6)), w = 1000.0 / 203.0;
  float mt = m * (1.0 + m / (w * w)) / (1.0 + m);
  o = vec4(srgb_oetf(clamp(lin * (mt / m), 0.0, 1.0)), 1.0);
}`;

/** WebGL2 renderer: draw(packed Uint8Array, format) */
export function glRenderer(canvas) {
  const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: false,
                                           premultipliedAlpha: false, preserveDrawingBuffer: true });
  if (!gl) throw new Error('no WebGL2');
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, GL_VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, GL_FS));
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
  return (bytes, f) => {
    const W = f.width, H = f.height;
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    gl.viewport(0, 0, W, H);
    const hbd = f.bitdepth > 8, fmt = hbd ? gl.R16UI : gl.R8UI, typ = hbd ? gl.UNSIGNED_SHORT : gl.UNSIGNED_BYTE;
    const mono = f.layout === 0, ssx = f.layout === 1 || f.layout === 2 ? 2 : 1, ssy = f.layout === 1 ? 2 : 1;
    const cw = Math.ceil(W / ssx), ch = Math.ceil(H / ssy);
    const T = hbd ? Uint16Array : Uint8Array, bps = hbd ? 2 : 1;
    const view = (o, n) => new T(bytes.buffer, bytes.byteOffset + o * bps, n);
    const up = (i, d, w, h) => { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, tex[i]);
      gl.texImage2D(gl.TEXTURE_2D, 0, fmt, w, h, 0, gl.RED_INTEGER, typ, d); };
    up(0, view(0, W * H), W, H);
    if (!mono) { up(1, view(W * H, cw * ch), cw, ch); up(2, view(W * H + cw * ch, cw * ch), cw, ch); }
    const c = colour(f);
    gl.uniform4f(U('cs'), c.yoff, c.ysc, c.coff, c.csc);
    gl.uniform4f(U('k'), c.kr2, c.kb2, c.kgb, c.kgr);
    gl.uniform4i(U('mode'), c.transfer, c.bt2020 ? 1 : 0, mono ? 1 : 0, 0);
    gl.uniform2f(U('ss'), ssx, ssy);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
}
