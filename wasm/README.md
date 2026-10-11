# av1-fullgrain in the browser

The browser's own AV1 decoder decodes a **grain-free copy** of a clip, and the grain is drawn on top from the grain
parameters of every frame:
- by WebGPU compute shaders;
- or, where WebGPU is missing, by this fork's grain code compiled to WebAssembly.

Full-frame grain and standard AV1 grain are both bit-identical to the C code (dav1d-fullgrain, and dav1d's conformant
grain).

[flipcomp](https://github.com/divergentnn/flipcomp) (private), my comparison viewer, uses this kit for AV1 clips with
film grain.

## How it fits together

1. **At upload time** (any machine with this repository built), split the clip in two:

   ```sh
   build/tools/av1grain strip IN.ivf OUT.ivf PARAMS.json
   tools/av1grain_mp4.py IN.mp4 OUT.mp4 PARAMS.json --verify      # the same for MP4
   ```

   - `OUT` is the clip with every frame header's grain syntax replaced by `apply_grain = 0`. Tile data and everything
     else are untouched, so it decodes to exactly the original's pictures with grain off.
   - `PARAMS.json` lists the grain dav1d applied to each output picture (sets of parameters + a seed per picture).
   - Positions come from dav1d's own frame header parser, through a hook in `src/obu.c`.
   - WebM / MKV / MOV: remux to MP4 first, e.g. with `ffmpeg -c copy`.
2. **In the browser**, `grain-clip.mjs`:
   - demuxes the MP4 (`mp4-demux.mjs`);
   - decodes with WebCodecs `VideoDecoder`, using hardware where the device has it;
   - copies each frame's planes into a GPU buffer (`VideoFrame.copyTo`);
   - applies the grain from `PARAMS.json`, full-frame or standard;
   - converts YUV to RGB itself (`yuv-render.mjs`).

| tier | when | grain | drawing |
|---|---|---|---|
| 1 | WebGPU | `gpu-grain.mjs` (WGSL compute) | WebGPU canvas; HDR canvas for PQ / HLG on HDR displays |
| 2 | no WebGPU | `fg-grain.wasm` in `grain-worker.mjs` (this fork's C grain code, SIMD + threads; loaded only then) | WebGL2 |
| none | no AV1 decoder (`VideoDecoder.isConfigSupported` says no) | | `createGrainClip` throws; show a message |

**Full-frame grain on the GPU.**
- One workgroup per band group (16 warm-up rows + 4 x 32 rows, exactly where the C code restarts the recursion) and
  plane, one thread per row.
- The AR filter runs as a wavefront: row r filters columns `[4(t-2r), 4(t-2r)+4)` in step t.
- The rows above are read from a ring of the last 8 steps in workgroup memory, into a sliding 3 x 7 register window.
- Only integer arithmetic is used, so results are exact on every GPU.

**Standard grain on the GPU.**
- The 82x73 templates are made with the LFSR noise and the AR filter (as a wavefront in workgroup memory).
- The 32x32 blocks take per-row LFSR offsets, with dav1d's overlap blending.

## Use

```js
import { av1Support, createGrainClip } from './grain-clip.mjs';
const s = await av1Support();                      // {decode, engine: 'webgpu' | 'wasm' | null, why}
if (s.decode && s.engine) {
  const clip = await createGrainClip(await (await fetch('clip.nograin.mp4')).arrayBuffer(),
                                     await (await fetch('clip.json')).text(), {fps: 24000 / 1001, mode: 'full'});
  document.body.append(clip); clip.play();
}
```

The canvas behaves like a muted `<video>`:
- `currentTime` with `seeking`/`seeked`, `duration`, `videoWidth`/`videoHeight`, `readyState`, `paused`;
- `play()`, `pause()`, `requestVideoFrameCallback()` (whose `mediaTime` is the frame on screen);
- also `setGrainMode('full'|'standard')`, `getFrame(n)` (the grained planes, for checks), `engine`, `stats`,
  `hdrOutput`, `dispose()`.

Decoded frames are cached on the GPU for stepping back. Seeking restarts the decoder at the previous keyframe.

The WebAssembly grain (tier 2) needs a cross-origin isolated page (COOP/COEP) for its threads. WebGPU and WebCodecs
need a secure context (https or localhost).

## Build

```sh
wasm/build.sh                 # -> wasm/dist/web/: the kit + fg-grain.mjs / fg-grain.wasm (grain only, no decoder)
DECODER=1 wasm/build.sh       # also the whole decoder as WebAssembly (dav1d-fg.wasm, the earlier player; unused now)
NODE_CLI=1 wasm/build.sh      # also the dav1d CLI for Node.js (sh tests/fullgrain/run.sh wasm/dist/node/dav1d-node)
```

`fg-grain.wasm`:
- is linked from the same `libdav1d.a` as the decoder, with `wasm/fgjs.c`; the linker leaves the decoder out;
- 211 KB (86 KB gzipped) + 76 KB of JS glue.

The build installs emsdk 6.0.12 into `wasm/emsdk` unless `EMSDK` points to an activated one. It needs meson and ninja.

## Tests

```sh
sh tests/fullgrain/strip.sh build/tools/av1grain       # strip tool: grain off == original; params + dav1d == original
python3 wasm/test/gpu-vectors.py build/tools/av1grain /tmp/vec tests/fullgrain/*.ivf clip.mp4   # test data + hashes
PLAYWRIGHT=/path/to/playwright-core/index.mjs node wasm/test/gpu-test.mjs /tmp/vec --mode full,standard
    # [--engine wasm] [--pp low-power|swiftshader] [--page webcodecs-test.html | player-test.html]
node wasm/test/selftest-serve.mjs /tmp/vec 8140        # selftest.html for another machine (results posted back)
```

- **`gpu-test.html`**: grain on the grain-free pictures (WebGPU or `--engine wasm`) vs dav1d, picture by picture
  (SHA-256).
- **`webcodecs-test.html`**: the browser's decoder on the grain-free MP4 vs dav1d with grain off, then GPU grain on its
  output.
- **`player-test.html`**: `grain-clip.mjs` end to end, frames reached by seeking in a scrambled order, plus playback.
- **`selftest.html`**: everything above in one report (decoder, exactness, grain / draw speed, playback).
- **`upload-bench.html`**: ways of getting decoded frames onto the GPU.

## Speed and exactness

**Exact** (every picture, SHA-256 against the C code), full-frame and standard, both engines:
- on all test streams: 8/10-bit; 4:2:0 / 4:2:2 / 4:4:4; lag 2 and 3; odd sizes; chroma from luma; restricted range;
- on a 1080p clip and on 4K PQ clips;
- on NVIDIA (Vulkan), AMD RDNA2 (Vulkan), SwiftShader, and Apple GPUs (Chrome on macOS, Metal).

The browser's decoder output for the grain-free copy is identical to dav1d with grain off: checked in Chromium on
Linux and Chrome on macOS.

**Grain, per picture** (full / standard):

| | 1080p | 4K |
|---|---:|---:|
| WebGPU, RTX 3090 | 1.7 / 0.65 ms | 3.2-3.5 / 0.9 ms |
| WebGPU, Apple (Chrome, macOS) | 3.1 / 1.2 ms | 7.2 / 2.0 ms |
| WebGPU, AMD 2-CU iGPU | 5.9 / 2.8 ms | 22-29 / 6.6-8.6 ms |
| WebAssembly, Mac (Chrome) | 4.3 / 2.6 ms | 16.3 / 10.3 ms |

**Decoding** with Chrome's WebCodecs (software dav1d with its asm): about 145 fps at 1080p and 120 fps at 4K, on the
Mac and on a Ryzen 5 9600X. The earlier all-WebAssembly decoder managed 90 / 30 fps.

## Credits

- **dav1d:** © VideoLAN and the dav1d authors, BSD-2-Clause.
- **`djs.c`** (the optional WASM decoder) follows the shape of [dav1d.js](https://github.com/Kagami/dav1d.js) by
  Kagami Hiiragi (CC0).
- **emscripten's `arm_neon.h`** is generated from [SIMDe](https://github.com/simd-everywhere/simde) (MIT).
