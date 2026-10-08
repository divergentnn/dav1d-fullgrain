# dav1d-fullgrain in the browser (WebAssembly)

This builds the fork, full-frame film grain included, as a WebAssembly module
with SIMD and threads. It comes with a small player kit: a decode Worker, and
a `<canvas>` that behaves like a `<video>` element and draws with WebGL2.

flipcomp, my comparison viewer, uses the kit to show AV1 clips that carry
film grain with either standard or full-frame grain.

## Build

```sh
wasm/build.sh                 # -> wasm/dist/web/ (decoder, worker, player, demo page)
NODE_CLI=1 wasm/build.sh      # also the dav1d CLI for Node.js, for the tests below
EMSDK=~/emsdk wasm/build.sh   # use an existing, activated emsdk
```

- `build.sh` installs emsdk 6.0.12 into `wasm/emsdk` when `EMSDK` is not set.
  It needs meson, ninja and git.
- It builds `libdav1d.a` with `-msimd128 -pthread` (dav1d has no WebAssembly
  asm, so this is the C code with clang's auto-vectorisation), then links
  `djs.c`, the small C API, into `dav1d-fg.mjs` + `dav1d-fg.wasm` (about
  800 KB).
- The grain variants use the fork's NEON kernels. They compile to WebAssembly
  SIMD through the `arm_neon.h` that emscripten ships, which is generated
  from SIMDe.

Test the build. The node CLI must give the same hashes as native x86-64 and
arm64:

```sh
sh tests/fullgrain/run.sh wasm/dist/node/dav1d-node   # SIMD == C == the committed hashes
python3 wasm/web/serve.py                              # then http://127.0.0.1:8077/demo.html
```

`serve.py` sends the COOP/COEP headers the threads need.

## Use

```js
import { wasmSupport, createWasmClip } from './wasm-clip.mjs';   // from wasm/dist/web
if (!wasmSupport().length) {                                      // [] = supported, else the reasons
  const clip = await createWasmClip(await (await fetch('clip.mp4')).arrayBuffer(),
                                    {fps: 24000 / 1001, mode: 'full', requireGrain: true});
  if (clip) { document.body.append(clip); clip.play(); }         // null: not AV1 / no film grain
}
```

**Inputs.** An AV1 MP4 held in memory; mp4box.js demuxes it.

**The canvas** offers:
- `currentTime` with `seeking`/`seeked` events, `duration`, `videoWidth`,
  `videoHeight`, `readyState`, `paused`;
- `play()`, `pause()`, and `requestVideoFrameCallback()` (whose
  `mediaTime` is the frame on screen);
- `setGrainMode('full'|'standard'|'dual'|'multi')`, `getFrame(n)` (the
  decoded planes), `stats` and `dispose()`.

**Seeking** restarts the decoder at the previous keyframe. Frames just before
the target are cached, so stepping backward stays fast.

**Colour.** BT.709/601/2020 matrices and limited or full range come from the
sequence header. Chroma is upsampled bilinearly. HDR transfers are left to
`<video>`.

**The page must be cross-origin isolated** (`Cross-Origin-Opener-Policy:
same-origin`, `Cross-Origin-Embedder-Policy: require-corp`), so that
SharedArrayBuffer and threads work. Its CSP must allow
`script-src 'self' 'wasm-unsafe-eval'` and `worker-src 'self'`.

## Speed and exactness

Measured on a Ryzen 5 9600X; the browser is headless Chromium 151 with 6
decoder threads:

| clip | Chromium, decode | Node 22 CLI, 4 threads |
|---|---:|---:|
| 1080p 10-bit AV1, 8 Mb/s, full-frame grain | 90-92 fps | 78 fps (27 fps on 1 thread) |
| same clip, standard grain | 104 fps | 91 fps |
| 3584x2160 10-bit AV1, full-frame grain | 30 fps | 23 fps |
| 1080p near-lossless AV1, 162 Mb/s | 21 fps | 21 fps |

**Real-time playback.** In the flipcomp viewer, a 1080p grain clip played
120 frames in 5 s with none dropped (headless, software WebGL). At 4K,
decoding keeps up only on fast machines.

**Grain step**, 1080p, one thread: full 12 ms per frame (NEON kernels on
SIMD128), standard 6.8 ms (dav1d's C code, since there is no WebAssembly asm).

**Exactness.** Output is bit-identical to native dav1d:
- in Chromium, all 240 frames of a 1080p clip plus random-access seeks, in
  both modes, and the first frames of a 4K clip;
- under Node, `tests/fullgrain/run.sh`.

## Credits

- **dav1d:** © VideoLAN and the dav1d authors, BSD-2-Clause.
- **The C API in `djs.c`** follows the shape of
  [dav1d.js](https://github.com/Kagami/dav1d.js) by Kagami Hiiragi (CC0),
  extended for 10-bit, threads and grain modes.
- **[discere-os/dav1d.wasm](https://github.com/discere-os/dav1d.wasm)** was
  checked too. Its `wasm` branch is upstream dav1d 1.5.2-dev with a new
  README, and its TypeScript wrapper is not published. The emscripten cross
  files it relies on are upstream dav1d's `package/crossfiles/wasm32.meson`,
  which this build also follows.
- **[mp4box.js](https://github.com/gpac/mp4box.js)** 2.4.1 (BSD-3-Clause) is
  vendored in `web/vendor/mp4box/`.
- **emscripten's `arm_neon.h`** is generated from
  [SIMDe](https://github.com/simd-everywhere/simde) (MIT).
