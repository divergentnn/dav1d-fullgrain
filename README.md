# dav1d-fullgrain

[dav1d](https://code.videolan.org/videolan/dav1d) 1.5.4 with full-frame AV1
film grain: the grain an AV1 file signals is drawn over the whole frame,
instead of tiling one small grain patch across it.

This repository is the decoder part of
**[av1-fullgrain](https://github.com/divergentnn/av1-fullgrain)**. Start
there for the comparisons and numbers, the install guides (mpv, ffmpeg,
Arch, macOS, Windows), the browser player and the grain seed fixer.

The fork keeps upstream's layout and stays small, so it can follow upstream
releases. Upstream's README is [README.dav1d.md](README.dav1d.md).

> **Not a conformant AV1 decoder for video with film grain.** With the
> default mode (`full`), decoded AV1 with film grain differs from every other
> decoder, although the grain has the same statistics. Video without film
> grain decodes exactly as with upstream dav1d. `DAV1D_GRAIN_MODE=standard`
> gives standard AV1 grain at any time.

## Build

```sh
meson setup build --buildtype=release
ninja -C build
build/tools/dav1d --version
```

The library has the same name, version and API as dav1d 1.5.4
(`libdav1d.so.7`, `libdav1d.7.dylib`), so it can replace the system copy.
[av1-fullgrain's install guides](https://github.com/divergentnn/av1-fullgrain#install)
explain how, per system and player.

## Options

| | |
|---|---|
| `DAV1D_GRAIN_MODE=full\|standard\|dual\|multi\|standard_c` | override the default mode at runtime; `standard` = conformant AV1 grain |
| `DAV1D_GRAIN_MODE_FILE=path` | re-read the mode from this file while decoding (every 100 ms) |
| `DAV1D_GRAIN_STATS=1` | print grain CPU time per frame when the decoder closes |
| `DAV1D_GRAIN_SIMD=0` | use the C paths (testing) |
| `DAV1D_GRAIN_TEMPLATES`, `DAV1D_GRAIN_WARMUP`, `DAV1D_GRAIN_BANDS` | tuning for multi/dual (16) and full (16 rows, 4 bands) |
| meson `-Dgrain_mode_default=full\|standard\|dual\|multi` | built-in default (`full`) |
| `DAV1D_GRAIN_FORCE=0\|1`, meson `-Dgrain_force=true\|false` | apply the variant even when the player turned grain off, hiding the grain parameters from it (default on; off = upstream behaviour) |

## What the fork changes

- `src/fg_variant.{c,h}`, `src/fg_variant_tmpl.c`, `src/fg_ar_skew.h`: the
  grain modes (`full`, `dual`, `multi`), in C with AVX2 (x86-64) and NEON
  (arm64, also WebAssembly SIMD128) kernels.
- `src/fg_apply*`, `src/filmgrain*`, `src/lib.c`, `src/thread_task.c`,
  `src/meson.build`: route grain through the selected mode; `grain_force`.
- `src/internal.h`, `src/obu.c`: `fg_hook`, a frame header parser hook for
  av1-fullgrain's `av1grain` tool. It is not reachable through the public
  API.
- `meson.build`, `meson_options.txt`: the `grain_mode_default` and
  `grain_force` options.
- `tests/fullgrain/`: synthetic test streams (aomenc's film grain test
  vectors), `run.sh` and `timing.sh`.
- `.github/workflows/fullgrain.yml`: CI on Linux x86-64 and macOS arm64.

## Tests

`sh tests/fullgrain/run.sh build/tools/dav1d [upstream/build/tools/dav1d]`
checks:
- every mode against committed hashes, which are the same on x86-64 and
  arm64;
- the SIMD paths against the C paths;
- given an upstream dav1d 1.5.4 build, that `standard` is md5-identical to
  it.

`sh tests/fullgrain/timing.sh build/tools/dav1d` prints the grain CPU time
per frame for each mode.

## Licence

dav1d is © VideoLAN and dav1d authors, under the BSD 2-Clause licence
([COPYING](COPYING)). This fork's changes are released under the same
licence.
