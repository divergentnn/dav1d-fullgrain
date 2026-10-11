#!/usr/bin/env python3
"""Test data for the browser grain checks (wasm/test/gpu-test.mjs).

    gpu-vectors.py AV1GRAIN OUTDIR INPUT.ivf|INPUT.mp4 ... [--frames N] [--threads N]

For every input: the grain-free stream and its PARAMS.json (av1grain strip), the grain-free pictures as raw packed
frames (<name>.yuv, av1grain decode --grain off), and the SHA-256 of every picture of the ORIGINAL decoded by dav1d
with full-frame and with standard grain. OUTDIR/vectors.json lists them. The browser check applies the grain from
PARAMS.json on the GPU to the grain-free pictures (or to its own decoder's output) and compares the hashes.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("av1grain_mp4", HERE.parent.parent / "tools" / "av1grain_mp4.py")
mp4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mp4)


def frame_bytes(p):
    w, h, layout, bps = p["width"], p["height"], p["layout"], 2 if p["bitdepth"] > 8 else 1
    if layout == 0:
        return w * h * bps
    ssx, ssy = int(layout in (1, 2)), int(layout == 1)
    return (w * h + 2 * ((w + ssx) >> ssx) * ((h + ssy) >> ssy)) * bps


def hashes(cmd, size, n):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = []
    while len(out) < n:
        b = p.stdout.read(size)
        if len(b) < size:
            break
        out.append(hashlib.sha256(b).hexdigest())
    p.stdout.close()
    p.wait()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("av1grain")
    ap.add_argument("outdir")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--frames", type=int, default=0, help="only the first N pictures (default: all)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    vecs = []
    for src in map(Path, a.inputs):
        name = src.stem
        ivf, cfg = out / f"{name}.ivf", None
        if src.suffix.lower() == ".mp4":
            buf = src.read_bytes()
            top, tracks = mp4.parse(buf)
            vt = mp4.video_track(tracks)
            mp4.write_ivf(ivf, [bytes(buf[o:o + n]) for o, n in vt.samples(buf)])
            c = vt.av1c_config(buf)
            if c:
                cfg = out / f"{name}.cfg"
                cfg.write_bytes(c)
        else:
            ivf.write_bytes(src.read_bytes())
        cf = ["--config", str(cfg)] if cfg else []
        ng, pj = out / f"{name}.nograin.ivf", out / f"{name}.json"
        subprocess.run([a.av1grain, "strip", str(ivf), str(ng), str(pj), "--threads", str(a.threads)] + cf, check=True)
        if src.suffix.lower() == ".mp4":                  # the grain-free MP4, for the browser's own decoder
            (out / f"{name}.nograin.mp4").write_bytes(mp4.rebuild(buf, top, tracks, vt, mp4.read_ivf(ng)))
            (out / f"{name}.mp4").write_bytes(buf)
        p = json.loads(pj.read_text())
        n = min(p["frames"], a.frames) if a.frames else p["frames"]
        fb = frame_bytes(p)
        lim = ["--frames", str(n), "--threads", str(a.threads)] + cf
        subprocess.run([a.av1grain, "decode", str(ng), str(out / f"{name}.yuv"), "--grain", "off"] + lim, check=True,
                       stderr=subprocess.DEVNULL)
        v = {"name": name, "width": p["width"], "height": p["height"], "bitdepth": p["bitdepth"], "layout": p["layout"],
             "frames": n, "frameBytes": fb, "params": pj.name, "yuv": f"{name}.yuv", "nograin": ng.name,
             "cfg": cfg.name if cfg else None,
             "mp4": f"{name}.mp4" if src.suffix.lower() == ".mp4" else None,
             "nograinMp4": f"{name}.nograin.mp4" if src.suffix.lower() == ".mp4" else None}
        for mode in ("full", "standard", "off"):
            v[mode] = hashes([a.av1grain, "decode", str(ivf), "-", "--grain", mode] + lim, fb, n)
            if len(v[mode]) != n:
                sys.exit(f"{name}: {mode}: {len(v[mode])} of {n} pictures")
        print(f"{name}: {p['width']}x{p['height']} {p['bitdepth']}-bit layout {p['layout']}, {n} pictures, "
              f"{sum(g[0] >= 0 for g in p['grain'][:n])} with grain", file=sys.stderr)
        vecs.append(v)
    (out / "vectors.json").write_text(json.dumps(vecs, indent=1))


if __name__ == "__main__":
    main()
