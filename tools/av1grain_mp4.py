#!/usr/bin/env python3
"""av1grain for MP4: a grain-free copy of an AV1 MP4 and the grain parameters it carried.

    av1grain_mp4.py IN.mp4 OUT.mp4 PARAMS.json [--verify] [--av1grain PATH] [--threads N]

OUT.mp4 is IN.mp4 with every AV1 frame header's film grain syntax replaced by apply_grain = 0 (tools/av1grain.c
strip, on dav1d's own parser). Every other box, track and sample is copied byte for byte; only the AV1 samples'
sizes and the chunk offsets change. PARAMS.json lists the grain dav1d applied to each picture, in sample order
(see tools/av1grain.c), plus the SHA-256 of both files.

--verify decodes both files with dav1d and checks that the grain-free copy with PARAMS.json applied is identical
to the original in full-frame and standard grain, and with grain off (6 decodes).

Exit status: 0 done; 3 the clip has no film grain (nothing written); 4 not an AV1 MP4 this tool can rewrite
(fragmented, no AV1 track, ...); 1 any other failure. Needs the av1grain binary from this repository's build:
--av1grain, $AV1GRAIN or av1grain on $PATH. Python standard library only.

Copyright (c) 2026, divergentnn (av1-fullgrain). SPDX-License-Identifier: BSD-2-Clause
"""
import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

CONTAINERS = {b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts", b"dinf"}


class Unsupported(Exception):
    pass


def boxes(buf, off, end):
    """-> [(type, start, size, header size)] of the boxes in buf[off:end]"""
    out = []
    while off + 8 <= end:
        size, typ = struct.unpack_from(">I4s", buf, off)
        hdr = 8
        if size == 1:
            size, hdr = struct.unpack_from(">Q", buf, off + 8)[0], 16
        elif size == 0:
            size = end - off
        if size < hdr or off + size > end:
            raise Unsupported(f"broken box {typ!r} at {off}")
        out.append((typ, off, size, hdr))
        off += size
    return out


def child(buf, box, typ):
    _, off, size, hdr = box
    return next((b for b in boxes(buf, off + hdr, off + size) if b[0] == typ), None)


def path(buf, box, *types):
    for t in types:
        box = child(buf, box, t) if box else None
    return box


class Track:
    def __init__(self, buf, trak):
        self.trak = trak
        hdlr = path(buf, trak, b"mdia", b"hdlr")
        self.handler = bytes(buf[hdlr[1] + hdlr[3] + 8:hdlr[1] + hdlr[3] + 12]) if hdlr else b""
        stbl = path(buf, trak, b"mdia", b"minf", b"stbl")
        if not stbl:
            raise Unsupported("a track without a sample table (fragmented MP4?)")
        stsd = child(buf, stbl, b"stsd")
        self.entry = boxes(buf, stsd[1] + stsd[3] + 8, stsd[1] + stsd[2])[0] if stsd else None
        self.codec = self.entry[0] if self.entry else b""
        stsz = child(buf, stbl, b"stsz")
        if not stsz:
            raise Unsupported("no stsz (compact sample sizes are not supported)")
        o = stsz[1] + stsz[3]
        const, n = struct.unpack_from(">II", buf, o + 4)
        self.stsz_off = o + 12
        self.sizes = [const] * n if const else list(struct.unpack_from(f">{n}I", buf, self.stsz_off))
        self.const_size = const
        co = child(buf, stbl, b"stco") or child(buf, stbl, b"co64")
        if not co:
            raise Unsupported("no chunk offsets")
        o = co[1] + co[3]
        nc = struct.unpack_from(">I", buf, o + 4)[0]
        self.co64 = co[0] == b"co64"
        self.co_off = o + 8
        self.chunks = list(struct.unpack_from(f">{nc}{'Q' if self.co64 else 'I'}", buf, self.co_off))
        stsc = child(buf, stbl, b"stsc")
        o = stsc[1] + stsc[3]
        ne = struct.unpack_from(">I", buf, o + 4)[0]
        ent = [struct.unpack_from(">III", buf, o + 8 + 12 * i) for i in range(ne)]
        # samples of each chunk
        self.per_chunk, s = [], 0
        for c in range(nc):
            spc = 0
            for first, cnt, _ in ent:
                if first - 1 <= c:
                    spc = cnt
            self.per_chunk.append(list(range(s, s + spc)))
            s += spc
        if s != n:
            raise Unsupported(f"sample table mismatch ({s} samples in chunks, {n} sizes)")

    def samples(self, buf):
        """-> [(file offset, size)] in sample order"""
        out = []
        for c, idx in enumerate(self.per_chunk):
            off = self.chunks[c]
            for i in idx:
                out.append((off, self.sizes[i]))
                off += self.sizes[i]
        return out

    def av1c_config(self, buf):
        """the av1C configOBUs (usually the sequence header), or b''"""
        typ, off, size, hdr = self.entry
        for b in boxes(buf, off + hdr + 78, off + size):        # VisualSampleEntry: 78 bytes before its boxes
            if b[0] == b"av1C":
                return bytes(buf[b[1] + b[3] + 4:b[1] + b[2]])
        return b""


def parse(buf):
    top = boxes(buf, 0, len(buf))
    types = [b[0] for b in top]
    if b"moof" in types:
        raise Unsupported("fragmented MP4")
    if types.count(b"moov") != 1 or types.count(b"mdat") != 1:
        raise Unsupported("needs exactly one moov and one mdat")
    moov = top[types.index(b"moov")]
    tracks = [Track(buf, b) for b in boxes(buf, moov[1] + moov[3], moov[1] + moov[2]) if b[0] == b"trak"]
    return top, tracks


def video_track(tracks):
    v = [t for t in tracks if t.codec == b"av01"]
    if not v:
        raise Unsupported("no AV1 track")
    if len(v) > 1:
        raise Unsupported("more than one AV1 track")
    return v[0]


def write_ivf(path, frames, w=0, h=0):
    with open(path, "wb") as f:
        f.write(struct.pack("<4sHH4sHHIIII", b"DKIF", 0, 32, b"AV01", w, h, 24000, 1001, len(frames), 0))
        for i, d in enumerate(frames):
            f.write(struct.pack("<IQ", len(d), i))
            f.write(d)


def read_ivf(path):
    b = Path(path).read_bytes()
    if b[:4] != b"DKIF":
        raise ValueError(f"{path}: not IVF")
    off, out = struct.unpack_from("<H", b, 6)[0], []
    while off + 12 <= len(b):
        n = struct.unpack_from("<I", b, off)[0]
        out.append(b[off + 12:off + 12 + n])
        off += 12 + n
    return out


def rebuild(buf, top, tracks, vt, new_samples):
    """-> the MP4 bytes with vt's samples replaced (same box order; chunk offsets and sample sizes patched)"""
    mdat = next(b for b in top if b[0] == b"mdat")
    m_payload = mdat[1] + mdat[3]
    chunks = []                                                  # (orig offset, track, chunk index)
    for t in tracks:
        for c, off in enumerate(t.chunks):
            if t.per_chunk[c]:
                chunks.append((off, t, c))
    chunks.sort(key=lambda x: x[0])
    parts, pos, new_off = [], 0, {}
    for off, t, c in chunks:
        idx = t.per_chunk[c]
        if t is vt:
            data = b"".join(new_samples[i] for i in idx)
        else:
            total = sum(t.sizes[i] for i in idx)
            if off < m_payload or off + total > mdat[1] + mdat[2]:
                raise Unsupported("a chunk outside the mdat")
            data = bytes(buf[off:off + total])
        new_off[(id(t), c)] = pos
        parts.append(data)
        pos += len(data)
    payload = b"".join(parts)
    out, base, moov_out = bytearray(), 0, 0
    for typ, off, size, hdr in top:
        if typ == b"mdat":
            if hdr == 16:
                out += struct.pack(">I4sQ", 1, b"mdat", 16 + len(payload))
            else:
                if 8 + len(payload) >= 1 << 32:
                    raise Unsupported("mdat too large")
                out += struct.pack(">I4s", 8 + len(payload), b"mdat")
            base = len(out)
            out += payload
        else:
            if typ == b"moov":
                moov_out = len(out)
            out += buf[off:off + size]
    delta = moov_out - next(b for b in top if b[0] == b"moov")[1]       # where the moov's bytes moved
    for t in tracks:
        for c in range(len(t.chunks)):
            if not t.per_chunk[c]:
                continue
            v = base + new_off[(id(t), c)]
            if t.co64:
                struct.pack_into(">Q", out, t.co_off + delta + 8 * c, v)
            else:
                if v >= 1 << 32:
                    raise Unsupported("chunk offset beyond 4 GiB")
                struct.pack_into(">I", out, t.co_off + delta + 4 * c, v)
    if vt.const_size:
        raise Unsupported("constant sample size on the AV1 track")
    for i, s in enumerate(new_samples):
        struct.pack_into(">I", out, vt.stsz_off + delta + 4 * i, len(s))
    return bytes(out)


def tool(a):
    exe = a.av1grain or os.environ.get("AV1GRAIN") or shutil.which("av1grain")
    if not exe or not os.access(exe, os.X_OK):
        sys.exit("av1grain_mp4: the av1grain binary was not found (--av1grain, $AV1GRAIN or $PATH)")
    return exe


def md5_of_decode(exe, ivf, cfg, threads, *opts):
    cmd = [exe, "decode", ivf, "-", "--threads", str(threads)] + list(opts) + (["--config", cfg] if cfg else [])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    h, n = hashlib.md5(), 0
    for chunk in iter(lambda: p.stdout.read(1 << 22), b""):
        h.update(chunk)
        n += len(chunk)
    err = p.stderr.read().decode(errors="replace")
    if p.wait() != 0 or not n:
        raise RuntimeError(f"decode failed: {' '.join(cmd)}\n{err}")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("params")
    ap.add_argument("--verify", action="store_true", help="decode both and check the identities (6 decodes)")
    ap.add_argument("--av1grain", help="path to the av1grain binary")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    exe = tool(a)
    buf = Path(a.src).read_bytes()
    try:
        top, tracks = parse(buf)
        vt = video_track(tracks)
    except Unsupported as e:
        print(f"av1grain_mp4: {a.src}: {e}", file=sys.stderr)
        return 4
    samples = [bytes(buf[o:o + n]) for o, n in vt.samples(buf)]
    with tempfile.TemporaryDirectory(prefix="av1grain-") as td:
        src_ivf, out_ivf, pj, cfg = (os.path.join(td, n) for n in ("in.ivf", "out.ivf", "p.json", "cfg.obu"))
        write_ivf(src_ivf, samples)
        config = vt.av1c_config(buf)
        if config:
            Path(cfg).write_bytes(config)
        r = subprocess.run([exe, "strip", src_ivf, out_ivf, pj, "--threads", str(a.threads)] +
                           (["--config", cfg] if config else []), capture_output=True, text=True)
        if r.returncode:
            print(f"av1grain_mp4: strip failed: {r.stderr.strip()}", file=sys.stderr)
            return 1
        print(r.stderr.strip(), file=sys.stderr)
        params = json.loads(Path(pj).read_text())
        if not params["sets"]:
            print(f"av1grain_mp4: {a.src}: no film grain", file=sys.stderr)
            return 3
        new = read_ivf(out_ivf)
        if len(new) != len(samples) or params["frames"] != len(samples):
            print(f"av1grain_mp4: {len(samples)} samples, {len(new)} rewritten, {params['frames']} pictures", file=sys.stderr)
            return 1
        try:
            out = rebuild(buf, top, tracks, vt, new)
        except Unsupported as e:
            print(f"av1grain_mp4: {a.src}: {e}", file=sys.stderr)
            return 4
        # read the result back: the same AV1 samples, every other track's samples unchanged
        top2, tracks2 = parse(out)
        vt2 = video_track(tracks2)
        if [bytes(out[o:o + n]) for o, n in vt2.samples(out)] != new:
            print("av1grain_mp4: the rewritten MP4 does not read back", file=sys.stderr)
            return 1
        for t, t2 in zip(tracks, tracks2):
            if t is not vt and [bytes(buf[o:o + n]) for o, n in t.samples(buf)] != [bytes(out[o:o + n]) for o, n in t2.samples(out)]:
                print("av1grain_mp4: another track changed", file=sys.stderr)
                return 1
        if a.verify:
            write_ivf(out_ivf, [bytes(out[o:o + n]) for o, n in vt2.samples(out)])
            c = cfg if config else None
            checks = [("grain off", md5_of_decode(exe, out_ivf, c, a.threads, "--grain", "off"),
                       md5_of_decode(exe, src_ivf, c, a.threads, "--grain", "off"))]
            for m in ("full", "standard"):
                checks.append((f"{m} grain from the params", md5_of_decode(exe, out_ivf, c, a.threads, "--grain", m, "--params", pj),
                               md5_of_decode(exe, src_ivf, c, a.threads, "--grain", m)))
            for name, got, want in checks:
                print(f"av1grain_mp4: verify {name}: {'ok' if got == want else 'FAILED'} ({got})", file=sys.stderr)
            if any(g != w for _, g, w in checks):
                return 1
        params["source"] = {"bytes": len(buf), "sha256": hashlib.sha256(buf).hexdigest()}
        params["nograin"] = {"bytes": len(out), "sha256": hashlib.sha256(out).hexdigest()}
        Path(a.out).write_bytes(out)
        Path(a.params).write_text(json.dumps(params, separators=(",", ":")) + "\n")
        print(f"av1grain_mp4: {a.out}: {len(buf)} -> {len(out)} bytes; {a.params}: {len(params['sets'])} set(s), "
              f"{sum(g[0] >= 0 for g in params['grain'])}/{params['frames']} pictures with grain", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
