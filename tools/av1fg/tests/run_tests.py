#!/usr/bin/env python3
"""av1-grain-seed tests (meson test --suite av1fg).

    run_tests.py TOOL FIXTURES [--dav1d-check CHECK] [--extra DIR] [--update-golden]

For every fixture (IVF, Section 5, Annex B, MP4, fragmented MP4, Matroska from
ffmpeg and mkvmerge, WebM; tools/av1fg/tests/fixtures and tests/fullgrain):
  - it parses; container variants of one stream give the same frames;
  - with CHECK (libdav1d): the grain parameters of every output picture equal
    dav1d's, and the patched file decodes to identical pictures with grain
    off and to the original plus new-seed grain with grain on;
  - -o writes a same-size file that differs only in the seed fields (and the
    Matroska CRC-32 elements, which stay valid), parses to the same frames
    with new seeds, and matches the committed md5 (the seed choice is the
    same on every platform);
  - --in-place gives the same bytes as -o; --set-seeds writes the given seeds;
  - laced Matroska blocks, truncated and non-AV1 files are refused.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
fails = 0
checks = 0


def check(ok, what):
    global fails, checks
    checks += 1
    print(("ok   " if ok else "FAIL ") + what, flush=True)
    if not ok:
        fails += 1


def run(cmd, env=None):
    e = dict(os.environ, DAV1D_GRAIN_MODE="standard", DAV1D_GRAIN_FORCE="0")
    if env:
        e.update(env)
    return subprocess.run(cmd, capture_output=True, text=True, env=e)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def seed_fields(tool, path):
    """byte offsets that hold seed bits, from --dump"""
    out = run([tool, "-q", "--dump", path]).stdout
    offs = set()
    for line in out.splitlines():
        m = re.match(r"obu (\d+) .* seed_bit (\d+) ", line)
        if m:
            o, b = int(m.group(1)), int(m.group(2))
            for k in range(b // 8, (b + 15) // 8 + 1):
                offs.add(o + k)
    return offs


def ebml_crcs(data):
    """(offset of CRC data, covered start, end) of every CRC-32 element in master elements we walk"""
    res = []

    def vint(i, keep):
        n = 1
        while n <= 8 and not data[i] & (0x80 >> (n - 1)):
            n += 1
        v = data[i] if keep else data[i] & (0xFF >> n)
        for k in range(1, n):
            v = v << 8 | data[i + k]
        return v, i + n, (not keep) and v == (1 << (7 * n)) - 1

    def walk(s, e):
        i, first = s, True
        while i < e:
            eid, j, _ = vint(i, True)
            sz, k, unk = vint(j, False)
            end = e if unk else k + sz
            if eid == 0xBF and first:
                res.append((k, end, e))
            if eid in (0x18538067, 0x1F43B675, 0xA0, 0x1654AE6B, 0xAE, 0x1549A966, 0x114D9B74, 0x1C53BB6B,
                       0x1254C367):
                walk(k, end)
            first = False
            i = end

    walk(0, len(data))
    return res


def main():
    args = sys.argv[1:]
    tool, fixtures = args[0], args[1]
    dcheck = args[args.index("--dav1d-check") + 1] if "--dav1d-check" in args else None
    extra = args[args.index("--extra") + 1] if "--extra" in args else None
    update = "--update-golden" in args
    golden_path = os.path.join(HERE, "expected.md5")
    golden = {}
    if os.path.exists(golden_path):
        for line in open(golden_path):
            if line.strip() and not line.startswith("#"):
                h, n = line.split()
                golden[n] = h
    files = sorted(os.path.join(fixtures, f) for f in os.listdir(fixtures) if not f.startswith("."))
    if extra:
        files += sorted(os.path.join(extra, f) for f in os.listdir(extra) if f.endswith(".ivf"))
    tmp = tempfile.mkdtemp(prefix="av1fg-")
    new_golden = {}
    dumps = {}
    try:
        for f in files:
            name = os.path.basename(f)
            ext = name.rsplit(".", 1)[-1]
            r = run([tool, "--dump-display", f])
            check(r.returncode == 0 and r.stdout.strip() != "", f"{name}: parses")
            dumps[name] = r.stdout
            if dcheck and ext in ("ivf", "obu"):
                d = run([dcheck, "params", f])
                check(d.returncode == 0 and d.stdout == r.stdout, f"{name}: grain parameters == libdav1d")
            out = os.path.join(tmp, "out." + ext)
            w = run([tool, "-q", "-o", out, f])
            check(w.returncode == 0, f"{name}: -o (rc {w.returncode} {w.stderr.strip()[-200:]})")
            if w.returncode:
                continue
            a, b = open(f, "rb").read(), open(out, "rb").read()
            diff = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
            fields = seed_fields(tool, f)
            crc_bytes = set()
            if ext in ("mkv", "webm"):
                crcs = ebml_crcs(b)
                check(all(zlib.crc32(b[s:e]) == int.from_bytes(b[k:k + 4], "little") for k, s, e in crcs),
                      f"{name}: {len(crcs)} CRC-32 elements valid after the edit")
                crc_bytes = {k + i for k, _, _ in crcs for i in range(4)}
            check(len(a) == len(b) and diff and all(i in fields or i in crc_bytes for i in diff),
                  f"{name}: same size, {len(diff)} bytes changed, all in seed fields{' or CRC-32' if crc_bytes else ''}")
            p = run([tool, "--dump-display", out]).stdout.splitlines()
            o = r.stdout.splitlines()
            strip = lambda s: re.sub(r" seed \d+ ", " ", s)
            check(len(p) == len(o) and all(strip(x) == strip(y) for x, y in zip(o, p)),
                  f"{name}: same frames and grain parameters, new seeds")
            seeds = [int(m.group(1)) for m in (re.search(r" seed (\d+) ", s) for s in p) if m]
            check(all(seeds[i] != seeds[i + 1] for i in range(len(seeds) - 1)), f"{name}: no seed twice in a row")
            h = md5(out)
            new_golden[name] = h
            if not update:
                check(golden.get(name) == h, f"{name}: output md5 {h} == expected {golden.get(name)}")
            if dcheck and ext in ("ivf", "obu"):
                e = run([dcheck, "edit", f, out])
                check(e.returncode == 0 and "OK" in e.stdout, f"{name}: decode check: {e.stdout.strip()}")
            ip = os.path.join(tmp, "inplace." + ext)
            shutil.copyfile(f, ip)
            check(run([tool, "-q", "--in-place", ip]).returncode == 0 and md5(ip) == h, f"{name}: --in-place == -o")
        # container variants of fp.ivf: same frames
        base = dumps.get("fp.ivf")
        for n, d in dumps.items():
            if n.startswith("fp.") and n != "fp.ivf":
                check(d == base, f"{n}: same frames and parameters as fp.ivf")
        # --set-seeds
        f = os.path.join(fixtures, "fp.ivf")
        out = os.path.join(tmp, "set.ivf")
        run([tool, "-q", "-o", out, "--set-seeds", "1234,4321,777", f])
        seeds = [int(m.group(1)) for m in (re.search(r" seed (\d+) ", s) for s in
                                            run([tool, "--dump-display", out]).stdout.splitlines()) if m]
        check(seeds[:6] == [1234, 4321, 777, 1234, 4321, 777], f"--set-seeds: {seeds[:6]}")
        # refusals
        bad = os.path.join(tmp, "laced.mkv")
        data = bytearray(open(os.path.join(fixtures, "fp.mkvmerge.mkv"), "rb").read())
        i = data.find(b"\xa3", data.find(b"\x1f\x43\xb6\x75"))  # first SimpleBlock of the first Cluster
        sz_len = 1
        while not data[i + 1] & (0x80 >> (sz_len - 1)):
            sz_len += 1
        data[i + 1 + sz_len + 3] |= 0x02  # flags: Xiph lacing (track number is one byte)
        open(bad, "wb").write(data)
        r = run([tool, "-o", os.path.join(tmp, "x.mkv"), bad])
        check(r.returncode == 1 and "lac" in r.stderr, "laced Matroska block refused")
        trunc = os.path.join(tmp, "trunc.ivf")
        open(trunc, "wb").write(open(f, "rb").read()[:3000])
        r = run([tool, "-o", os.path.join(tmp, "x.ivf"), trunc])
        check(r.returncode == 1, "truncated IVF refused")
        junk = os.path.join(tmp, "junk.bin")
        open(junk, "wb").write(b"this is not a video file" * 40)
        check(run([tool, junk]).returncode == 1, "non-AV1 file refused")
        r = run([tool, "--analyze", os.path.join(fixtures, "fp.mkvmerge.mkv")])
        check(r.returncode == 0 and "grain seeds:" in r.stdout, "--analyze report")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if update:
        with open(golden_path, "w") as fh:
            fh.write("# md5 of `av1-grain-seed -o` outputs (default options); the same on every platform\n")
            for n in sorted(new_golden):
                fh.write(f"{new_golden[n]} {n}\n")
        print(f"wrote {golden_path}")
    print(f"{checks} checks, {'all passed' if not fails else str(fails) + ' FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
