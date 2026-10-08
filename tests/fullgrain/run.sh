#!/bin/sh
# dav1d-fullgrain tests.
#
#   tests/fullgrain/run.sh FORK_DAV1D_CLI [UPSTREAM_DAV1D_CLI]
#
# 1. every grain mode decodes each stream to the expected md5, both with the
#    SIMD kernels (AVX2 / NEON, 4 threads) and with the C paths
#    (DAV1D_GRAIN_SIMD=0, 1 thread); the hashes are the same on every
#    architecture, so x86-64 and arm64 are checked against one table
# 2. the default mode (no DAV1D_GRAIN_MODE) is the meson default
#    (DEFAULT_MODE, default "full")
# 3. if an upstream dav1d CLI is given: DAV1D_GRAIN_MODE=standard (and
#    standard_c) is md5-identical to upstream on every stream
#
# The streams are synthetic gradients encoded by aomenc with its film grain
# test vectors (--film-grain-test=N): AR lag 2 and 3, overlap on/off,
# chroma_scaling_from_luma, grain_scale_shift 0-2, both clip ranges,
# 4:2:0 / 4:2:2 / 4:4:4, 8 and 10 bit, an odd frame size.
set -u
FORK=$1
UP=${2:-}
DEFAULT_MODE=${DEFAULT_MODE:-full}
DIR=$(cd "$(dirname "$0")" && pwd)
unset DAV1D_GRAIN_MODE DAV1D_GRAIN_SIMD DAV1D_GRAIN_MODE_FILE DAV1D_GRAIN_STATS DAV1D_GRAIN_FORCE
fail=0
n=0

dec() { # mode simd threads file
    DAV1D_GRAIN_MODE=$1 DAV1D_GRAIN_SIMD=$2 "$FORK" -q -i "$4" --muxer md5 -o - \
        --filmgrain 1 --threads "$3"
}
check() { # label got want
    n=$((n + 1))
    if [ "$2" = "$3" ]; then
        echo "ok   $1"
    else
        echo "FAIL $1: got $2, want $3"
        fail=1
    fi
}

while read -r name mode want; do
    case "$name" in ''|'#'*) continue ;; esac
    f="$DIR/$name"
    check "$name $mode simd" "$(dec "$mode" 1 4 "$f")" "$want"
    check "$name $mode C"    "$(dec "$mode" 0 1 "$f")" "$want"
    if [ "$mode" = "$DEFAULT_MODE" ]; then
        check "$name default=$mode" \
            "$("$FORK" -q -i "$f" --muxer md5 -o - --filmgrain 1)" "$want"
    fi
done < "$DIR/expected.md5"

# grain_force (default on): a caller that turns grain off (--filmgrain 0, as
# ffmpeg does for mpv's GPU grain) still gets the default variant; with
# DAV1D_GRAIN_FORCE=0, or in standard mode, it gets no grain (upstream
# behaviour). Checked on the first stream of the default mode.
first=$(awk -v m="$DEFAULT_MODE" '$1 !~ /^#/ && $2 == m {print $1, $3; exit}' "$DIR/expected.md5")
if [ -n "$first" ] && [ "$DEFAULT_MODE" != standard ] && [ "$DEFAULT_MODE" != standard_c ]; then
    f="$DIR/${first%% *}"; want="${first#* }"
    nog=$(DAV1D_GRAIN_MODE=standard "$FORK" -q -i "$f" --muxer md5 -o - --filmgrain 0)
    check "${first%% *} force: grain off -> $DEFAULT_MODE" \
        "$("$FORK" -q -i "$f" --muxer md5 -o - --filmgrain 0)" "$want"
    check "${first%% *} DAV1D_GRAIN_FORCE=0: grain off -> none" \
        "$(DAV1D_GRAIN_FORCE=0 "$FORK" -q -i "$f" --muxer md5 -o - --filmgrain 0)" "$nog"
fi

if [ -n "$UP" ]; then
    for f in "$DIR"/*.ivf; do
        u=$("$UP" -q -i "$f" --muxer md5 -o - --filmgrain 1)
        b=$(basename "$f")
        check "$b standard == upstream" "$(dec standard 1 4 "$f")" "$u"
        check "$b standard_c == upstream" "$(dec standard_c 1 1 "$f")" "$u"
    done
fi

echo "$n checks, $( [ $fail = 0 ] && echo 'all passed' || echo 'FAILURES' )"
exit $fail
