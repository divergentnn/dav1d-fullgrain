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
unset DAV1D_GRAIN_MODE DAV1D_GRAIN_SIMD DAV1D_GRAIN_MODE_FILE DAV1D_GRAIN_STATS
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
