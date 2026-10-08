#!/bin/sh
# Grain-step CPU time per frame for each mode (thread CPU clock around the
# grain code, DAV1D_GRAIN_STATS=1), plus wall time of the whole decode.
#   tests/fullgrain/timing.sh FORK_DAV1D_CLI [STREAM] [THREADS]
FORK=$1
DIR=$(cd "$(dirname "$0")" && pwd)
F=${2:-$DIR/timing_720p_10bit_fgt03.ivf}
T=${3:-2}
unset DAV1D_GRAIN_MODE DAV1D_GRAIN_SIMD DAV1D_GRAIN_MODE_FILE
now() { perl -MTime::HiRes=time -e 'printf "%.3f", time'; }
for simd in 1 0; do
    for mode in standard standard_c multi dual full; do
        [ "$simd" = 0 ] && [ "$mode" = standard ] && continue
        [ "$simd" = 0 ] && [ "$mode" = standard_c ] && continue
        start=$(now)
        line=$(DAV1D_GRAIN_STATS=1 DAV1D_GRAIN_SIMD=$simd DAV1D_GRAIN_MODE=$mode \
               "$FORK" -q -i "$F" --muxer null -o - --filmgrain 1 --threads "$T" 2>&1 |
               grep 'frames=' | sed 's/.*per_frame_ms=\([0-9.]*\).*/\1/')
        end=$(now)
        printf '%-11s simd=%s  grain %8s ms/frame   wall %s s\n' "$mode" "$simd" "$line" \
            "$(awk "BEGIN { printf \"%.2f\", $end - $start }")"
    done
done
