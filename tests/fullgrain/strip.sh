#!/bin/sh
# av1grain strip checks (tools/av1grain.c), on every test stream:
#
#   tests/fullgrain/strip.sh BUILD/tools/av1grain
#
# 1. the stripped stream, decoded with grain off, is md5-identical to the
#    original decoded with grain off (only the grain syntax changed);
# 2. the stripped stream has no grain left: grain on == grain off;
# 3. the stripped stream with the grain from PARAMS.json applied (dav1d's
#    synthesis, full and standard) is md5-identical to the original decoded
#    with grain, in both modes.
set -u
AG=$1
DIR=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
unset DAV1D_GRAIN_MODE DAV1D_GRAIN_SIMD DAV1D_GRAIN_MODE_FILE DAV1D_GRAIN_STATS DAV1D_GRAIN_FORCE
fail=0
n=0
h() { "$AG" decode "$@" - 2>/dev/null | md5sum | cut -d' ' -f1; }
check() { # label got want
    n=$((n + 1))
    if [ "$2" = "$3" ] && [ "$2" != d41d8cd98f00b204e9800998ecf8427e ]; then
        echo "ok   $1"
    else
        echo "FAIL $1: got $2, want $3"
        fail=1
    fi
}
for f in "$DIR"/*.ivf; do
    b=$(basename "$f" .ivf)
    if ! "$AG" strip "$f" "$TMP/$b.ivf" "$TMP/$b.json" 2>"$TMP/$b.log"; then
        echo "FAIL $b: strip: $(cat "$TMP/$b.log")"; fail=1; continue
    fi
    off=$(h "$f" --grain off)
    check "$b stripped, grain off == original, grain off" "$(h "$TMP/$b.ivf" --grain off)" "$off"
    check "$b stripped has no grain" "$(h "$TMP/$b.ivf" --grain full)" "$off"
    for m in full standard; do
        check "$b stripped + params ($m) == original ($m)" \
            "$(h "$TMP/$b.ivf" --grain $m --params "$TMP/$b.json")" "$(h "$f" --grain $m)"
    done
done
echo "$n checks, $( [ $fail = 0 ] && echo 'all passed' || echo 'FAILURES' )"
exit $fail
