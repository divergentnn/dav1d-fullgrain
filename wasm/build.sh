#!/bin/sh
# Build dav1d-fullgrain for WebAssembly (SIMD128 + threads).
#
#   wasm/build.sh                 -> wasm/dist/dav1d-fg.mjs + dav1d-fg.wasm (ES module for a Worker)
#                                    and wasm/dist/web/ (decoder + worker + player + demo page)
#   NODE_CLI=1 wasm/build.sh      -> also wasm/dist/node/dav1d.js, the dav1d CLI for Node.js
#                                    (sh tests/fullgrain/run.sh wasm/dist/node/dav1d-node)
#
# Needs emsdk. With EMSDK set (an activated emsdk), that one is used; otherwise
# emsdk is cloned into wasm/emsdk and EMSDK_VERSION is installed there.
# Also needs meson and ninja.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(dirname "$HERE")
OUT="$HERE/dist"
BUILD="$HERE/build"
EMSDK_VERSION=${EMSDK_VERSION:-6.0.12}
JOBS=${JOBS:-4}
MODE=${GRAIN_MODE:-full}

if [ -z "${EMSDK:-}" ]; then
    EMSDK="$HERE/emsdk"
    if [ ! -d "$EMSDK" ]; then
        git clone --depth 1 https://github.com/emscripten-core/emsdk.git "$EMSDK"
    fi
    "$EMSDK/emsdk" install "$EMSDK_VERSION"
    "$EMSDK/emsdk" activate "$EMSDK_VERSION" >/dev/null
fi
# shellcheck disable=SC1091
. "$EMSDK/emsdk_env.sh" >/dev/null 2>&1
emcc --version | head -n 1

mkdir -p "$BUILD" "$OUT"
cat > "$BUILD/wasm32.meson" <<'EOF'
[binaries]
c = 'emcc'
cpp = 'em++'
ar = 'emar'
strip = 'emstrip'
exe_wrapper = 'node'

[built-in options]
c_args = ['-msimd128', '-pthread']
c_link_args = ['-msimd128', '-pthread']

[host_machine]
system = 'emscripten'
cpu_family = 'wasm32'
cpu = 'wasm32'
endian = 'little'
EOF

# 1. libdav1d.a (no asm on wasm; the grain variants use their NEON kernels
#    through emscripten's SIMDe arm_neon.h, everything else is C)
if [ ! -f "$BUILD/lib/build.ninja" ]; then
    meson setup "$BUILD/lib" "$SRC" --cross-file "$BUILD/wasm32.meson" --buildtype=release \
        -Ddefault_library=static -Denable_tools=false -Denable_tests=false \
        -Denable_asm=false -Dlogging=false -Dgrain_mode_default="$MODE"
fi
ninja -C "$BUILD/lib" -j "$JOBS"

# 2. the JS module: djs.c + libdav1d.a
FUNCS='_djs_open,_djs_send,_djs_drain,_djs_queued,_djs_pop,_djs_w,_djs_h,_djs_bpc,_djs_layout,_djs_stride,_djs_plane,_djs_grain,_djs_mtrx,_djs_trc,_djs_pri,_djs_full_range,_djs_flush,_djs_close,_malloc,_free'
emcc -O3 -msimd128 -pthread "$HERE/djs.c" "$BUILD/lib/src/libdav1d.a" \
    -I "$SRC/include" -I "$BUILD/lib/include" -I "$BUILD/lib/include/dav1d" \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createDav1d \
    -sENVIRONMENT=web,worker,node \
    -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=128MB -sMAXIMUM_MEMORY=2GB \
    -sPTHREAD_POOL_SIZE=8 -sSTACK_SIZE=1MB -sDEFAULT_PTHREAD_STACK_SIZE=2MB \
    -sEXPORTED_FUNCTIONS="$FUNCS" \
    -sEXPORTED_RUNTIME_METHODS=HEAPU8,stringToNewUTF8,wasmMemory \
    -Wno-pthreads-mem-growth \
    -o "$OUT/dav1d-fg.mjs"
ls -l "$OUT"/dav1d-fg.*

# 3. the player kit, flat in one directory (the worker imports its siblings)
mkdir -p "$OUT/web"
cp "$OUT/dav1d-fg.mjs" "$OUT/dav1d-fg.wasm" "$HERE/web/dav1d-worker.mjs" "$HERE/web/wasm-clip.mjs" \
   "$HERE/web/demo.html" "$HERE"/web/vendor/mp4box/*.mjs "$OUT/web/"
cp "$HERE/web/vendor/mp4box/LICENSE" "$OUT/web/mp4box.LICENSE"
echo "player kit: $OUT/web (python3 $HERE/web/serve.py, then /demo.html)"

# 4. optional: the dav1d CLI for Node.js (runs tests/fullgrain/run.sh)
if [ "${NODE_CLI:-0}" = 1 ]; then
    cat > "$BUILD/wasm32-node.meson" <<'EOF'
[binaries]
c = 'emcc'
cpp = 'em++'
ar = 'emar'
strip = 'emstrip'
exe_wrapper = 'node'

[built-in options]
c_args = ['-msimd128', '-pthread']
c_link_args = ['-msimd128', '-pthread', '-sNODERAWFS=1', '-sALLOW_MEMORY_GROWTH=1', '-sPTHREAD_POOL_SIZE=12', '-sENVIRONMENT=node', '-sSTACK_SIZE=2MB', '-sDEFAULT_PTHREAD_STACK_SIZE=2MB', '-Wno-pthreads-mem-growth']

[host_machine]
system = 'emscripten'
cpu_family = 'wasm32'
cpu = 'wasm32'
endian = 'little'
EOF
    if [ ! -f "$BUILD/node/build.ninja" ]; then
        meson setup "$BUILD/node" "$SRC" --cross-file "$BUILD/wasm32-node.meson" --buildtype=release \
            -Ddefault_library=static -Denable_tools=true -Denable_tests=false \
            -Denable_asm=false -Dlogging=false -Dgrain_mode_default="$MODE"
    fi
    ninja -C "$BUILD/node" -j "$JOBS"
    mkdir -p "$OUT/node"
    cp "$BUILD/node/tools/dav1d.js" "$BUILD/node/tools/dav1d.wasm" "$OUT/node/"
    printf '#!/bin/sh\nexec node "$(dirname "$0")/dav1d.js" "$@"\n' > "$OUT/node/dav1d-node"
    chmod +x "$OUT/node/dav1d-node"
    echo "node CLI: $OUT/node/dav1d-node"
fi
