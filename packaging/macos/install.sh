#!/bin/sh
# Install dav1d-fullgrain on macOS by swapping the libdav1d dylib inside
# Homebrew's dav1d keg, so Homebrew's ffmpeg and mpv (and anything else that
# links Homebrew's libdav1d) use it. The stock dylib is kept next to it as
# libdav1d.7.dylib.stock.
#
#   packaging/macos/install.sh            # build, test, install (default mode: full)
#   GRAIN_MODE=dual packaging/macos/install.sh
#   packaging/macos/uninstall.sh          # put the stock dylib back
#
# Undone automatically by `brew upgrade dav1d` / `brew reinstall dav1d`: run
# this script again afterwards (or `brew pin dav1d` to hold the version).
# Apps that bundle their own libdav1d (IINA, mpv.app, browsers, VLC) are not
# affected.
set -eu

SRC=$(cd "$(dirname "$0")/../.." && pwd)
BUILD="$SRC/build"
MODE=${GRAIN_MODE:-full}

command -v brew >/dev/null 2>&1 || { echo "Homebrew is required (https://brew.sh)"; exit 1; }

echo "==> Homebrew packages"
brew list --formula dav1d >/dev/null 2>&1 || brew install dav1d
brew list --formula meson >/dev/null 2>&1 || brew install meson
brew list --formula ninja >/dev/null 2>&1 || brew install ninja
if [ "$(uname -m)" = x86_64 ]; then
    brew list --formula nasm >/dev/null 2>&1 || brew install nasm
fi

KEG=$(brew --prefix dav1d)
LIBDIR=$(cd "$KEG/lib" && pwd -P)
LIB="$LIBDIR/libdav1d.7.dylib"
[ -f "$LIB" ] || [ -f "$LIB.stock" ] || { echo "not found: $LIB"; exit 1; }

# The fork is dav1d 1.5.4 (API 7.0). Refuse to replace a different minor
# version unless forced: newer Homebrew builds of ffmpeg could use API
# functions this fork does not have.
BREWVER=$(basename "$(dirname "$LIBDIR")")
case "$BREWVER" in
    1.5.*) ;;
    *) if [ "${FORCE:-0}" != 1 ]; then
           echo "Homebrew's dav1d is $BREWVER, the fork is 1.5.4. Re-run with FORCE=1 to replace it anyway."
           exit 1
       fi ;;
esac

echo "==> Building dav1d-fullgrain (grain_mode_default=$MODE)"
if [ -d "$BUILD" ]; then
    meson setup --reconfigure "$BUILD" "$SRC" --buildtype=release -Denable_tools=true \
        -Denable_tests=false -Dgrain_mode_default="$MODE"
else
    meson setup "$BUILD" "$SRC" --buildtype=release -Denable_tools=true \
        -Denable_tests=false -Dgrain_mode_default="$MODE"
fi
ninja -C "$BUILD"

echo "==> Testing"
DEFAULT_MODE="$MODE" sh "$SRC/tests/fullgrain/run.sh" "$BUILD/tools/dav1d" | tail -1

echo "==> Installing into $LIBDIR"
[ -f "$LIB.stock" ] || cp -p "$LIB" "$LIB.stock"
NEW="$LIB.fullgrain.tmp"
cp "$BUILD/src/libdav1d.7.dylib" "$NEW"
chmod u+w "$NEW"
install_name_tool -id "$(otool -D "$LIB.stock" | tail -n 1)" "$NEW"
codesign --force --sign - "$NEW"
mv -f "$NEW" "$LIB"

echo "==> Check"
DAV1D_GRAIN_STATS=1 "$KEG/bin/dav1d" -q -i "$SRC/tests/fullgrain/fgt16_420_10bit_odd.ivf" \
    --muxer null -o - --filmgrain 1 2>&1 | head -n 1
echo "Done. Homebrew's ffmpeg/mpv now use dav1d-fullgrain (default mode: $MODE)."
echo "For mpv also add contrib/mpv/mpv.conf to ~/.config/mpv/mpv.conf."
