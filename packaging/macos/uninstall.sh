#!/bin/sh
# Put Homebrew's stock libdav1d back (undo packaging/macos/install.sh).
set -eu
command -v brew >/dev/null 2>&1 || { echo "Homebrew is required"; exit 1; }
LIBDIR=$(cd "$(brew --prefix dav1d)/lib" && pwd -P)
LIB="$LIBDIR/libdav1d.7.dylib"
if [ -f "$LIB.stock" ]; then
    mv -f "$LIB.stock" "$LIB"
    echo "Restored the stock $LIB"
else
    echo "No backup found; reinstalling Homebrew's dav1d"
    brew reinstall dav1d
fi
