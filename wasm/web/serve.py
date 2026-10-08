#!/usr/bin/env python3
"""Serve a directory with the headers the WebAssembly player needs
(cross-origin isolation for SharedArrayBuffer threads).

    python3 wasm/web/serve.py [DIR] [PORT]     # default: wasm/dist/web on 8077
"""
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    extensions_map = dict(SimpleHTTPRequestHandler.extensions_map,
                          **{".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm"})

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "dist" / "web")
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8077
    print(f"serving {root} on http://127.0.0.1:{port}/demo.html", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(root))).serve_forever()
