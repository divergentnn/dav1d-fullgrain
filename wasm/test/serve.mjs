// Static file server for the browser checks: cross-origin isolated (COOP/COEP),
// with Range requests. Mounts: [[urlPrefix, directory], ...].
//
//   import { serve } from './serve.mjs';
//   const srv = await serve(8131, [['/web/', 'wasm/web'], ['/test/', 'wasm/test'], ['/vec/', '/tmp/vec']]);
//
// Copyright © 2026, divergentnn (av1-fullgrain). SPDX-License-Identifier: BSD-2-Clause
import http from 'http';
import { createReadStream, statSync } from 'fs';
import { join, normalize, extname } from 'path';

const TYPES = { '.html': 'text/html; charset=utf-8', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json',
  '.wasm': 'application/wasm', '.mp4': 'video/mp4', '.css': 'text/css' };

export function serve(port, mounts, host = '127.0.0.1') {
  const srv = http.createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    const m = mounts.find(([pre]) => url.startsWith(pre));
    const hdr = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp',
                  'Cross-Origin-Resource-Policy': 'same-origin', 'Cache-Control': 'no-cache' };
    if (!m) { res.writeHead(404, hdr); return res.end('not found'); }
    const rel = normalize(url.slice(m[0].length));
    if (rel.startsWith('..')) { res.writeHead(403, hdr); return res.end(); }
    const file = join(m[1], rel);
    let st;
    try { st = statSync(file); if (!st.isFile()) throw 0; } catch { res.writeHead(404, hdr); return res.end('not found'); }
    hdr['Content-Type'] = TYPES[extname(file)] || 'application/octet-stream';
    hdr['Accept-Ranges'] = 'bytes';
    const r = /^bytes=(\d+)-(\d*)$/.exec(req.headers.range || '');
    if (r) {
      const a = +r[1], b = r[2] ? Math.min(+r[2], st.size - 1) : st.size - 1;
      if (a >= st.size || b < a) { res.writeHead(416, { ...hdr, 'Content-Range': `bytes */${st.size}` }); return res.end(); }
      res.writeHead(206, { ...hdr, 'Content-Range': `bytes ${a}-${b}/${st.size}`, 'Content-Length': b - a + 1 });
      return req.method === 'HEAD' ? res.end() : createReadStream(file, { start: a, end: b }).pipe(res);
    }
    res.writeHead(200, { ...hdr, 'Content-Length': st.size });
    return req.method === 'HEAD' ? res.end() : createReadStream(file).pipe(res);
  });
  return new Promise(ok => srv.listen(port, host, () => ok(srv)));
}
