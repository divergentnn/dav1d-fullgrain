// The browser self-test (selftest.html) for another machine, reached through an SSH tunnel:
//
//   node wasm/test/selftest-serve.mjs VECTOR_DIR [PORT=8140] [RESULTS_DIR=VECTOR_DIR/results]
//   (on the other machine)  ssh -N -L 8140:127.0.0.1:8140 HOST   then open http://localhost:8140/
//
// localhost is a secure context (WebGPU, WebCodecs) and the server sends COOP/COEP (WebAssembly threads).
// The page POSTs its results to /results; they are saved as RESULTS_DIR/<time>-<browser>.json.
//
// Copyright © 2026, divergentnn (av1-fullgrain). SPDX-License-Identifier: BSD-2-Clause
import { mkdirSync, writeFileSync } from 'fs';
import { dirname, resolve, join } from 'path';
import { fileURLToPath } from 'url';
import { serve } from './serve.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const vec = resolve(process.argv[2]), port = +(process.argv[3] || 8140);
const out = resolve(process.argv[4] || join(vec, 'results'));
mkdirSync(out, { recursive: true });
const srv = await serve(port, [['/web/', resolve(here, '../web')], ['/web/', resolve(here, '../dist')],
                               ['/test/', here], ['/vec/', vec]]);
srv.prependListener('request', (req, res) => {
  if (req.url === '/' ) { res.writeHead(302, { Location: '/test/selftest.html?vec=/vec' }); res.end(); req.handled = true; }
  if (req.method === 'POST' && req.url === '/results') {
    let body = '';
    req.on('data', c => { body += c; if (body.length > 4e6) req.destroy(); });
    req.on('end', () => {
      let name = 'results';
      try { const j = JSON.parse(body); name = (j.env && j.env.browser || 'browser').replace(/[^a-z0-9]+/gi, '-').slice(0, 40); } catch {}
      const f = join(out, `${new Date().toISOString().replace(/[:.]/g, '-')}-${name}.json`);
      writeFileSync(f, body);
      console.log('saved', f);
      res.writeHead(200, { 'Content-Type': 'text/plain' }); res.end('saved');
    });
    req.handled = true;
  }
});
console.log(`self-test on http://127.0.0.1:${port}/  (vectors ${vec}, results ${out})`);
