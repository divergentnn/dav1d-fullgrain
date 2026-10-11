// Headless Chromium run of gpu-test.html: GPU grain == dav1d, picture by picture.
//
//   PLAYWRIGHT=.../node_modules/playwright-core/index.mjs node wasm/test/gpu-test.mjs VECTOR_DIR \
//       [--mode full,standard] [--pp high-performance|low-power|swiftshader] [--only a,b] [--frames N] [--reps N]
//       [--page webcodecs-test.html [--hw prefer-software]] [--engine wasm]   (wasm: fg-grain.wasm from wasm/dist)
//
// VECTOR_DIR comes from wasm/test/gpu-vectors.py. WebGPU in headless Chromium on Linux needs Vulkan
// (--enable-unsafe-webgpu --enable-features=Vulkan --use-angle=vulkan); --pp swiftshader uses the CPU.
// Exit status 0 when every picture is identical.
//
// Copyright © 2026, divergentnn (av1-fullgrain). SPDX-License-Identifier: BSD-2-Clause
import { readdirSync } from 'fs';
import { homedir } from 'os';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import { serve } from './serve.mjs';

const { chromium } = await import(process.env.PLAYWRIGHT || 'playwright-core');
const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const vecDir = resolve(args[0]);
const pp = opt('pp', 'high-performance');
const port = +opt('port', 8131);
const srv = await serve(port, [['/web/', resolve(here, '../web')], ['/web/', resolve(here, '../dist')], ['/test/', here], ['/vec/', vecDir]]);
const pw = homedir() + '/.cache/ms-playwright';
const dir = readdirSync(pw).filter(d => /^chromium-\d+$/.test(d)).sort().pop();
const flags = pp === 'swiftshader'
  ? ['--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader', '--enable-unsafe-swiftshader']
  : ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--ignore-gpu-blocklist'];
const browser = await chromium.launch({ executablePath: process.env.CHROME || `${pw}/${dir}/chrome-linux64/chrome`, args: flags });
const page = await browser.newPage();
page.on('console', m => console.log(m.text()));
page.on('pageerror', e => console.log('pageerror:', e.message));
const q = new URLSearchParams({ vec: '/vec', mode: opt('mode', 'full'), pp: pp === 'swiftshader' ? 'high-performance' : pp,
  frames: opt('frames', '0'), reps: opt('reps', '3'), bench: opt('bench', '0') });
if (opt('only')) q.set('only', opt('only'));
if (opt('dump')) q.set('dump', '1');
if (args.includes('--repcheck')) q.set('repcheck', '1');
if (opt('hw')) q.set('hw', opt('hw'));
if (opt('engine')) q.set('engine', opt('engine'));
await page.goto(`http://127.0.0.1:${port}/test/${opt('page', 'gpu-test.html')}?${q}`);
await page.waitForFunction(() => window.results && window.results.done, null, { timeout: +opt('timeout', 1800) * 1000, polling: 500 });
const r = await page.evaluate(() => window.results);
if (args.includes('--print')) console.log(JSON.stringify(r, (k, v) => (k === 'dump' ? undefined : v), 1));
if (opt('dump') && r.dump) {
  (await import('fs')).writeFileSync(opt('dump'), Buffer.from(r.dump.b64, 'base64'));
  console.log(`wrote ${r.dump.name} ${r.dump.mode} frame ${r.dump.frame} to ${opt('dump')}`);
}
await browser.close();
srv.close();
process.exit(r.fail || r.error || !r.pass ? 1 : 0);
