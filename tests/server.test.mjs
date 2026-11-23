import test from 'node:test';
import assert from 'node:assert/strict';
import http from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';

const crcTable = Array.from({ length: 256 }, (_, n) => {
  for (let k = 0; k < 8; k++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1;
  return n >>> 0;
});
function crc32(buf) {
  let c = 0xffffffff;
  for (const b of buf) c = crcTable[(c ^ b) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const head = Buffer.from(type);
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(Buffer.concat([head, data])), 0);
  return Buffer.concat([len, head, data, crc]);
}
function makeSolidPng(width, height, r, g, b, innerRect = null) {
  const row = 1 + width * 3;
  const raw = Buffer.alloc(height * row);
  for (let y = 0; y < height; y++) {
    raw[y * row] = 0;
    for (let x = 0; x < width; x++) {
      const idx = y * row + 1 + x * 3;
      const inBox = innerRect && x >= innerRect[0] && x < innerRect[2] && y >= innerRect[1] && y < innerRect[3];
      raw[idx] = inBox ? innerRect[4] : r;
      raw[idx + 1] = inBox ? innerRect[5] : g;
      raw[idx + 2] = inBox ? innerRect[6] : b;
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  return Buffer.concat([
    Buffer.from('89504e470d0a1a0a', 'hex'),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
test('standalone asset, session, export and optional refinement APIs', async t => {
  const data = await mkdtemp(path.join(os.tmpdir(), 'scene-demo-test-'));
  const child = spawn(process.execPath, ['server.mjs'], { cwd: root, env: { ...process.env, PORT: '0', HOST: '127.0.0.1', SCENE_DATA_DIR: data, COMFYUI_URL: '' }, windowsHide: true });
  t.after(async () => {
    const closed = new Promise(resolve => child.once('close', resolve));
    child.kill();
    await closed;
    await rm(data, { recursive: true, force: true });
  });
  const base = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Server startup timeout')), 10000);
    child.once('error', reject);
    child.stdout.on('data', chunk => {
      const match = chunk.toString().match(/http:\/\/127\.0\.0\.1:\d+/);
      if (match) { clearTimeout(timeout); resolve(match[0]); }
    });
  });
  const call = async (url, method = 'GET', input) => {
    const res = await fetch(base + url, { method, headers: { 'Content-Type': 'application/json' }, body: input ? JSON.stringify(input) : undefined });
    assert.equal(res.status, 200, await res.clone().text());
    return (await res.json()).data;
  };
  assert.equal((await call('/api/health')).refinement, false);
  const home = await fetch(base + '/');
  assert.equal(home.status, 200);
  assert.ok(home.headers.get('content-security-policy')?.includes("'wasm-unsafe-eval'"));
  const scenes = await call('/api/scenes/3d');
  assert.ok(scenes.scenes.some(s => s.code === 'VILLAGE_COMPLETE'));
  const calibrated = scenes.scenes.find(s => s.code === 'VILLAGE_CALIBRATED');
  if (calibrated?.splat_url) {
    const splatHead = await fetch(base + calibrated.splat_url, { method: 'HEAD' });
    assert.equal(splatHead.status, 200);
    assert.ok(Number(splatHead.headers.get('content-length')) > 1000);
  }
  assert.equal((await fetch(base + '/api/scenes/3d/VILLAGE_COMPLETE/splat', { method: 'HEAD' })).status, 404);
  const assets = await call('/api/assets/3d');
  const asset = assets.assets.find(a => a.code === 'Tank_model1');
  assert.ok(asset);
  const head = await fetch(base + asset.url, { method: 'HEAD' });
  assert.equal(head.headers.get('content-type'), 'model/gltf-binary');
  assert.ok(Number(head.headers.get('content-length')) > 1000);
  const s = await call('/api/scene/sessions', 'POST', { title: 'demo', ground_texture: 'scene:VILLAGE_COMPLETE', objects: [{ asset_code: asset.code, position: [1, 0, 2], rotation: [0, 1, 0], scale: [1, 1, 1] }] });
  await call(`/api/scene/sessions/${s.id}`, 'PUT', { title: 'saved', lighting: { intensity: 2 } });
  const restored = await call(`/api/scene/sessions/${s.id}`);
  assert.deepEqual(restored.objects[0].position, [1, 0, 2]);
  assert.equal(restored.lighting.intensity, 2);
  const task = await call('/api/generate', 'POST', { type: 'scene_export', params: { frames: [{ image_base64: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', width: 1, height: 1, objects: [{ name: 'demo', xmin: 0, ymin: 0, xmax: 1, ymax: 1 }] }] } });
  const zip = Buffer.from(await (await fetch(base + `/api/scene/exports/${task.task_id}/download`)).arrayBuffer());
  assert.equal(zip.readUInt32LE(0), 0x04034b50);
  for (const content of ['images/view_01.png', 'annotations/view_01.xml', '<name>demo</name>', 'manifest.json']) assert.ok(zip.includes(Buffer.from(content)));
  assert.equal((await call('/api/scene/exports')).exports.length, 1);
  assert.equal((await fetch(base + '/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ type: 'scene_refine' }) })).status, 503);
  await call(`/api/scene/sessions/${s.id}`, 'DELETE');
  assert.equal((await call('/api/scene/sessions')).sessions.length, 0);
});

test('MiniMax-H3-Image redraw endpoints, source lookup, ping and secret isolation', async t => {
  const data = await mkdtemp(path.join(os.tmpdir(), 'scene-demo-redraw-'));
  const child = spawn(process.execPath, ['server.mjs'], {
    cwd: root,
    env: { ...process.env, PORT: '0', HOST: '127.0.0.1', SCENE_DATA_DIR: data, COMFYUI_URL: '', MINIMAX_H3_COMFY_URL: '', MINIMAX_API_KEY: '' },
    windowsHide: true,
  });
  const repaintedPng = makeSolidPng(128, 128, 30, 120, 210);
  const gwState = { upstream: null };
  const mockUpstream = http.createServer(async (req, res) => {
    if (req.url === '/_gateway/status') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ state: 'ready', recovery_stage: 'connected', upstream_url: gwState.upstream }));
    }
    if (req.url === '/_gateway/set-upstream' && req.method === 'POST') {
      const chunks = [];
      for await (const c of req) chunks.push(c);
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      gwState.upstream = body.url || body.upstream_url;
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ ok: true, upstream_url: gwState.upstream }));
    }
    if (req.url === '/system_stats') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ system: { comfyui_version: '0.47.0' } }));
    }
    if (req.url === '/object_info/MiniMaxH3ImageConditioning') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ MiniMaxH3ImageConditioning: { input: {} } }));
    }
    if (req.url === '/object_info/MiniMaxH3ImageToVideo') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({ MiniMaxH3ImageToVideo: { input: {} } }));
    }
    if (req.url === '/v1/images/edits' && req.method === 'POST') {
      assert.equal(req.headers.authorization, 'Bearer ui-secret-token');
      const chunks = [];
      for await (const c of req) chunks.push(c);
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      assert.equal(body.model, 'MiniMax-H3-Image');
      assert.equal(body.images.length, 2);
      res.writeHead(200, { 'Content-Type': 'application/json' });
      return res.end(JSON.stringify({
        request_id: 'mock_req_1',
        data: [{ b64_json: repaintedPng.toString('base64'), revised_prompt: 'repainted ok', size: '128x128' }],
      }));
    }
    res.writeHead(404);
    res.end();
  });
  await new Promise(resolve => mockUpstream.listen(0, '127.0.0.1', resolve));
  const mockUrl = `http://127.0.0.1:${mockUpstream.address().port}`;
  t.after(async () => {
    mockUpstream.close();
    const closed = new Promise(resolve => child.once('close', resolve));
    child.kill();
    await closed;
    await rm(data, { recursive: true, force: true });
  });
  const base = await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error('Server startup timeout')), 10000);
    child.once('error', reject);
    child.stdout.on('data', chunk => {
      const match = chunk.toString().match(/http:\/\/127\.0\.0\.1:\d+/);
      if (match) { clearTimeout(timeout); resolve(match[0]); }
    });
  });
  const call = async (url, method = 'GET', input) => {
    const res = await fetch(base + url, { method, headers: { 'Content-Type': 'application/json' }, body: input ? JSON.stringify(input) : undefined });
    const body = await res.json();
    assert.ok(res.ok, JSON.stringify(body));
    return body.data;
  };

  const scenes = await call('/api/scenes/3d');
  const blender = scenes.scenes.find(s => s.code === 'VILLAGE_BLENDER');
  assert.equal(blender?.source_url, '/api/scenes/3d/VILLAGE_BLENDER/source');
  const srcRes = await fetch(base + blender.source_url);
  assert.equal(srcRes.status, 200);
  assert.equal(srcRes.headers.get('content-type'), 'image/png');

  const cfg = await call('/api/scene/redraw/config');
  assert.equal(cfg.providers.minimax_api.configured, false);
  assert.equal(cfg.presets.length, 3);

  const ping = await call('/api/scene/redraw/ping', 'POST', { url: mockUrl });
  assert.equal(ping.ok, true);
  assert.equal(ping.has_minimax_h3_image, true);
  assert.equal(ping.has_minimax_h3_video, true);
  assert.equal(ping.gateway_state, 'ready');

  const takeover = await call('/api/scene/redraw/gateway/set-upstream', 'POST', {
    gateway_url: mockUrl,
    upstream_url: 'https://demo-tunnel.trycloudflare.com',
  });
  assert.equal(takeover.ok, true);
  assert.equal(takeover.upstream_url, 'https://demo-tunnel.trycloudflare.com');
  assert.equal(gwState.upstream, 'https://demo-tunnel.trycloudflare.com');

  const shotPng = makeSolidPng(128, 128, 90, 90, 90, [32, 32, 96, 96, 220, 40, 40]);
  const envPng = makeSolidPng(128, 128, 80, 80, 80);
  const maskPng = makeSolidPng(128, 128, 0, 0, 0, [32, 32, 96, 96, 255, 255, 255]);

  const started = await call('/api/scene/redraw', 'POST', {
    provider: 'minimax_api',
    api_key: 'ui-secret-token',
    api_base_url: `${mockUrl}/v1`,
    scene_code: 'VILLAGE_BLENDER',
    style_preset: 'faithful',
    protect_equipment: true,
    source_image: 'data:image/png;base64,' + shotPng.toString('base64'),
    environment_image: 'data:image/png;base64,' + envPng.toString('base64'),
    protection_mask: 'data:image/png;base64,' + maskPng.toString('base64'),
  });
  assert.ok(started.task_id.startsWith('redraw_'));

  let state = null;
  for (let i = 0; i < 40; i++) {
    state = await call(`/api/scene/redraw/${started.task_id}`);
    if (state.status !== 'running') break;
    await new Promise(r => setTimeout(r, 150));
  }
  assert.equal(state?.status, 'completed', JSON.stringify(state));
  assert.equal(state.equipment_protected, true);
  assert.ok(state.result_url && state.mask_url);

  const inputDisk = await readFile(path.join(data, 'redraws', started.task_id, 'input.json'), 'utf8');
  assert.ok(!inputDisk.includes('ui-secret-token'));
  const metaDisk = await readFile(path.join(data, 'redraws', started.task_id, 'result.meta.json'), 'utf8');
  assert.ok(!metaDisk.includes('ui-secret-token'));

  const resultImg = await fetch(base + state.result_url);
  assert.equal(resultImg.status, 200);
  assert.equal(resultImg.headers.get('content-type'), 'image/png');

  const list = await call('/api/scene/redraws');
  assert.equal(list.redraws.length, 1);
  assert.equal(list.redraws[0].task_id, started.task_id);
});
