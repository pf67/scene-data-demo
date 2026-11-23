// Offline, dependency-free HTTP service for the existing SceneEditor API.
import http from 'node:http';
import { createReadStream } from 'node:fs';
import { readFile, writeFile, rename, mkdir, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createBuilder } from './builder/service.mjs';

const ROOT = process.env.SCENE_ROOT || path.dirname(fileURLToPath(import.meta.url));
const DATA = process.env.SCENE_DATA_DIR || path.join(ROOT, 'data');
const PORT = Number(process.env.PORT || 8090);
const HOST = process.env.HOST || '127.0.0.1';
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.glb': 'model/gltf-binary', '.ply': 'application/octet-stream', '.spz': 'application/octet-stream', '.splat': 'application/octet-stream', '.ksplat': 'application/octet-stream', '.sog': 'application/octet-stream', '.wasm': 'application/wasm', '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.zip': 'application/zip', '.woff2': 'font/woff2' };
const lightingDefault = { azimuth: 55, elevation: 58, intensity: 2.6, ambient: .6, temperature: 5700 };
await mkdir(path.join(DATA, 'sessions'), { recursive: true });
await mkdir(path.join(DATA, 'exports'), { recursive: true });
await mkdir(path.join(DATA, 'refinements'), { recursive: true });
await mkdir(path.join(DATA, 'redraws'), { recursive: true });
const activeRefinements = new Set();
const activeRedraws = new Set();
const builder = await createBuilder({ root: ROOT, data: DATA, file, json, fail, body, readJson });

async function startRefinement(input) {
  if (!process.env.COMFYUI_URL) fail('请配置 COMFYUI_URL 并安装 refinement/requirements.txt 后启用 Qwen 重绘', 503);
  const params = input.params || {};
  if (params.workflow_template && params.workflow_template !== 'scene_refine_qwen21') fail('Only Qwen-Image-2.1 is supported');
  for (const key of ['base64_image', 'protection_mask', 'environment_image', 'prompt']) {
    if (typeof params[key] !== 'string' || !params[key]) fail(`Missing ${key}`);
  }
  const taskId = `refine_${randomUUID()}`;
  const directory = path.join(DATA, 'refinements', taskId);
  await mkdir(directory);
  await atomicJson(path.join(directory, 'input.json'), params);
  await atomicJson(path.join(directory, 'status.json'), { status: 'running' });
  activeRefinements.add(taskId);
  const child = spawn(process.env.SCENE_PYTHON || (process.platform === 'win32' ? 'python' : 'python3'), [path.join(ROOT, 'refinement', 'worker.py'), directory], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] });
  let stderr = '', spawnError = '';
  child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4000); });
  child.on('error', error => { spawnError = error.message; });
  child.on('close', async code => {
    try {
      await atomicJson(path.join(directory, 'status.json'), code === 0 ? { status: 'completed' } : { status: 'failed', error_message: spawnError || stderr || `Worker exited ${code}` });
    } catch (error) { console.error(error.message); }
    finally { activeRefinements.delete(taskId); }
  });
  return { task_id: taskId, status: 'running' };
}

async function refinementResult(input) {
  const taskId = safeName(input.task_id);
  const directory = path.join(DATA, 'refinements', taskId);
  const state = await readJson(path.join(directory, 'status.json'), null);
  if (!state) fail('Task not found', 404);
  if (state.status === 'running' && !activeRefinements.has(taskId)) return { status: 'failed', error_message: 'Service restarted during this task; submit again' };
  if (state.status === 'completed') state.result = [{ image: (await readFile(path.join(directory, 'result.png'))).toString('base64') }];
  return state;
}

function safeSourceImage(value) {
  if (typeof value !== 'string' || value.includes('..') || !/^[a-zA-Z0-9._-]+\.(png|jpe?g|webp)$/i.test(value)) return null;
  return value;
}

async function resolveSceneSourcePath(code) {
  const fallback = path.join(ROOT, 'assets', 'scene_sources', 'village_road_near.png');
  if (typeof code !== 'string' || !code.trim()) return fallback;
  const built = await builder.sourceFile(code);
  if (built) return built;
  try {
    const safeCode = safeName(code);
    const meta = await readJson(path.join(ROOT, 'assets', 'scene_meshes', `${safeCode}.json`), {});
    const sourceName = safeSourceImage(meta.source_image);
    if (sourceName) {
      const candidate = path.join(ROOT, 'assets', 'scene_sources', sourceName);
      if ((await stat(candidate)).isFile()) return candidate;
    }
    if (meta.source_url === '/api/builder/sources/village/image') return fallback;
  } catch { /* fallback to default village source */ }
  return fallback;
}

function validateExternalHttpUrl(raw, label = '服务地址') {
  const cleaned = String(raw || '').trim().replace(/\/+$/, '');
  let parsed;
  try { parsed = new URL(cleaned); } catch { fail(`无效的${label}`); }
  if (!['http:', 'https:'].includes(parsed.protocol) || !parsed.hostname || parsed.username || parsed.password || parsed.search || parsed.hash) {
    fail(`请检查${label}（需为合法的 http/https 地址，且不含账号密码或参数）`);
  }
  return cleaned;
}

function redrawConfig() {
  const minimaxConfigured = Boolean(process.env.MINIMAX_API_KEY);
  const comfyUrl = (process.env.MINIMAX_H3_COMFY_URL || process.env.COMFYUI_URL || '').trim();
  const comfyConfigured = Boolean(comfyUrl);
  const configuredModels = (process.env.MINIMAX_IMAGE_MODEL || 'MiniMax-H3-Image,MiniMax-H3-Image-Highspeed')
    .split(',').map(s => s.trim()).filter(Boolean);
  return {
    default_provider: !minimaxConfigured && comfyConfigured ? 'comfyui' : 'minimax_api',
    providers: {
      minimax_api: {
        configured: minimaxConfigured,
        base_url: (process.env.MINIMAX_API_BASE_URL || 'https://api.minimaxi.com/v1').trim(),
        models: [...new Set([...configuredModels, 'MiniMax-H3-Image', 'MiniMax-H3-Image-Highspeed'])],
        default_model: configuredModels[0] || 'MiniMax-H3-Image',
      },
      comfyui: {
        configured: comfyConfigured,
        url: comfyUrl,
      },
    },
    presets: [
      { key: 'faithful', label: '高保真还原原图', description: '严格锁定 3D 截图几何与机位，忠实复现原图航拍色彩、建筑材质与光照氛围' },
      { key: 'detailed', label: '强化写实细节', description: '保留几何布局的同时显著增强红砖、屋瓦、路面风化与植被真实细节' },
      { key: 'match_3d_light', label: '匹配当前3D光照', description: '沿用当前 3D 太阳方向与明暗情绪，仅迁移原图的真实材质肌理与摄影光学感' },
    ],
  };
}

async function pingComfyUI(rawUrl) {
  const target = (rawUrl || process.env.MINIMAX_H3_COMFY_URL || process.env.COMFYUI_URL || '').trim();
  if (!target) fail('请填写 ComfyUI 服务地址');
  const url = validateExternalHttpUrl(target, 'ComfyUI 服务地址');
  const exe = process.env.SCENE_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
  return await new Promise(resolve => {
    const child = spawn(exe, ['-X', 'utf8', path.join(ROOT, 'refinement', 'minimax_h3_image.py'), 'ping', url], {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '', stderr = '', expired = false;
    const timer = setTimeout(() => { expired = true; child.kill(); }, 25_000);
    child.stdout.setEncoding('utf8').on('data', chunk => { stdout += chunk; });
    child.stderr.setEncoding('utf8').on('data', chunk => { stderr = (stderr + chunk).slice(-1000); });
    child.on('error', err => {
      clearTimeout(timer);
      resolve({ ok: false, url, comfyui_version: null, has_minimax_h3_image: false, message: `连通性检测进程启动失败：${err.message}` });
    });
    child.on('close', () => {
      clearTimeout(timer);
      if (expired) {
        return resolve({ ok: false, url, comfyui_version: null, has_minimax_h3_image: false, message: 'ComfyUI 连通性检测超时' });
      }
      try {
        resolve(JSON.parse(stdout));
      } catch {
        resolve({ ok: false, url, comfyui_version: null, has_minimax_h3_image: false, message: `无法连接 ComfyUI：${stderr.trim() || '无响应'}` });
      }
    });
  });
}

async function setComfyGatewayUpstream(rawGatewayUrl, rawUpstreamUrl) {
  const gatewayBase = (rawGatewayUrl || process.env.MINIMAX_H3_COMFY_URL || process.env.COMFYUI_URL || 'http://127.0.0.1:8188').trim();
  const gatewayUrl = validateExternalHttpUrl(gatewayBase, 'ComfyUI 网关地址');
  if (!rawUpstreamUrl || !String(rawUpstreamUrl).trim()) fail('请填写正在运行的 trycloudflare 隧道地址');
  const upstreamUrl = validateExternalHttpUrl(rawUpstreamUrl, '上游隧道地址');
  const exe = process.env.SCENE_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
  return await new Promise((resolve, reject) => {
    const child = spawn(exe, ['-X', 'utf8', path.join(ROOT, 'refinement', 'minimax_h3_image.py'), 'set-upstream', gatewayUrl, upstreamUrl], {
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '', stderr = '', expired = false;
    const timer = setTimeout(() => { expired = true; child.kill(); }, 20_000);
    child.stdout.setEncoding('utf8').on('data', chunk => { stdout += chunk; });
    child.stderr.setEncoding('utf8').on('data', chunk => { stderr = (stderr + chunk).slice(-1000); });
    child.on('error', err => { clearTimeout(timer); reject(Object.assign(new Error(`热接管进程启动失败：${err.message}`), { status: 500 })); });
    child.on('close', code => {
      clearTimeout(timer);
      if (expired) return reject(Object.assign(new Error('网关热接管请求超时'), { status: 504 }));
      if (code !== 0) return reject(Object.assign(new Error(stderr.trim() || '网关热接管失败'), { status: 502 }));
      try {
        resolve({ ok: true, gateway_url: gatewayUrl, upstream_url: upstreamUrl, response: JSON.parse(stdout) });
      } catch {
        resolve({ ok: true, gateway_url: gatewayUrl, upstream_url: upstreamUrl });
      }
    });
  });
}

function redrawImageUrls(taskId, hasMask) {
  return {
    result_url: `/api/scene/redraw/${taskId}/image/result`,
    model_url: `/api/scene/redraw/${taskId}/image/model`,
    source_url: `/api/scene/redraw/${taskId}/image/source`,
    reference_url: `/api/scene/redraw/${taskId}/image/reference`,
    ...(hasMask ? { mask_url: `/api/scene/redraw/${taskId}/image/mask` } : {}),
  };
}

async function startRedraw(input) {
  const provider = !input?.provider || input.provider === 'minimax_api'
    ? 'minimax_api'
    : input.provider === 'comfyui'
      ? 'comfyui'
      : fail('不支持的重绘引擎');
  if (typeof input?.source_image !== 'string' || !input.source_image) fail('缺少 3D 视口截图');
  const runtimeApiKey = typeof input.api_key === 'string' ? input.api_key.trim() : '';
  const apiBaseUrl = typeof input.api_base_url === 'string' && input.api_base_url.trim()
    ? validateExternalHttpUrl(input.api_base_url, 'MiniMax API 地址')
    : null;
  const comfyuiUrl = typeof input.comfyui_url === 'string' && input.comfyui_url.trim()
    ? validateExternalHttpUrl(input.comfyui_url, 'ComfyUI 服务地址')
    : null;
  if (provider === 'minimax_api' && !runtimeApiKey && !process.env.MINIMAX_API_KEY) {
    fail('请先配置 MINIMAX_API_KEY（服务端 .env 或页面连接配置）', 400);
  }
  if (provider === 'comfyui' && !comfyuiUrl && !process.env.MINIMAX_H3_COMFY_URL && !process.env.COMFYUI_URL) {
    fail('请先配置 ComfyUI 服务地址（MINIMAX_H3_COMFY_URL 或页面连接配置）', 400);
  }
  const referenceImagePath = input.reference_image ? null : await resolveSceneSourcePath(input.scene_code);
  const taskId = `redraw_${Date.now()}_${randomUUID().slice(0, 8)}`;
  const directory = path.join(DATA, 'redraws', taskId);
  await mkdir(directory, { recursive: true });
  // Persist non-secret job parameters only; pass any runtime api_key exclusively via child env.
  const sanitizedInput = {
    provider,
    model: typeof input.model === 'string' ? input.model.trim() : '',
    size: typeof input.size === 'string' ? input.size.trim() : 'auto',
    style_preset: typeof input.style_preset === 'string' ? input.style_preset : 'faithful',
    prompt: typeof input.prompt === 'string' ? input.prompt.slice(0, 4000) : '',
    protect_equipment: input.protect_equipment !== false,
    api_base_url: apiBaseUrl,
    comfyui_url: comfyuiUrl,
    steps: Number.isFinite(Number(input.steps)) ? Number(input.steps) : 6,
    megapixels: Number.isFinite(Number(input.megapixels)) ? Number(input.megapixels) : 4.0,
    seed: Number.isFinite(Number(input.seed)) ? Math.trunc(Number(input.seed)) : null,
    scene_code: typeof input.scene_code === 'string' ? input.scene_code.slice(0, 120) : '',
    scene_label: typeof input.scene_label === 'string' ? input.scene_label.slice(0, 150) : '',
    source_image: input.source_image,
    reference_image: typeof input.reference_image === 'string' && input.reference_image ? input.reference_image : null,
    reference_image_path: referenceImagePath,
    environment_image: typeof input.environment_image === 'string' && input.environment_image ? input.environment_image : null,
    protection_mask: typeof input.protection_mask === 'string' && input.protection_mask ? input.protection_mask : null,
  };
  await atomicJson(path.join(directory, 'input.json'), sanitizedInput);
  const initialStatus = {
    task_id: taskId,
    status: 'running',
    provider,
    model: sanitizedInput.model || (provider === 'comfyui' ? 'MiniMax-H3-Image (ComfyUI)' : 'MiniMax-H3-Image'),
    style_preset: sanitizedInput.style_preset,
    scene_code: sanitizedInput.scene_code,
    scene_label: sanitizedInput.scene_label,
    created_at: new Date().toISOString(),
  };
  await atomicJson(path.join(directory, 'status.json'), initialStatus);
  activeRedraws.add(taskId);
  const exe = process.env.SCENE_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
  const child = spawn(exe, ['-X', 'utf8', path.join(ROOT, 'refinement', 'minimax_h3_image.py'), 'run', directory], {
    windowsHide: true,
    stdio: ['ignore', 'ignore', 'pipe'],
    env: { ...process.env, ...(runtimeApiKey ? { MINIMAX_RUNTIME_API_KEY: runtimeApiKey } : {}) },
  });
  let stderr = '', spawnError = '';
  child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4000); });
  child.on('error', error => { spawnError = error.message; });
  child.on('close', async code => {
    try {
      const done = code === 0
        ? { ...initialStatus, status: 'completed', finished_at: new Date().toISOString() }
        : { ...initialStatus, status: 'failed', finished_at: new Date().toISOString(), error_message: (spawnError || stderr || `Worker exited ${code}`).trim().slice(-1500) };
      await atomicJson(path.join(directory, 'status.json'), done);
    } catch (error) { console.error(error.message); }
    finally { activeRedraws.delete(taskId); }
  });
  return { task_id: taskId, status: 'running' };
}

async function redrawStatus(taskId) {
  const safeId = safeName(taskId);
  const directory = path.join(DATA, 'redraws', safeId);
  const state = await readJson(path.join(directory, 'status.json'), null);
  if (!state) fail('重绘任务不存在', 404);
  if (state.status === 'running' && !activeRedraws.has(safeId)) {
    return { ...state, status: 'failed', error_message: '服务已重启，请重新发起截图重绘' };
  }
  if (state.status === 'completed') {
    const meta = await readJson(path.join(directory, 'result.meta.json'), {});
    let hasMask = false;
    try { hasMask = (await stat(path.join(directory, 'protection.png'))).isFile(); } catch { /* optional mask */ }
    return { ...state, ...meta, ...redrawImageUrls(safeId, hasMask) };
  }
  return state;
}

async function listRedraws() {
  const redrawRoot = path.join(DATA, 'redraws');
  const names = await readdir(redrawRoot).catch(() => []);
  const items = [];
  for (const name of names) {
    if (!/^redraw_[a-zA-Z0-9_-]+$/.test(name)) continue;
    const directory = path.join(redrawRoot, name);
    const state = await readJson(path.join(directory, 'status.json'), null);
    if (!state || state.status !== 'completed') continue;
    const meta = await readJson(path.join(directory, 'result.meta.json'), {});
    let hasMask = false;
    try { hasMask = (await stat(path.join(directory, 'protection.png'))).isFile(); } catch { /* optional */ }
    items.push({ ...state, ...meta, ...redrawImageUrls(name, hasMask) });
  }
  return items.sort((a, b) => String(b.created_at || '').localeCompare(String(a.created_at || ''))).slice(0, 20);
}

const REDRAW_IMAGE_FILES = {
  result: 'result.png',
  model: 'result-model.png',
  source: 'source.png',
  mask: 'protection.png',
  reference: 'reference.png',
};

function safeName(value) {
  if (!/^[a-zA-Z0-9_-]{1,120}$/.test(value)) throw Object.assign(new Error('Invalid resource identifier'), { status: 400 });
  return value;
}
function fail(message, status = 400) { throw Object.assign(new Error(message), { status }); }
function json(res, data, status = 200) {
  res.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' });
  res.end(JSON.stringify(status < 400 ? { status: 0, message: 'ok', data } : { detail: data }));
}
async function readJson(file, fallback) {
  try { return JSON.parse(await readFile(file, 'utf8')); }
  catch (error) { if (error.code === 'ENOENT') return fallback; throw error; }
}
async function atomicJson(file, data) {
  const temporary = `${file}.${randomUUID()}.tmp`;
  await writeFile(temporary, JSON.stringify(data, null, 2));
  await rename(temporary, file);
}
async function body(req, limit = 160 * 1024 * 1024) {
  const chunks = []; let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > limit) fail('请求超出大小限制，请减少输入内容', 413);
    chunks.push(chunk);
  }
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); }
  catch { fail('Invalid JSON'); }
}
async function file(req, res, filename) {
  const info = await stat(filename);
  if (!info.isFile()) fail('File not found', 404);
  let start = 0, end = info.size - 1, status = 200;
  const range = req.headers.range;
  if (range) {
    const match = /^bytes=(\d+)-(\d*)$/.exec(range);
    if (!match) { res.writeHead(416, { 'Content-Range': `bytes */${info.size}` }); return res.end(); }
    start = Number(match[1]); end = match[2] ? Number(match[2]) : end;
    if (start > end || end >= info.size) { res.writeHead(416, { 'Content-Range': `bytes */${info.size}` }); return res.end(); }
    status = 206;
  }
  res.writeHead(status, { 'Content-Type': MIME[path.extname(filename)] || 'application/octet-stream', 'Content-Length': end - start + 1, 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-cache', ...(status === 206 ? { 'Content-Range': `bytes ${start}-${end}/${info.size}` } : {}) });
  if (req.method === 'HEAD') return res.end();
  const stream = createReadStream(filename, { start, end });
  stream.on('error', () => res.destroy());
  res.on('close', () => stream.destroy());
  stream.pipe(res);
}
function safeSplatFile(value) {
  if (typeof value !== 'string' || value.includes('..') || !/^[a-zA-Z0-9._-]+\.(ply|spz|splat|ksplat|sog)$/.test(value)) return null;
  return value;
}
async function library(kind) {
  const folder = kind === 'assets' ? 'equipment_3d' : 'scene_meshes';
  const directory = path.join(ROOT, 'assets', folder);
  const names = await readdir(directory);
  const result = [];
  for (const name of names.filter(n => n.endsWith('.glb')).sort()) {
    const code = path.basename(name, '.glb'); safeName(code);
    const meta = await readJson(path.join(directory, `${code}.json`), {});
    const common = { code, label: meta.label || code, has_model: true, url: `/api/${kind}/3d/${code}/file` };
    if (kind === 'assets') {
      result.push({ ...common, category: meta.category || '装备', dimensions: meta.dimensions || { length: 4, width: 2, height: 2 }, origin: meta.origin || [0, 0, 0], up_axis: meta.up_axis || 'y', yaw_offset: meta.yaw_offset || 0, default_scale: meta.default_scale || 1, thumbnail: null });
    } else {
      const scene = { ...common, kind: meta.kind || 'scene_mesh', map_id: meta.map_id || null };
      for (const key of ['source_url', 'plan_url', 'report_url', 'ground_url']) {
        if (typeof meta[key] === 'string' && meta[key].startsWith('/api/builder/')) scene[key] = meta[key];
      }
      const sourceFile = safeSourceImage(meta.source_image);
      if (!scene.source_url && sourceFile) {
        try {
          if ((await stat(path.join(ROOT, 'assets', 'scene_sources', sourceFile))).isFile()) {
            scene.source_url = `/api/scenes/3d/${code}/source`;
          }
        } catch { /* ignore missing optional source image */ }
      }
      const splatFile = safeSplatFile(meta.splat_file);
      if (splatFile) {
        try {
          if ((await stat(path.join(directory, splatFile))).isFile()) {
            scene.splat_url = `/api/scenes/3d/${code}/splat`;
            if (meta.splat_transform && typeof meta.splat_transform === 'object') scene.splat_transform = meta.splat_transform;
          }
        } catch { /* ignore missing optional splat file */ }
      }
      result.push(scene);
    }
  }
  return result;
}
function vector(raw, fallback) {
  if (!Array.isArray(raw) || raw.length !== 3 || raw.some(v => !Number.isFinite(v))) return fallback;
  return raw;
}
function normalizeSession(input, existing = {}) {
  const light = Object.fromEntries(Object.entries(lightingDefault).map(([k, v]) => [k, Number.isFinite(input.lighting?.[k]) ? input.lighting[k] : existing.lighting?.[k] ?? v]));
  return { id: existing.id || randomUUID(), title: String(input.title || existing.title || '未命名场景').slice(0, 150), lighting: light, ground_texture: input.ground_texture ?? existing.ground_texture ?? null,
    objects: (input.objects || existing.objects || []).map((o, i) => ({ id: o.id || randomUUID(), asset_code: safeName(o.asset_code), label: String(o.label || o.asset_code), kind: o.kind || 'equipment', position: vector(o.position, [0, 0, 0]), rotation: vector(o.rotation, [0, 0, 0]), scale: vector(o.scale, [1, 1, 1]), sort_order: i })),
    created_at: existing.created_at || new Date().toISOString(), updated_at: new Date().toISOString() };
}
const crcTable = Array.from({ length: 256 }, (_, n) => { for (let k = 0; k < 8; k++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1; return n >>> 0; });
function crc32(bytes) { let c = 0xffffffff; for (const b of bytes) c = crcTable[(c ^ b) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
function makeZip(entries) {
  const locals = [], central = []; let offset = 0;
  for (const [name, bytes] of entries) {
    const n = Buffer.from(name), crc = crc32(bytes), header = Buffer.alloc(30), directory = Buffer.alloc(46);
    header.writeUInt32LE(0x04034b50, 0); header.writeUInt16LE(20, 4); header.writeUInt16LE(0x800, 6); header.writeUInt32LE(crc, 14); header.writeUInt32LE(bytes.length, 18); header.writeUInt32LE(bytes.length, 22); header.writeUInt16LE(n.length, 26);
    directory.writeUInt32LE(0x02014b50, 0); directory.writeUInt16LE(20, 4); directory.writeUInt16LE(20, 6); directory.writeUInt16LE(0x800, 8); directory.writeUInt32LE(crc, 16); directory.writeUInt32LE(bytes.length, 20); directory.writeUInt32LE(bytes.length, 24); directory.writeUInt16LE(n.length, 28); directory.writeUInt32LE(offset, 42);
    locals.push(header, n, bytes); central.push(directory, n); offset += header.length + n.length + bytes.length;
  }
  const directory = Buffer.concat(central), end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0); end.writeUInt16LE(entries.length, 8); end.writeUInt16LE(entries.length, 10); end.writeUInt32LE(directory.length, 12); end.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, directory, end]);
}
function xml(value) { return String(value).replace(/[<>&"']/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' })[c]); }
async function exportFrames(input) {
  if (input.type !== 'scene_export') fail('This package supports scene_export only');
  const frames = input.params?.frames;
  if (!Array.isArray(frames) || !frames.length || frames.length > 24) fail('Expected 1–24 frames');
  const taskId = `scene_export_${Date.now()}_${randomUUID().slice(0, 8)}`;
  const entries = [];
  for (const [index, frame] of frames.entries()) {
    const png = Buffer.from(frame.image_base64 || '', 'base64');
    if (png.length < 24 || png.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a') fail('Expected PNG data');
    const width = png.readUInt32BE(16), height = png.readUInt32BE(20);
    if (width !== frame.width || height !== frame.height || width > 4096 || height > 4096) fail('PNG dimensions mismatch');
    const name = `view_${String(index + 1).padStart(2, '0')}`;
    const objects = (frame.objects || []).map(o => {
      const coords = ['xmin', 'ymin', 'xmax', 'ymax'].map((k, i) => Math.round(Math.max(0, Math.min(i % 2 ? height : width, Number(o[k])))));
      if (coords.some(v => !Number.isFinite(v))) fail('Invalid bounding box');
      return `<object><name>${xml(o.name)}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult><bndbox><xmin>${coords[0]}</xmin><ymin>${coords[1]}</ymin><xmax>${coords[2]}</xmax><ymax>${coords[3]}</ymax></bndbox></object>`;
    }).join('');
    entries.push([`images/${name}.png`, png], [`annotations/${name}.xml`, Buffer.from(`<?xml version="1.0" encoding="UTF-8"?><annotation><folder>images</folder><filename>${name}.png</filename><size><width>${width}</width><height>${height}</height><depth>3</depth></size>${objects}</annotation>`)]);
  }
  const meta = { task_id: taskId, frames: frames.length, created_at: new Date().toISOString(), url: `/api/scene/exports/${taskId}/download`, annotation_note: 'Projected 3D bounding boxes, not visibility masks; occlusion is not removed.' };
  entries.push(['manifest.json', Buffer.from(JSON.stringify(meta, null, 2))]);
  await writeFile(path.join(DATA, 'exports', `${taskId}.zip`), makeZip(entries));
  await atomicJson(path.join(DATA, 'exports', `${taskId}.json`), meta);
  return { task_id: taskId, status: 'completed' };
}

const server = http.createServer(async (req, res) => {
  res.setHeader('Content-Security-Policy', "default-src 'self'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; connect-src 'self' blob:; font-src 'self'; worker-src 'self' blob:; object-src 'none'; base-uri 'self'; frame-ancestors 'self'");
  res.setHeader('X-Content-Type-Options', 'nosniff');
  try {
    const url = new URL(req.url, 'http://localhost');
    const route = decodeURIComponent(url.pathname);
    if (req.headers.origin && req.headers.origin !== `http://${req.headers.host}` && req.headers.origin !== `https://${req.headers.host}`) fail('Cross-origin request not accepted', 403);
    if (route === '/api/health') return json(res, {
      ready: true,
      version: '1.7.0',
      mode: 'standalone-scene-demo',
      refinement: Boolean(process.env.COMFYUI_URL),
      minimax_redraw: {
        minimax_api: Boolean(process.env.MINIMAX_API_KEY),
        comfyui: Boolean(process.env.MINIMAX_H3_COMFY_URL || process.env.COMFYUI_URL),
      },
    });
    if (route === '/api/assets/3d') return json(res, { assets: await library('assets') });
    if (route.startsWith('/api/builder/')) return await builder.handle(req, res, route);
    if (route === '/api/scenes/3d') return json(res, { scenes: [...await library('scenes'), ...await builder.list()] });
    const sourceRoute = route.match(/^\/api\/scenes\/3d\/([^/]+)\/source$/);
    if (sourceRoute && (req.method === 'GET' || req.method === 'HEAD')) {
      return await file(req, res, await resolveSceneSourcePath(sourceRoute[1]));
    }
    const splatRoute = route.match(/^\/api\/scenes\/3d\/([^/]+)\/splat$/);
    if (splatRoute) {
      const code = safeName(splatRoute[1]);
      const dir = path.join(ROOT, 'assets', 'scene_meshes');
      const meta = await readJson(path.join(dir, `${code}.json`), {});
      const splatFile = safeSplatFile(meta.splat_file);
      if (!splatFile) fail('Splat not found', 404);
      return await file(req, res, path.join(dir, splatFile));
    }
    const model = route.match(/^\/api\/(assets|scenes)\/3d\/([^/]+)\/file$/);
    if (model) return await file(req, res, (model[1] === 'scenes' && builder.model(model[2])) || path.join(ROOT, 'assets', model[1] === 'assets' ? 'equipment_3d' : 'scene_meshes', `${safeName(model[2])}.glb`));
    if (route === '/api/scene/training_sets') return json(res, { training_sets: [] });
    if (route === '/api/scene/sessions' && req.method === 'GET') {
      const sessions = [];
      for (const name of (await readdir(path.join(DATA, 'sessions'))).filter(n => n.endsWith('.json'))) {
        const s = await readJson(path.join(DATA, 'sessions', name));
        sessions.push({ id: s.id, title: s.title, ground_texture: s.ground_texture, created_at: s.created_at, updated_at: s.updated_at, object_count: s.objects.length });
      }
      return json(res, { sessions: sessions.sort((a, b) => b.updated_at.localeCompare(a.updated_at)) });
    }
    if (route === '/api/scene/sessions' && req.method === 'POST') {
      const session = normalizeSession(await body(req));
      await atomicJson(path.join(DATA, 'sessions', `${session.id}.json`), session); return json(res, session);
    }
    const sessionMatch = route.match(/^\/api\/scene\/sessions\/([^/]+)$/);
    if (sessionMatch) {
      const filename = path.join(DATA, 'sessions', `${safeName(sessionMatch[1])}.json`);
      const existing = await readJson(filename, null);
      if (!existing) fail('Scene not found', 404);
      if (req.method === 'GET') return json(res, existing);
      if (req.method === 'PUT') { const s = normalizeSession(await body(req), existing); await atomicJson(filename, s); return json(res, s); }
      if (req.method === 'DELETE') { await mkdir(path.join(DATA, 'trash'), { recursive: true }); await rename(filename, path.join(DATA, 'trash', `${existing.id}-${Date.now()}.json`)); return json(res, { deleted: true }); }
    }
    if (route === '/api/generate' && req.method === 'POST') {
      const input = await body(req);
      return json(res, input.type === 'scene_refine' ? await startRefinement(input) : await exportFrames(input));
    }
    if (route === '/api/get_task_result' && req.method === 'POST') return json(res, await refinementResult(await body(req)));
    if (route === '/api/scene/exports') {
      const exports = [];
      for (const name of (await readdir(path.join(DATA, 'exports'))).filter(n => n.endsWith('.json'))) exports.push(await readJson(path.join(DATA, 'exports', name)));
      return json(res, { exports: exports.sort((a, b) => b.created_at.localeCompare(a.created_at)) });
    }
    const download = route.match(/^\/api\/scene\/exports\/([^/]+)\/download$/);
    if (download) { res.setHeader('Content-Disposition', `attachment; filename="${safeName(download[1])}.zip"`); return await file(req, res, path.join(DATA, 'exports', `${download[1]}.zip`)); }
    if (route === '/api/scene/redraw/config' && req.method === 'GET') return json(res, redrawConfig());
    if (route === '/api/scene/redraw/ping' && req.method === 'POST') return json(res, await pingComfyUI((await body(req, 16 * 1024))?.url));
    if (route === '/api/scene/redraw/gateway/set-upstream' && req.method === 'POST') {
      const input = await body(req, 16 * 1024);
      return json(res, await setComfyGatewayUpstream(input?.gateway_url, input?.upstream_url ?? input?.url));
    }
    if (route === '/api/scene/redraw' && req.method === 'POST') return json(res, await startRedraw(await body(req)), 202);
    if (route === '/api/scene/redraws' && req.method === 'GET') return json(res, { redraws: await listRedraws() });
    const redrawImg = route.match(/^\/api\/scene\/redraw\/([^/]+)\/image\/(result|model|source|mask|reference)$/);
    if (redrawImg && (req.method === 'GET' || req.method === 'HEAD')) {
      const taskId = safeName(redrawImg[1]);
      const kind = redrawImg[2];
      const imgFile = path.join(DATA, 'redraws', taskId, REDRAW_IMAGE_FILES[kind]);
      if (url.searchParams.get('download') === '1') {
        res.setHeader('Content-Disposition', `attachment; filename="${taskId}-${kind}.png"`);
      }
      return await file(req, res, imgFile);
    }
    const redrawJob = route.match(/^\/api\/scene\/redraw\/([^/]+)$/);
    if (redrawJob && req.method === 'GET') return json(res, await redrawStatus(redrawJob[1]));
    if (route.startsWith('/api/')) fail('API not found in standalone scene package', 404);
    const web = path.resolve(ROOT, 'web');
    const requested = path.resolve(web, `.${route === '/' ? '/scene-standalone.html' : route}`);
    if (!requested.startsWith(web + path.sep)) fail('Invalid path', 403);
    return await file(req, res, requested);
  } catch (error) {
    const status = error.status || (error.code === 'ENOENT' ? 404 : 500);
    if (status === 500) console.error(error);
    if (!res.headersSent) json(res, status === 500 ? 'Server error; see console' : error.message, status);
    else res.destroy();
  }
});
server.listen(PORT, HOST, () => console.log(`Scene Studio: http://127.0.0.1:${server.address().port}\nData: ${DATA}\nPress Ctrl+C to stop.`));
server.on('error', error => { console.error(error.message); process.exitCode = 1; });
