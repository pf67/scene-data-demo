import { createHash, randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { copyFile, mkdir, mkdtemp, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { orbitService } from './orbit-service.mjs';
import { orbitTools } from './orbit-config.mjs';

export async function createBuilder({ root, data, file, json, fail, body, readJson }) {
  const home = path.join(root, 'builder');
  const store = path.join(data, 'builder');
  const scenes = path.join(store, 'scenes');
  const sources = path.join(store, 'sources');
  const jobs = path.join(store, 'ground-jobs');
  const activeJobs = new Set();
  for (const dir of [scenes, sources, jobs]) await mkdir(dir, { recursive: true });
  const schema = JSON.parse(await readFile(path.join(home, 'plan.schema.json'), 'utf8'));
  const maxSupplements = schema.properties.supplements.maxItems;
  const catalog = await readJson(path.join(home,'sample-views','manifest.json'));
  const samples = new Map(catalog.views.map(v => [v.source,v]));
  const materialSources = new Set((catalog.materialReferences || []).map(v => v.source));
  const planner = Boolean(process.env.SCENE_PLANNER_URL && process.env.SCENE_PLANNER_MODEL);
  const sceneId = /^BUILD_[a-f0-9]{32}$/;
  const sourceId = /^src_[a-f0-9]{64}$/;
  let busy = 0;

  async function scratch(operation) {
    if (busy >= 3) fail('构建工具正在处理其他请求，请稍后再试', 503);
    busy++;
    let dir;
    try {
      dir = await mkdtemp(path.join(store, '.work-'));
      return await operation(dir);
    } finally {
      busy--;
      if (dir) await rm(dir, { recursive: true, force: true });
    }
  }

  function worker(args, maxTime = 120_000) {
    return new Promise((resolve, reject) => {
      const exe = process.env.SCENE_PYTHON || (process.platform === 'win32' ? 'python' : 'python3');
      const proc = spawn(exe, ['-X', 'utf8', path.join(home, 'worker.py'), ...args],
        { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
      let output = '', error = '', expired = false;
      const timeout = setTimeout(() => { expired = true; proc.kill(); }, maxTime);
      proc.stdout.setEncoding('utf8').on('data', buf => { output += buf; });
      proc.stderr.setEncoding('utf8').on('data', buf => { error = (error + buf).slice(-4000); });
      proc.on('error', () => reject(Object.assign(new Error('建模 Python 无法启动，请检查 SCENE_PYTHON'), { status: 503 })));
      proc.on('close', code => {
        clearTimeout(timeout);
        if (expired) return reject(Object.assign(new Error('建模超时，请减少图像或建筑数量后重试'), { status: 504 }));
        if (code === 2) return reject(Object.assign(new Error(error.trim().slice(-800)), { status: 422 }));
        if (code !== 0) {
          console.error('Scene builder:', error.trim());
          const redraw = args[0] === 'ground-edit' && args.some(a => a.startsWith('--prompt='));
          const message = args[0] === 'orbit-view' ? '旋转机位处理失败，请检查所选模型服务和场景服务日志' : redraw ? 'Qwen 地面重绘失败，请检查模型服务、工作流依赖和部署日志' : '建模工具运行失败，请检查服务端 Python 依赖和日志';
          return reject(Object.assign(new Error(message), { status: 503 }));
        }
        try { resolve(JSON.parse(output)); } catch { reject(new Error('Invalid worker result')); }
      });
    });
  }

  function photoPath(id) {
    if (id === 'village') return path.join(root, 'assets', 'scene_sources', 'village_road_near.png');
    if (id === 'village-depth-ground') return path.join(home, 'village-ground.png');
    if (samples.has(id)) return path.join(home,'sample-views',id+'.jpg');
    if (typeof id === 'string' && /^orb_[a-f0-9]{32}$/.test(id)) return path.join(store,'orbit-jobs',id.slice(4),'result','view.jpg');
    if (typeof id !== 'string' || !sourceId.test(id)) fail('无效的参考图编号', 422);
    return path.join(sources, id, 'photo.jpg');
  }
  const photoUrl = id => '/api/builder/sources/' + id + '/image';
  async function sourceFiles(plan, includeMasks = false) {
    if (!Array.isArray(plan?.views) || plan.views.length < 1 || plan.views.length > 6) fail('请选择 1–6 张参考图', 422);
    const files = new Map(), views = new Set();
    for (const view of plan.views) {
      const id = view?.source;
      if (views.has(id)) fail('每个视图需要不同的参考图', 422);
      views.add(id);
      if (typeof id === 'string' && (id.startsWith('orb_') || id.startsWith('village-orbit-'))) fail('生成图请放入 supplements，原始相机需使用实拍图',422);
      for (const key of includeMasks && view?.groundMask ? [id, view.groundMask] : [id]) {
        const name = photoPath(key);
        try { await stat(name); } catch (e) { if (e.code !== 'ENOENT') throw e; fail('参考图或地面掩膜不存在，请先上传图片', 422); }
        files.set(key, name);
      }
    }
    if (includeMasks && plan.supplements !== undefined) {
      if (!Array.isArray(plan.supplements) || plan.supplements.length>maxSupplements) fail(`最多可选用 ${maxSupplements} 张补充视图`,422);
      for (const spec of plan.supplements) {
        const name=photoPath(spec?.source);
        if (files.has(spec.source) || !views.has(spec.reference)) fail('补充视图重复或缺少对应的原始照片',422);
        try { await stat(name); } catch (e) { if (e.code!=='ENOENT') throw e; fail('补充视图不存在，请先生成或导入',422); }
        files.set(spec.source,name);
      }
    }
    return files;
  }

  async function imageBody(req) {
    const chunks = [];
    let size = 0;
    for await (const chunk of req) {
      size += chunk.length;
      if (size > 16 * 1024 * 1024) fail('每张图像最多 16 MB', 413);
      chunks.push(chunk);
    }
    return Buffer.concat(chunks);
  }

  async function upload(req) {
    return scratch(async dir => {
      const bytes = await imageBody(req);
      const source = 'src_' + createHash('sha256').update(bytes).digest('hex');
      const destination = path.join(sources, source);
      const cached = await readJson(path.join(destination, 'info.json'), null);
      if (cached) return cached;
      const input = path.join(dir, 'input');
      await writeFile(input, bytes);
      const info = { source, url: photoUrl(source), ...await worker(['photo', input, path.join(dir, 'photo.jpg')]) };
      await rm(input);
      await writeFile(path.join(dir, 'info.json'), JSON.stringify(info));
      try { await rename(dir, destination); }
      catch (e) { if (!(await readJson(path.join(destination, 'info.json'), null))) throw e; }
      return info;
    });
  }

  async function construct(action, plan) {
    const files = await sourceFiles(plan, true);
    return scratch(async dir => {
      const input = path.join(dir, 'plan.json');
      await writeFile(input, JSON.stringify(plan));
      const args = [action, '--plan', input];
      for (const [id, name] of files) args.push('--source', id, name);
      const timeout = 120_000 + 25_000*(plan.supplements?.length || 0);
      if (action === 'validate') return worker(args,timeout);
      const code = 'BUILD_' + randomUUID().replaceAll('-', '');
      args.push('--out', path.join(dir, 'scene.glb'), '--report', path.join(dir, 'report.json'), '--ground-dir', path.join(dir, 'ground'));
      const report = await worker(args,timeout);
      const scene = await publish(dir, code, plan, plan.name);
      return { scene, report };
    });
  }

  async function publish(dir, code, plan, label) {
    const base = '/api/builder/scenes/' + code;
    const scene = { code, label, kind: 'calibrated_building_scene', has_model: true, map_id: null,
      url: '/api/scenes/3d/' + code + '/file', source_url: photoUrl(plan.views[0].source),
      plan_url: base + '/plan', report_url: base + '/report', ground_url: base + '/ground', created_at: new Date().toISOString() };
    await writeFile(path.join(dir, 'scene.json'), JSON.stringify(scene));
    await rename(dir, path.join(scenes, code));
    return scene;
  }

  function groundPaths(code) {
    if (code === 'VILLAGE_CALIBRATED') return {
      ground: path.join(home, 'example-ground'), plan: path.join(home, 'example.json'), report: path.join(home, 'example-report.json'),
      model: path.join(root, 'assets', 'scene_meshes', code + '.glb') };
    if (!sceneId.test(code)) fail('此场景没有地面构建资料', 404);
    const dir = path.join(scenes, code);
    return { ground: path.join(dir, 'ground'), plan: path.join(dir, 'plan.json'), report: path.join(dir, 'report.json'), model: path.join(dir, 'scene.glb') };
  }

  async function editGround(code, newCode, input, usingModel) {
    const paths = groundPaths(code);
    const report = await readJson(paths.report);
    if (!report?.mesh?.ground) fail('此版本缺少地面覆盖图，请从构建方案重新生成场景', 422);
    return scratch(async dir => {
      const temporary = path.join(dir, 'input');
      const args = ['ground-edit', '--model', paths.model, '--ground-dir', paths.ground, '--out', path.join(dir, 'scene.glb'), '--out-ground', path.join(dir, 'ground')];
      if (usingModel) args.push('--prompt=' + input);
      else { await writeFile(temporary, await imageBody(input)); args.push('--image', temporary); }
      const result = await worker(args, usingModel ? 2_100_000 : 120_000);
      await rm(usingModel ? path.join(dir, 'comfy-task.json') : temporary, { force: true });
      Object.assign(report.mesh.ground, result, { parentScene: code });
      await copyFile(paths.plan, path.join(dir, 'plan.json'));
      await writeFile(path.join(dir, 'report.json'), JSON.stringify(report));
      const plan = await readJson(paths.plan);
      const scene = await publish(dir, newCode, plan, plan.name + (usingModel ? ' · 地面重绘' : ' · 地面补图'));
      return { scene, report };
    });
  }

  async function startGroundJob(code, input) {
    if (!process.env.COMFYUI_URL) fail('请先配置 Qwen-Image-2.1 服务，或使用离线补全和导入补图', 503);
    if (typeof input?.prompt !== 'string' || input.prompt.length > 2000) fail('重绘说明最多 2000 字', 422);
    const base = await readJson(groundPaths(code).report);
    if (!base?.mesh?.ground) fail('此版本缺少地面覆盖图，请重新构建场景', 422);
    if (busy >= 3) fail('构建工具正在处理其他请求，请稍后再试', 503);
    const id = randomUUID().replaceAll('-', '');
    const jobFile = path.join(jobs, id + '.json');
    const running = { status: 'running', taskId: id };
    await writeFile(jobFile, JSON.stringify(running));
    activeJobs.add(id);
    // Publishing the scene atomically also publishes the successful result under the same ID.
    editGround(code, 'BUILD_' + id, input.prompt, true).catch(async e => {
      const tmp = jobFile + '.tmp';
      await writeFile(tmp, JSON.stringify({ status: 'failed', taskId: id, message: e.message }));
      await rename(tmp, jobFile);
    }).catch(e => console.error('Ground job:', e.message)).finally(() => activeJobs.delete(id));
    return running;
  }

  async function groundJob(id) {
    const dir = path.join(scenes, 'BUILD_' + id);
    const status = await readJson(path.join(jobs, id + '.json'));
    if (!status) fail('地面重绘任务不存在', 404);
    const scene = await readJson(path.join(dir, 'scene.json'));
    if (scene) return { status: 'completed', taskId: id, scene, report: await readJson(path.join(dir, 'report.json')) };
    if (status.status === 'running' && !activeJobs.has(id)) return { status: 'failed', taskId: id, message: '服务已重启，请重新提交地面重绘' };
    return status;
  }

  async function propose(input) {
    if (!planner) fail('尚未配置视觉模型，请设置 SCENE_PLANNER_URL 和 SCENE_PLANNER_MODEL', 503);
    const { plan, instruction } = input || {};
    if (typeof instruction !== 'string' || !instruction.trim() || instruction.length > 3000) fail('请提供 1–3000 字的建模要求');
    const files = await sourceFiles(plan);
    return scratch(async dir => {
      const parts = [{ type: 'text', text: instruction }];
      for (const [source, file] of files) {
        const mime = source === 'village' ? 'image/png' : 'image/jpeg';
        parts.push({ type: 'text', text: 'Image source ID: ' + source },
          { type: 'image_url', image_url: { url: 'data:' + mime + ';base64,' + (await readFile(file)).toString('base64') } });
      }
      const system = 'Draft editable building shells for the supplied photographs. Return a JSON object with a buildings array only. '
        + 'Use normalized [x/width,y/height] image coordinates, top-left origin. Trace convex horizontal eave polygons in perimeter order. '
        + 'For the same building across views use exactly the same physical corner 1 and winding. Single-view heights are estimates. '
        + 'Do not invent surveyed points. Use only supplied source IDs. Building schema and definitions: '
        + JSON.stringify({ items: schema.properties.buildings, definitions: schema.definitions });
      const headers = { 'Content-Type': 'application/json' };
      if (process.env.SCENE_PLANNER_KEY) headers.Authorization = 'Bearer ' + process.env.SCENE_PLANNER_KEY;
      let result;
      try {
        const res = await fetch(process.env.SCENE_PLANNER_URL, { method: 'POST', headers, redirect: 'error',
          signal: AbortSignal.timeout(90_000), body: JSON.stringify({ model: process.env.SCENE_PLANNER_MODEL,
            messages: [{ role: 'system', content: system }, { role: 'user', content: parts }],
            response_format: { type: 'json_object' } }) });
        if (!res.ok) fail('视觉模型返回 HTTP ' + res.status + '，请检查服务端配置', 502);
        result = await res.json();
      } catch (e) { if (e.status) throw e; fail('视觉模型无法访问或响应超时', 502); }
      let buildings;
      try { buildings = JSON.parse(result.choices[0].message.content).buildings; }
      catch { fail('视觉模型返回的内容不是建筑 JSON 草案', 422); }
      if (!Array.isArray(buildings) || !buildings.length || buildings.length > 150) fail('视觉模型需提供 1–150 栋建筑', 422);
      for (const b of buildings) {
        if (!b || typeof b !== 'object' || Array.isArray(b)) fail('模型草案中的建筑无效', 422);
        b.heightSource = 'estimated';
      }
      const draft = { ...plan, buildings };
      const name = path.join(dir, 'draft.json');
      await writeFile(name, JSON.stringify(draft));
      await worker(['draft', '--plan', name]);
      return { plan: draft };
    });
  }

  async function readPlan(code) {
    if (typeof code!=='string') fail('无效的场景编号',422);
    const plan=await readJson(groundPaths(code).plan);
    if (!plan) fail('此场景没有可编辑的建模方案',404);
    return plan;
  }
  const orbits = await orbitService({store,scratch,worker,sourceFiles,photoPath,body,file,json,fail,readJson,samples,readPlan,construct,maxSupplements});
  const actions = [
    ['validate_scene_plan','/api/builder/validate',p=>construct('validate',p),200],
    ['build_scene','/api/builder/build',p=>construct('build',p),201],
    ['list_orbit_providers','/api/builder/orbit-providers',()=>orbits.capabilities(),200],
    ['create_orbit_view','/api/builder/orbit-views',p=>orbits.start(p),202],
    ['get_orbit_job','/api/builder/orbit-jobs/{taskId}',p=>orbits.status(p.taskId),200],
    ['build_with_orbit_views','/api/builder/orbit-build',p=>orbits.build(p),201]
  ];
  const functions = [
    ['validate_scene_plan','Fit cameras and return geometric quality measurements.'],
    ['build_scene','Build, persist and register the calibrated textured GLB.']
  ].map(([name,description])=>({type:'function',function:{name,description,parameters:schema}})).concat(orbitTools(schema));
  return {
    async list() {
      const ids = (await readdir(scenes)).filter(id => sceneId.test(id));
      const items = await Promise.all(ids.map(id => readJson(path.join(scenes, id, 'scene.json'))));
      return items.sort((a, b) => b.created_at.localeCompare(a.created_at));
    },
    async sourceFile(code) {
      if (!sceneId.test(code)) return null;
      const scene = await readJson(path.join(scenes, code, 'scene.json'), null);
      const match = typeof scene?.source_url === 'string' && scene.source_url.match(/^\/api\/builder\/sources\/([^/]+)\/image$/);
      if (!match) return null;
      try {
        const candidate = materialSources.has(match[1]) ? path.join(home, 'sample-views', match[1] + '.jpg') : photoPath(match[1]);
        await stat(candidate);
        return candidate;
      } catch {
        return null;
      }
    },
    model(code) { return sceneId.test(code) ? path.join(scenes, code, 'scene.glb') : null; },
    async handle(req, res, route) {
      const get = req.method === 'GET' || req.method === 'HEAD';
      const post = req.method === 'POST';
      if (get && ['/api/builder/schema', '/api/builder/example'].includes(route))
        return file(req, res, path.join(home, route.endsWith('/schema') ? 'plan.schema.json' : 'example.json'));
      if (get && route === '/api/builder/example-report') return file(req, res, path.join(home, 'example-report.json'));
      if (get && route === '/api/builder/tools') {
        return json(res, { version: 2, planner, orbitViews: orbits.configured,
          orbitProvider: orbits.defaultProvider,
          groundRedraw: Boolean(process.env.COMFYUI_URL), schema_url: '/api/builder/schema',
          orbit: { options:'/api/builder/orbit-options', providers:'/api/builder/orbit-providers', create:'/api/builder/orbit-views', status:'/api/builder/orbit-jobs/{taskId}', apply:'/api/builder/orbit-build', build:'/api/builder/build', planKey:'supplements' },
          ground: { base: 'scene.ground_url', maps: ['texture', 'mask', 'coverage'], editable_mask: 'white',
            import: { method: 'POST', suffix: '/apply', body: 'PNG or JPEG bytes' },
            redraw: { method: 'POST', suffix: '/redraw', body: { prompt: 'string' }, status: '/api/builder/ground-jobs/{taskId}' } },
          call:'/api/builder/tools/call', functions,
          endpoints: Object.fromEntries(actions.map(([name,route])=>[name,route])) });
      }
      if (post && route==='/api/builder/tools/call') {
        const input=await body(req,2*1024*1024);
        const action=actions.find(([name])=>name===input?.name);
        if (!action) fail('未知的建模工具',404);
        const args=input.arguments ?? {};
        if (typeof args!=='object' || Array.isArray(args)) fail('工具参数须为 JSON 对象',422);
        return json(res,await action[2](args),action[3]);
      }
      if (post && route === '/api/builder/sources') return json(res, await upload(req), 201);
      const sampleView = route.match(/^\/api\/builder\/sample-views\/([a-z0-9-]+)\/(guide|overlay)$/);
      if (get && sampleView && samples.has(sampleView[1])) return file(req, res, path.join(home, 'sample-views', sampleView[1]+'-'+sampleView[2]+'.jpg'));
      if (route.startsWith('/api/builder/orbit-') && await orbits.handle(req,res,route)) return;
      const image = route.match(/^\/api\/builder\/sources\/([^/]+)\/image$/);
      if (get && image) {
        const source = image[1];
        const localPath = materialSources.has(source) ? path.join(home, 'sample-views', source + '.jpg') : photoPath(source);
        return file(req, res, localPath);
      }
      if (post && ['/api/builder/build', '/api/builder/validate'].includes(route))
        return json(res, await construct(route.endsWith('/build') ? 'build' : 'validate', await body(req, 2 * 1024 * 1024)), route.endsWith('/build') ? 201 : 200);
      if (post && route === '/api/builder/propose') return json(res, await propose(await body(req, 2 * 1024 * 1024)));
      const ground = route.match(/^\/api\/builder\/scenes\/([^/]+)\/ground\/(texture|mask|coverage|apply|redraw)$/);
      if (ground) {
        const [, code, action] = ground;
        if (get && ['texture', 'mask', 'coverage'].includes(action)) return file(req, res, path.join(groundPaths(code).ground, action + '.png'));
        if (post && action === 'apply') return json(res, await editGround(code, 'BUILD_' + randomUUID().replaceAll('-', ''), req, false), 201);
        if (post && action === 'redraw') return json(res, await startGroundJob(code, await body(req, 12_000)), 202);
      }
      const job = route.match(/^\/api\/builder\/ground-jobs\/([a-f0-9]{32})$/);
      if (get && job) return json(res, await groundJob(job[1]));
      const saved = route.match(/^\/api\/builder\/scenes\/([^/]+)\/(plan|report)$/);
      if (get && saved && sceneId.test(saved[1])) return file(req, res, path.join(scenes, saved[1], saved[2] + '.json'));
      fail('构建工具接口不存在', 404);
    }
  };
}
