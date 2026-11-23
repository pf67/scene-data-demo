import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
async function launch(data, extra = {}) {
  const child = spawn(process.execPath, ['server.mjs'], { cwd: root, windowsHide: true, env: { ...process.env,
    SCENE_DATA_DIR: data, PORT: '0', HOST: '127.0.0.1', SCENE_PLANNER_URL: '', SCENE_PLANNER_MODEL: '', COMFYUI_URL: '', QMA_API_URL:'',
    key:'', OPENAI_API_KEY:'', ORBIT_IMAGE_API_KEY:'', ORBIT_IMAGE_BASE_URL:'', ORBIT_IMAGE_MODELS:'', ORBIT_IMAGE_SIZE:'',
    ORBIT_IMAGE_STREAM:'', ORBIT_DEFAULT_PROVIDER:'', ...extra } });
  const url = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => { child.kill(); reject(new Error('Server startup timed out')); }, 12000);
    child.once('error', reject);
    child.stdout.on('data', chunk => {
      const match = chunk.toString().match(/http:\/\/127\.0\.0\.1:\d+/);
      if (match) { clearTimeout(timer); resolve(match[0]); }
    });
  });
  return { url, close: async () => {
    if (child.exitCode !== null || child.signalCode) return;
    const closed = once(child, 'close'); child.kill(); await closed;
  } };
}
const post = (base, route, payload) => fetch(base + route, { method: 'POST',
  headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });

test('uploaded source → validated plan → persistent GLB, plan and report across restart', async t => {
  const data = await mkdtemp(path.join(os.tmpdir(), 'scene-builder-'));
  let server = await launch(data);
  t.after(async () => { await server.close(); await rm(data, { recursive: true, force: true }); });
  const url = server.url;
  const tools = (await (await fetch(url + '/api/builder/tools')).json()).data;
  assert.equal(tools.planner, false);
  const offline = (await (await fetch(url + tools.orbit.options)).json()).data;
  assert.equal(offline.ready,false);
  assert.equal(offline.provider,null);
  assert.equal(tools.functions[1].function.parameters.title, 'Scene construction plan v1');
  const plan = await (await fetch(url + '/api/builder/example')).json();
  const bad = structuredClone(plan);
  bad.views[0].controls = [];
  assert.equal((await post(url, tools.endpoints.build_scene, bad)).status, 422);
  assert.equal((await post(url, '/api/builder/propose', { plan, instruction: 'draw' })).status, 503);
  const bytes = await readFile(path.join(root, 'assets/scene_sources/village_road_near.png'));
  const upload = async () => {
    const res = await fetch(url + '/api/builder/sources', { method: 'POST', body: bytes });
    assert.equal(res.status, 201);
    return (await res.json()).data;
  };
  const [source, repeated] = await Promise.all([upload(), upload()]);
  assert.deepEqual(source, repeated);
  assert.equal(source.width, 1536);
  assert.equal((await fetch(url + source.url)).headers.get('content-type'), 'image/jpeg');
  assert.equal((await fetch(url + '/api/builder/sources', { method: 'POST', body: 'fake image' })).status, 422);
  delete plan.supplements; // This fixture covers a single newly uploaded photograph.
  plan.views[0].source = source.source;
  for (const b of plan.buildings) b.observations[0].source = source.source;
  const check = await post(url, tools.endpoints.validate_scene_plan, plan);
  assert.equal(check.status, 200);
  assert.equal((await check.json()).data.buildings.length, 64);
  const result = await post(url, tools.endpoints.build_scene, plan);
  assert.equal(result.status, 201);
  const { scene, report } = (await result.json()).data;
  assert.ok(report.mesh.photoFaces > 64);
  assert.ok(report.mesh.inferredFaces > 0);
  await server.close();
  server = await launch(data);
  const list = (await (await fetch(server.url + '/api/scenes/3d')).json()).data.scenes;
  assert.equal(list.filter(s => s.code.startsWith('BUILD_')).length, 1);
  const sample = list.find(s => s.code === 'VILLAGE_CALIBRATED');
  assert.equal((await (await fetch(server.url + sample.report_url)).json()).buildings.length, 64);
  for (const s of [scene, sample]) {
    for (const name of ['mask','coverage','texture']) {
      const r = await fetch(server.url + s.ground_url + '/' + name, { method:'HEAD' });
      assert.equal(r.status, 200); assert.equal(r.headers.get('content-type'), 'image/png');
    }
  }
  assert.equal((await post(server.url, scene.ground_url + '/redraw', { prompt:'fields' })).status, 503);
  assert.equal(list.find(s => s.code === scene.code).source_url, source.url);
  assert.deepEqual(await (await fetch(server.url + scene.plan_url)).json(), plan);
  assert.deepEqual(await (await fetch(server.url + scene.report_url)).json(), report);
  const glb = await fetch(server.url + scene.url, { headers: { Range: 'bytes=0-11' } });
  assert.equal(glb.status, 206);
  assert.equal(Buffer.from(await glb.arrayBuffer()).toString('ascii', 0, 4), 'glTF');
});

test('dual-image orbit job registers a new camera, rebuilds offline and survives restart', async t => {
  const candidate = await readFile(path.join(root,'builder/sample-views/village-orbit-right.jpg'));
  const uploads = [];
  let graph;
  const model = http.createServer(async (req,res) => {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    const body = Buffer.concat(chunks);
    if (req.url.startsWith('/view?')) return res.end(candidate);
    res.setHeader('Content-Type','application/json');
    if (req.url === '/upload/image') {
      const form = await new Response(body,{headers:{'Content-Type':req.headers['content-type']}}).formData();
      const bytes = Buffer.from(await form.get('image').arrayBuffer());
      uploads.push([bytes.readUInt32BE(16),bytes.readUInt32BE(20)]);
      return res.end(JSON.stringify({name:'input-'+uploads.length+'.png'}));
    }
    if (req.url === '/prompt') {
      graph = JSON.parse(Buffer.from(body)).prompt;
      return res.end(JSON.stringify({prompt_id:'orbit-qa'}));
    }
    res.end(JSON.stringify({'orbit-qa':{outputs:{9:{images:[{filename:'candidate.jpg'}]}}}}));
  });
  model.listen(0,'127.0.0.1'); await once(model,'listening');
  const data = await mkdtemp(path.join(os.tmpdir(),'scene-orbit-'));
  let server = await launch(data,{COMFYUI_URL:'http://127.0.0.1:'+model.address().port});
  t.after(async () => { await server.close(); model.close(); await rm(data,{recursive:true,force:true}); });
  const plan = await (await fetch(server.url+'/api/builder/example')).json();
  assert.deepEqual((await (await fetch(server.url+'/api/builder/orbit-options')).json()).data,{provider:'comfyui',ready:true});
  const payload = {plan, reference:'village',angle:38,seed:528138};
  assert.equal((await post(server.url,'/api/builder/orbit-views',{...payload,angle:2})).status,422);
  const response = await post(server.url,'/api/builder/orbit-views',payload);
  assert.equal(response.status,202);
  const id = (await response.json()).data.taskId, route = '/api/builder/orbit-jobs/'+id;
  let state;
  for (let i=0;i<200;i++) {
    state = (await (await fetch(server.url+route)).json()).data;
    if (state.status!=='running') break;
    await new Promise(resolve=>setTimeout(resolve,250));
  }
  assert.equal(state.status,'completed',state.message);
  assert.equal(state.quality.status,'aligned');
  assert.equal(state.spec.provenance,'qwen-image-2.1');
  assert.ok(Math.abs(state.quality.actualAngle-38)<5);
  assert.equal(state.generation.seed,528138);
  assert.deepEqual(uploads,[[1152,768],[1536,1024]]);
  assert.equal(graph[1].inputs.image,'input-1.png');
  assert.equal(graph[10].inputs.image,'input-2.png');
  assert.deepEqual(graph[6].inputs['images.image_1'],['1',0]);
  assert.deepEqual(graph[6].inputs['images.image_2'],['10',0]);
  assert.equal(graph[7].inputs.seed,528138);
  for (const url of [state.imageUrl,state.guideUrl,state.overlayUrl]) {
    assert.equal((await fetch(server.url+url,{method:'HEAD'})).headers.get('content-type'),'image/jpeg');
  }
  await server.close();
  server = await launch(data);
  assert.equal((await (await fetch(server.url+route)).json()).data.quality.status,'aligned');
  assert.equal((await post(server.url,'/api/builder/orbit-views',payload)).status,503);
  const result = await post(server.url,'/api/builder/build',{...plan,supplements:[state.spec]});
  assert.equal(result.status,201,await result.clone().text());
  const {report} = (await result.json()).data;
  assert.equal(report.views.length,1);
  assert.equal(report.buildings.length,64);
  assert.ok(report.mesh.supplements.buildingFaces>0 && report.mesh.ground.syntheticTexels>0);
});

test('QMA options recover live and orbit jobs carry selected or suggested strength through the worker', async t => {
  const bytes = await readFile(path.join(root,'builder/sample-views/village-orbit-qma.jpg'));
  const received = [];
  let online = false;
  const azimuths = ['front-right quarter view','back-right quarter view'];
  const model = http.createServer(async (req,res) => {
    if (req.url === '/v1/jobs/view/images/0') return res.end(bytes);
    res.setHeader('Content-Type','application/json');
    if (req.url === '/v1/options') {
      if (!online) res.statusCode=503;
      return res.end(JSON.stringify(online ? {default_strength:.37,azimuths} : {error:'offline'}));
    }
    if (req.method==='POST' && req.url==='/v1/jobs') {
      const parts=[];
      for await(const part of req) parts.push(part);
      const {strength,azimuth}=JSON.parse(Buffer.concat(parts));
      received.push({strength,azimuth});
      return res.end(JSON.stringify({job_id:'view'}));
    }
    if (req.url==='/v1/jobs/view') return res.end(JSON.stringify({status:'succeeded',images:[{}]}));
    res.writeHead(404); res.end();
  });
  model.listen(0,'127.0.0.1'); await once(model,'listening');
  const data=await mkdtemp(path.join(os.tmpdir(),'scene-qma-options-'));
  const server=await launch(data,{QMA_API_URL:'http://127.0.0.1:'+model.address().port});
  t.after(async()=>{await server.close(); model.close(); await rm(data,{recursive:true,force:true});});
  const result=async route=>(await (await fetch(server.url+route)).json()).data;
  const tools=await result('/api/builder/tools');
  assert.equal(tools.orbitProvider,'qma');
  const offline=await result(tools.orbit.options);
  assert.equal(offline.ready,false);
  assert.match(offline.message,/QMA/);
  online=true;
  const ready=await result(tools.orbit.options);
  assert.equal(ready.ready,true);
  assert.equal(ready.defaultStrength,.37);
  assert.deepEqual(ready.directions.map(d=>[d.angle,d.suggestedStrength]),[[45,.37],[135,.77]]);
  const plan=await (await fetch(server.url+'/api/builder/example')).json();
  const input={plan,reference:'village',angle:135};
  for (const strength of [null,'',-0.01,1.01]) {
    assert.equal((await post(server.url,tools.orbit.create,{...input,strength})).status,422);
  }
  for (const override of [undefined,.873]) {
    const res=await post(server.url,tools.orbit.create,{...input,strength:override});
    assert.equal(res.status,202);
    const {taskId}= (await res.json()).data;
    let state;
    for(let i=0;i<250;i++){
      state=await result('/api/builder/orbit-jobs/'+taskId);
      if(state.status!=='running')break;
      await new Promise(resolve=>setTimeout(resolve,250));
    }
    assert.equal(state.status,'completed',state.message);
    assert.equal(state.generation.strength,override??.77);
  }
  assert.deepEqual(received,[{strength:.77,azimuth:azimuths[1]},{strength:.873,azimuth:azimuths[1]}]);
});

test('tool catalog selects an explicit image API, registers its camera and rebuilds offline', async t => {
  const bytes=await readFile(path.join(root,'builder/sample-views/village-orbit-right.jpg'));
  const received=[];
  const model=http.createServer(async (req,res)=>{
    res.setHeader('Content-Type','application/json');
    if (req.url==='/v1/options') return res.end(JSON.stringify({default_strength:.35,azimuths:['right side view']}));
    if (req.url!=='/v1/images/edits' || req.method!=='POST') {res.writeHead(404);return res.end();}
    const chunks=[]; for await (const chunk of req) chunks.push(chunk);
    const form=await new Response(Buffer.concat(chunks),{headers:{'Content-Type':req.headers['content-type']}}).formData();
    const images=await Promise.all([...form.getAll('image[]'),...form.getAll('image')].map(async file=>{
      const png=Buffer.from(await file.arrayBuffer());
      return [file.name,png.readUInt32BE(16),png.readUInt32BE(20)];
    }));
    received.push({model:form.get('model'),quality:form.get('quality'),size:form.get('size'),prompt:form.get('prompt'),
      auth:req.headers.authorization,images});
    res.end(JSON.stringify({data:[{b64_json:bytes.toString('base64')}]}));
  });
  model.listen(0,'127.0.0.1');await once(model,'listening');
  const data=await mkdtemp(path.join(os.tmpdir(),'scene-image-api-'));
  const endpoint='http://127.0.0.1:'+model.address().port;
  let server=await launch(data,{QMA_API_URL:endpoint,ORBIT_DEFAULT_PROVIDER:'qma',ORBIT_IMAGE_BASE_URL:endpoint+'/v1',
    ORBIT_IMAGE_MODELS:'test-first,test-facades',ORBIT_IMAGE_API_KEY:'fixture-token'});
  t.after(async()=>{await server.close();model.close();await rm(data,{recursive:true,force:true});});
  const invoke=(name,args={})=>post(server.url,'/api/builder/tools/call',{name,arguments:args});
  const manifest=(await (await fetch(server.url+'/api/builder/tools')).json()).data;
  assert.ok(manifest.functions.some(t=>t.function.name==='build_with_orbit_views'));
  const catalog=(await (await invoke('list_orbit_providers')).json()).data;
  assert.equal(catalog.defaultProvider,'qma');
  assert.deepEqual(catalog.providers.find(p=>p.id==='openai').models,['test-first','test-facades']);
  assert.ok(!JSON.stringify([manifest,catalog]).includes('fixture-token'));
  const input={sceneCode:'VILLAGE_CALIBRATED',angle:38,provider:'openai',model:'test-facades',quality:'medium',guidance:'layout',instruction:'Retain courtyard stone.'};
  for (const invalid of [{model:'unconfigured'},{provider:'comfyui'},{angle:5},{strength:.5},{image:''},{points:[]}]) {
    assert.ok((await invoke('create_orbit_view',{...input,...invalid})).status>=400);
  }
  assert.equal(received.length,0);
  const response=await invoke('create_orbit_view',input);
  assert.equal(response.status,202);const {taskId}=(await response.json()).data;
  async function finish(id) {
    for (let i=0;i<350;i++){
      const state=(await (await invoke('get_orbit_job',{taskId:id})).json()).data;
      if (state.status!=='running')return state;
      await new Promise(resolve=>setTimeout(resolve,250));
    }
    throw new Error('Image job timed out');
  }
  const job=await finish(taskId);
  assert.equal(job.status,'completed',job.message); assert.equal(job.spec.provider,'openai');
  assert.equal(job.spec.model,'test-facades'); assert.equal(job.spec.provenance,'image-api');
  assert.equal(job.quality.status,'aligned',job.quality.message);
  assert.ok(Math.abs(job.quality.actualAngle-38)<6);
  assert.equal(received.length,1);
  assert.deepEqual(received[0].images,[['target-layout.png',1152,768],['reference.png',1536,1024]]);
  assert.equal(received[0].auth,'Bearer fixture-token');
  assert.equal(received[0].quality,'medium'); assert.equal(received[0].model,input.model);
  assert.equal(received[0].size,'1248x832'); assert.match(received[0].prompt,/courtyard stone/);
  await server.close();server=await launch(data);
  const changed=await (await fetch(server.url+'/api/builder/example')).json();
  changed.views[0].fovY-=.5;
  assert.equal((await invoke('build_with_orbit_views',{plan:changed,taskIds:[taskId],keepSources:[]})).status,422);
  const built=await invoke('build_with_orbit_views',{sceneCode:input.sceneCode,taskIds:[taskId],keepSources:[]});
  assert.equal(built.status,201,await built.clone().text());
  const result=(await built.json()).data;
  assert.equal(result.plan.supplements[0].model,input.model);
  assert.ok(result.report.mesh.supplements.byView[job.spec.source]>0);
  assert.equal((await fetch(server.url+result.scene.url,{method:'HEAD'})).status,200);
  assert.equal((await invoke('create_orbit_view',input)).status,503);
});

test('ground texture import and Qwen job create separate persistent scene versions', async t => {
  let upload, workflow;
  const fake = http.createServer(async (req,res) => {
    const chunks = []; for await (const c of req) chunks.push(c);
    const raw = Buffer.concat(chunks);
    if (req.url.startsWith('/view?')) return res.end(upload);
    res.setHeader('content-type','application/json');
    if (req.url === '/upload/image') {
      const form = await new Response(raw,{headers:{'content-type':req.headers['content-type']}}).formData();
      upload = Buffer.from(await form.get('image').arrayBuffer());
      return res.end(JSON.stringify({name:'ground.png'}));
    }
    if (req.url === '/prompt') {
      workflow = JSON.parse(raw).prompt;
      return res.end(JSON.stringify({prompt_id:'ground-test'}));
    }
    res.end(JSON.stringify({'ground-test':{outputs:{9:{images:[{filename:'ground.png',type:'output'}]}}}}));
  });
  fake.listen(0,'127.0.0.1'); await once(fake,'listening');
  const data = await mkdtemp(path.join(os.tmpdir(),'scene-ground-'));
  let server = await launch(data,{COMFYUI_URL:'http://127.0.0.1:'+fake.address().port});
  t.after(async () => { await server.close(); fake.close(); await rm(data,{recursive:true,force:true}); });
  const base = '/api/builder/scenes/VILLAGE_CALIBRATED/ground';
  const photo = await readFile(path.join(root,'assets/scene_sources/village_road_near.png'));
  assert.equal((await fetch(server.url+base+'/apply',{method:'POST',body:photo})).status,422);
  const image = await readFile(path.join(root,'builder/example-ground/texture.png'));
  const valid = await fetch(server.url+base+'/apply',{method:'POST',body:image});
  assert.equal(valid.status,201);
  const parent = (await valid.json()).data;
  assert.equal(parent.report.mesh.ground.method,'imported');
  const submission = await post(server.url,parent.scene.ground_url+'/redraw',{prompt:'Extend the meadow'});
  assert.equal(submission.status,202);
  const {taskId} = (await submission.json()).data;
  const route = '/api/builder/ground-jobs/'+taskId;
  let job;
  for (let i = 0; i < 180; i++) {
    job = (await (await fetch(server.url+route)).json()).data;
    if (job.status !== 'running') break;
    await new Promise(resolve => setTimeout(resolve,200));
  }
  assert.equal(job.status,'completed',job.message);
  assert.equal(job.report.mesh.ground.parentScene,parent.scene.code);
  assert.equal(job.report.mesh.ground.method,'qwen-image-2.1');
  assert.match(workflow['2'].inputs.unet_name,/qwen_image_2\.1/);
  assert.match(workflow['6'].inputs.prompt,/orthographic.*Extend the meadow/s);
  assert.equal(upload.readUInt32BE(16),1024); assert.equal(upload.readUInt32BE(20),1024);
  assert.equal(workflow['7'].inputs.denoise,1);
  await server.close(); server = await launch(data);
  const restored = (await (await fetch(server.url+route)).json()).data;
  assert.equal(restored.scene.code,job.scene.code);
  assert.equal((await fetch(server.url+restored.scene.url,{method:'HEAD'})).status,200);
  const library = (await (await fetch(server.url+'/api/scenes/3d')).json()).data.scenes;
  assert.equal(library.filter(s=>s.code.startsWith('BUILD_')).length,2);
});

test('vision adapter submits only to the configured endpoint and returns editable, estimated outlines', async t => {
  const plan = JSON.parse(await readFile(path.join(root, 'builder/example.json'), 'utf8'));
  let response, received;
  const model = http.createServer(async (req, res) => {
    let text = ''; for await (const part of req) text += part;
    received = { path: req.url, auth: req.headers.authorization, input: JSON.parse(text) };
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify({ choices: [{ message: { content: JSON.stringify(response) } }] }));
  });
  model.listen(0, '127.0.0.1'); await once(model, 'listening');
  const data = await mkdtemp(path.join(os.tmpdir(), 'scene-planner-'));
  const server = await launch(data, { SCENE_PLANNER_URL: 'http://127.0.0.1:' + model.address().port + '/v1/chat/completions',
    SCENE_PLANNER_MODEL: 'test-vision', SCENE_PLANNER_KEY: 'test-only' });
  t.after(async () => { await server.close(); model.close(); await rm(data, { recursive: true, force: true }); });
  const b = structuredClone(plan.buildings[0]); b.heightSource = 'measured';
  response = { buildings: [b] };
  const res = await post(server.url, '/api/builder/propose', { plan, instruction: 'Identify the roof' });
  assert.equal(res.status, 200, await res.clone().text());
  assert.equal(received.path, '/v1/chat/completions');
  assert.equal(received.auth, 'Bearer test-only');
  assert.equal(received.input.model, 'test-vision');
  assert.ok(received.input.messages[1].content.some(x => x.image_url?.url.startsWith('data:image/png;base64,')));
  const draft = (await res.json()).data.plan;
  assert.deepEqual(draft.views, plan.views);
  assert.equal(draft.buildings[0].heightSource, 'estimated');
  response.buildings[0].observations[0].source = 'unknown';
  assert.equal((await post(server.url, '/api/builder/propose', { plan, instruction: 'test' })).status, 422);
});
