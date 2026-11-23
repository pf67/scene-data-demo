import { mkdir, rename, writeFile } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import path from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { configuredProviders, orbitDefaults } from './orbit-config.mjs';

// Keep candidates separate from built scenes. An unaligned image can be inspected
// and registered again with corresponding points without regenerating it.
export async function orbitService({ store, scratch, worker, sourceFiles, photoPath, body, file, json, fail, readJson, samples, readPlan, construct, maxSupplements }) {
  const jobs = path.join(store, 'orbit-jobs');
  await mkdir(jobs, { recursive: true });
  const active = new Set();
  const providers = configuredProviders(process.env);
  const profileFor = id => providers.find(p => p.id === id);
  const defaultProvider = profileFor(process.env.ORBIT_DEFAULT_PROVIDER)?.id ?? providers[0]?.id ?? null;
  async function resolvePlan(input) {
    if (Boolean(input?.plan) === Boolean(input?.sceneCode)) fail('请提供构建方案或一个可编辑场景编号',422);
    const plan = input.plan || await readPlan(input.sceneCode);
    if (!Array.isArray(plan?.views) || !Array.isArray(plan?.buildings) ||
        (plan.supplements !== undefined && (!Array.isArray(plan.supplements) || plan.supplements.some(s=>!s?.source))))
      fail('构建方案缺少有效的原图、建筑或补充机位列表',422);
    return plan;
  }
  function fingerprint(plan) {
    const data = JSON.stringify([plan.views,plan.buildings,plan.ground || null], (_, v) =>
      v && typeof v === 'object' && !Array.isArray(v) ? Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])) : v);
    return createHash('sha256').update(data).digest('hex');
  }
  async function capabilities() {
    const available = await Promise.all(providers.map(async p => {
      if (p.id !== 'qma') return {...p, ready:true};
      try { return {...p, ready:true, ...await worker(['orbit-options'],12_000)}; }
      catch (e) { return {...p, ready:false, message:e.message}; }
    }));
    return {defaultProvider, providers:available, minAngle:orbitDefaults.minAngle, maxAngle:orbitDefaults.maxAngle,
      presets:orbitDefaults.presets, defaultAngle:orbitDefaults.defaultAngle};
  }
  async function status(taskId) {
    if (typeof taskId !== 'string' || !/^[a-f0-9]{32}$/.test(taskId)) fail('无效的机位任务编号',422);
    const state=await readJson(path.join(jobs,taskId,'status.json'));
    if (!state) fail('补充机位任务不存在',404);
    if (state.status==='running' && !active.has(taskId)) return {...state,status:'failed',message:'服务已重启，请重新提交机位'};
    return state;
  }
  async function build(input) {
    const plan=await resolvePlan(input);
    const {taskIds,keepSources,acceptMeasuredPose=false}=input;
    if (!Array.isArray(taskIds) || taskIds.length>maxSupplements || new Set(taskIds).size!==taskIds.length || typeof acceptMeasuredPose!=='boolean')
      fail(`请选择最多 ${maxSupplements} 个不重复的任务，并明确是否采用实测机位`,422);
    const previous=plan.supplements || [];
    if (keepSources!==undefined && (!Array.isArray(keepSources) || keepSources.some(id=>!previous.some(s=>s.source===id))))
      fail('所保留的机位须来自当前方案',422);
    const selected=new Map(previous.filter(s=>!keepSources || keepSources.includes(s.source)).map(s=>[s.source,s]));
    for (const id of taskIds) {
      const job=await status(id);
      if (job.status!=='completed' || job.quality?.status!=='aligned') fail('所选机位尚未通过配准',422);
      if (job.planFingerprint && job.planFingerprint!==fingerprint(plan)) fail('场景几何或参考图已改变，请重新配准该机位',422);
      if (job.quality.targetMismatch && !acceptMeasuredPose) fail('机位与请求方向有偏差，请核对后按实测机位采用',422);
      selected.set(job.spec.source,job.spec);
    }
    if (selected.size>maxSupplements) fail(`最多可采用 ${maxSupplements} 个机位，请通过 keepSources 选择要保留的已有机位`,422);
    const updated={...plan,supplements:[...selected.values()]};
    return {...await construct('build',updated),plan:updated};
  }
  async function start(input) {
    const plan=await resolvePlan(input);
    const { reference=plan.views?.[0]?.source, angle, image, points, seed, strength, provider, model, guidance, quality, instruction, registration } = input;
    if (!Number.isFinite(angle) || Math.abs(angle)<orbitDefaults.minAngle || Math.abs(angle)>orbitDefaults.maxAngle)
      fail(`请选择左右 ${orbitDefaults.minAngle}° 至 ${orbitDefaults.maxAngle}° 的旋转角度`,422);
    if (seed !== undefined && (!Number.isSafeInteger(seed) || seed<0)) fail('种子须为非负安全整数',422);
    if (!Array.isArray(plan?.buildings) || !plan.buildings.length) fail('请先在原图标出建筑轮廓',422);
    if (!Array.isArray(plan.views) || !plan.views.some(v=>v?.source===reference)) fail('请选择方案中的原始参考图',422);
    if (strength !== undefined && (!Number.isFinite(strength) || strength<0 || strength>1)) fail('LoRA 强度须在 0 到 1 之间',422);
    if (provider !== undefined && !['openai','qma','comfyui'].includes(provider)) fail('未知的视角生成服务',422);
    if (registration!==undefined && (!image || !['measured','target'].includes(registration))) fail('导入图片时可指定 measured 或 target 配准',422);
    if (image!==undefined && (typeof image!=='string' || !image)) fail('请提供有效的候选图编号',422);
    if (points!==undefined && !image) fail('同名点配准需要已生成或已导入的候选图',422);
    const profile = profileFor(provider ?? defaultProvider);
    if (!image && !profile) fail('所选图像服务尚未启用，可查询可用服务或导入已有图片',503);
    if (model !== undefined && (typeof model!=='string' || !model.length || model.length>160 || (!image && !profile.models.includes(model)))) fail('所选模型未在该服务上启用',422);
    if (!image && strength!==undefined && !profile.supportsStrength) fail('所选服务不支持 LoRA 强度',422);
    if (!image && seed!==undefined && profile.id==='openai') fail('所选图像 API 不支持指定随机种子',422);
    if (guidance!==undefined && !profile?.guidanceModes?.some(g=>g.value===guidance)) fail('无效的机位引导方式',422);
    if (quality!==undefined && !profile?.qualities?.includes(quality)) fail('无效的出图质量',422);
    if (instruction!==undefined && (!profile?.acceptsInstruction || typeof instruction!=='string' || instruction.length>1200)) fail('补充说明最多 1200 字且需所选服务支持',422);
    if (image && plan.views.some(v=>v.source===image)) fail('请为补充机位使用独立的图像',422);
    const imported = image ? photoPath(image) : undefined;
    const paths = await sourceFiles({...plan,supplements:[]},true);
    const id = randomUUID().replaceAll('-','');
    const dir = path.join(jobs,id);
    await mkdir(dir);
    const statePath = path.join(dir,'status.json');
    const running = { taskId:id, status:'running', reference, angle, planFingerprint:fingerprint(plan) };
    await writeFile(statePath,JSON.stringify(running)); active.add(id);
    async function persist(state) {
      const tmp = statePath+'.tmp';
      await writeFile(tmp,JSON.stringify(state));
      // Polling clients can briefly hold the destination open on Windows.
      for (let attempt=0; ; attempt++) {
        try { await rename(tmp,statePath); return; }
        catch (error) {
          if (process.platform!=='win32' || !['EPERM','EBUSY','EACCES'].includes(error.code) || attempt===5) throw error;
          await delay(20 * 2**attempt);
        }
      }
    }
    scratch(async work => {
      const inputPath = path.join(work,'plan.json'), output = path.join(work,'orbit-'+id);
      await writeFile(inputPath,JSON.stringify(plan));
      const args = ['orbit-view','--plan',inputPath,'--reference',reference,'--angle',String(angle),'--directory',output];
      for (const [key,name] of paths) args.push('--source',key,name);
      if (imported) args.push('--image',imported);
      if (image?.startsWith('orb_') || image?.startsWith('village-orbit-')) args.push('--reuse-candidate');
      if (seed !== undefined && !imported) args.push('--seed',String(seed));
      if (!imported && strength !== undefined) args.push('--strength',String(strength));
      const prior = image?.startsWith('orb_') ? await readJson(path.join(jobs,image.slice(4),'status.json')) : null;
      const sample = samples.get(image);
      const pipeline = imported ? prior?.spec?.provider || sample?.provider || provider : profile.id;
      if (pipeline) args.push('--provider',pipeline);
      const registerMode = registration ?? prior?.spec?.registration;
      if (imported && registerMode) args.push('--registration',registerMode);
      if (pipeline === 'openai') {
        const editModel = imported ? prior?.spec?.model || sample?.model || model : model || profile.models[0];
        if (editModel) args.push('--model='+editModel);
        if (!imported) {
          args.push('--guidance',guidance || profile.defaultGuidance,'--quality',quality || profile.defaultQuality);
          if (instruction) args.push('--instruction='+instruction);
        }
      }
      if (points !== undefined) {
        const name = path.join(work,'points.json');
        await writeFile(name,JSON.stringify(points)); args.push('--points',name);
      }
      const result = await worker(args,imported ? 120_000 : 2_100_000);
      result.spec.source = 'orb_'+id;
      if (prior?.spec) result.spec.provenance=prior.spec.provenance;
      else if (sample) result.spec.provenance=sample.provenance || 'qwen-image-2.1';
      if (prior?.generation && imported) result.generation=prior.generation;
      await rename(output,path.join(dir,'result'));
      const base = '/api/builder/orbit-jobs/'+id;
      await persist({...running,status:'completed',...result, imageUrl:base+'/view', guideUrl:base+'/guide', overlayUrl:base+'/overlay'});
    }).catch(e=>persist({...running,status:'failed',message:e.message})).catch(e=>console.error('Orbit job:',e.message)).finally(()=>active.delete(id));
    return running;
  }
  async function handle(req,res,route) {
    if (req.method==='GET' && route==='/api/builder/orbit-providers') { json(res,await capabilities()); return true; }
    if (req.method==='POST' && route==='/api/builder/orbit-build') { json(res,await build(await body(req,2*1024*1024)),201); return true; }
    if (req.method === 'GET' && route === '/api/builder/orbit-options') {
      if (process.env.QMA_API_URL) {
        try {
          const options = await worker(['orbit-options'], 12_000);
          json(res, {provider:'qma', ready:true, ...options});
        } catch (e) {
          json(res, {provider:'qma', ready:false, message:e.message});
        }
      } else if (process.env.COMFYUI_URL) {
        json(res, {provider:'comfyui', ready:true});
      } else if (profileFor('openai')) {
        json(res, {provider:'openai', ready:true});
      } else {
        json(res, {provider:null, ready:false, message:'尚未配置图像服务，可先导入已有视角。'});
      }
      return true;
    }
    if (req.method==='POST' && route==='/api/builder/orbit-views') {
      json(res,await start(await body(req,2*1024*1024)),202); return true;
    }
    const match=route.match(/^\/api\/builder\/orbit-jobs\/([a-f0-9]{32})(?:\/(view|guide|overlay))?$/);
    if (!match || !['GET','HEAD'].includes(req.method)) return false;
    const dir=path.join(jobs,match[1]), state=await status(match[1]);
    if (match[2]) {
      if (state.status!=='completed') fail('补充视图尚未完成',404);
      await file(req,res,path.join(dir,'result',match[2]+'.jpg'));
    } else {
      json(res,state);
    }
    return true;
  }
  return {handle, capabilities, start, status, build, defaultProvider, configured:providers.length>0};
}
