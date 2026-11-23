import { readFileSync } from 'node:fs';

export const orbitDefaults = JSON.parse(readFileSync(new URL('./orbit-defaults.json', import.meta.url), 'utf8'));

export function configuredProviders(env) {
  const api = orbitDefaults.imageApi;
  const providers = [];
  const models = [...new Set((env.ORBIT_IMAGE_MODELS || api.model).split(',').map(m => m.trim()).filter(Boolean))];
  if (models.length && (env.ORBIT_IMAGE_BASE_URL || env.ORBIT_IMAGE_API_KEY || env.OPENAI_API_KEY || env.key)) {
    providers.push({id:'openai', label:'图像编辑 API', models, angleMode:'continuous', defaultAngle:orbitDefaults.defaultAngle,
      guidanceModes:api.guidanceModes, defaultGuidance:api.guidance, qualities:api.qualities, defaultQuality:api.quality,
      acceptsInstruction:true});
  }
  if (env.QMA_API_URL) providers.push({id:'qma', label:'QMA 机位 API', models:['Qwen-Image-2.1'],
    angleMode:'directions', defaultAngle:45, supportsStrength:true});
  if (env.COMFYUI_URL) providers.push({id:'comfyui', label:'ComfyUI 双图工作流', models:['Qwen-Image-2.1'],
    angleMode:'continuous', defaultAngle:38});
  return providers;
}

export function orbitTools(planSchema) {
  const maxItems = planSchema.properties.supplements.maxItems;
  const number = {type:'number'};
  const text = {type:'string'};
  const jobId = {type:'string', pattern:'^[a-f0-9]{32}$'};
  const scene = {plan:planSchema, sceneCode:{...text, description:'A saved editable scene code, or supply plan instead.'}};
  const context = {oneOf:[{required:['sceneCode']},{required:['plan']}], definitions:planSchema.definitions};
  function tool(name, description, properties, required=[], extra={}) {
    return {type:'function',function:{name,description,parameters:{type:'object',properties,required,additionalProperties:false,...extra}}};
  }
  return [
    tool('list_orbit_providers','Read configured image services, selectable models, supported camera parameters and defaults.',{}),
    tool('create_orbit_view','Generate a novel view or register an uploaded candidate. Poll the returned taskId.',
      {...scene,reference:text,angle:planSchema.definitions.supplement.properties.angle,provider:text,model:text,
       guidance:{enum:['layout','photo']},quality:text,instruction:{...text,maxLength:1200},
       strength:{...number,minimum:0,maximum:1},seed:{type:'integer',minimum:0},image:text,
       registration:{enum:['measured','target']},points:planSchema.definitions.supplement.properties.points},['angle'],context),
    tool('get_orbit_job','Get candidate images, generation settings, fitted camera and registration diagnostics.',
      {taskId:jobId},['taskId']),
    tool('build_with_orbit_views','Reconstruct a new GLB from selected registered jobs. Review any target mismatch before opting in.',
      {...scene,taskIds:{type:'array',maxItems,uniqueItems:true,items:jobId},keepSources:{type:'array',maxItems,uniqueItems:true,items:text},
       acceptMeasuredPose:{type:'boolean'}},['taskIds'],context)
  ];
}
