<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import { ApiError, getJson } from "../api/http";
import { buildOrbit, createOrbit, getOrbit, getOrbitProviders, imageUrl, maxSupplementViews, uploadPhoto,
  type Alignment, type BuildResult, type OrbitCamera, type OrbitCatalog, type OrbitProvider, type OrbitPoint, type Pair, type Plan, type Report, type Supplement } from "../api/builder";
import type { SceneMeshInfo } from "../api/scene3d";

const props=defineProps<{scene?:SceneMeshInfo;draft?:Plan;previewTarget?:string;initialView?:string}>();
const emit=defineEmits<{result:[BuildResult,Plan];busy:[boolean];camera:[OrbitCamera]}>();
type Entry={spec:Supplement;quality?:Alignment;guide?:string;overlay?:string;strength?:number};
const plan=ref<Plan>(), report=ref<Report>(), entries=ref<Entry[]>([]), chosen=ref<string[]>([]);
const selected=ref(""), reference=ref(""), angle=ref<number>();
const catalog = ref<OrbitCatalog>(), providerId=ref(""), model=ref(""), guidance=ref(""), quality=ref(""), instruction=ref("");
const connectionError=ref("");
const provider=computed(()=>catalog.value?.providers.find(p=>p.id===providerId.value));
const checkingOptions = ref(true);
const direction = computed(() => provider.value?.directions?.find(d => d.angle === angle.value));
const manualStrength = ref<string | null>(null);
const strengthFormat = new Intl.NumberFormat("zh-CN", { minimumFractionDigits: 2, maximumFractionDigits: 6 });
const strength = computed(() => {
  if (manualStrength.value === null) return direction.value?.suggestedStrength;
  if (manualStrength.value.trim() === "") return undefined;
  const value = Number(manualStrength.value);
  return Number.isFinite(value) && value >= 0 && value <= 1 ? value : undefined;
});
function setStrength(event: Event) {
  manualStrength.value = (event.target as HTMLInputElement).value;
}
const validAngle=computed(()=>!!catalog.value && Number.isFinite(angle.value) &&
  Math.abs(angle.value!)>=catalog.value.minAngle && Math.abs(angle.value!)<=catalog.value.maxAngle);
const canGenerate = computed(() => !checkingOptions.value && provider.value?.ready && !!model.value &&
  (plan.value?.buildings.length ?? 0)>0 && validAngle.value &&
  (provider.value.angleMode!=="directions" || !!direction.value) && (!provider.value.supportsStrength || strength.value!==undefined));
const layer=ref<"image"|"guide"|"overlay">("image"), working=ref(""), error=ref(""), fileInput=ref<HTMLInputElement>();
const active=computed(()=>entries.value.find(e=>e.spec.source===selected.value));
const surfaceDetail=computed(()=>report.value?.mesh?.surfaceDetail);
const infillCount=computed(()=>Object.values(surfaceDetail.value?.context || {}).reduce((n,view)=>n+view.infillBuildings,0));
const globalSurface=computed(()=>surfaceDetail.value?.globalSurface);
const preview=computed(()=>{
  if (!active.value) return reference.value ? imageUrl(reference.value) : "";
  return (layer.value==="overlay" ? active.value.overlay : layer.value==="guide" ? active.value.guide : "") || imageUrl(active.value.spec.source);
});
const name=(a:number)=>(a<0?"左转 ":"右转 ")+Math.abs(a)+"°";
const caption=(e:Entry)=>name(e.quality?.status==="aligned" && e.quality.actualAngle!=null ? e.quality.actualAngle : e.spec.angle);
const qualityLabels:Record<string,string>={low:"低",medium:"中",high:"高",auto:"自动"};
const manual=ref(false), pairs=ref<OrbitPoint[]>([]), pending=ref<Pair>(), manualDialog=ref<HTMLDialogElement>();
const holdout=ref(false), fitCount=computed(()=>pairs.value.filter(p=>!p.check).length);
let epoch=0;
const message=(e:unknown)=>e instanceof Error?e.message:String(e);
function busy(s:string){working.value=s;emit("busy",!!s);}
function select(id:string){ selected.value=id; layer.value="image"; }
function toggle(id:string){
  if(chosen.value.includes(id)) chosen.value=chosen.value.filter(k=>k!==id);
  else if(chosen.value.length<maxSupplementViews) chosen.value=[...chosen.value,id];
}
function savedEntry(spec:Supplement,summary?:Alignment):Entry {
  let base="";
  if(spec.source.startsWith("orb_")) base="/api/builder/orbit-jobs/"+spec.source.slice(4);
  if(spec.source.startsWith("village-orbit-")) base="/api/builder/sample-views/"+spec.source;
  return {spec,quality:summary,guide:base?base+"/guide":undefined,overlay:base?base+"/overlay":undefined};
}
const storageKey=()=> "scene.orbit.job."+(props.scene?.code||((plan.value?.name||"")+"."+plan.value?.views.map(v=>v.source).join(".")));
function storeJob(id:string){try{if(id)localStorage.setItem(storageKey(),id);else localStorage.removeItem(storageKey());}catch{/* job can also be queried by API */}}
async function poll(id:string,token:number){
  while(token===epoch){
    const job=await getOrbit(id);
    if(token!==epoch)return;
    if(job.status==="failed"){storeJob("");throw new Error(job.message||"旋转视图生成失败");}
    if(job.status==="completed"){
      if(!job.spec || !job.quality)throw new Error("补充视图缺少配准报告");
      const e:Entry={spec:job.spec,quality:job.quality,guide:job.guideUrl,overlay:job.overlayUrl,strength:job.generation?.strength};
      const existing=entries.value.findIndex(v=>v.spec.source===e.spec.source);
      if(existing<0)entries.value.push(e);else entries.value[existing]=e;
      select(e.spec.source);
      if(e.quality?.status==="aligned" && !e.quality.targetMismatch && !chosen.value.includes(e.spec.source))toggle(e.spec.source);
      return;
    }
    await new Promise(resolve=>setTimeout(resolve,1800));
  }
}
function chooseProvider(next:OrbitProvider|undefined) {
  const same = next?.id===providerId.value;
  providerId.value=next?.id || "";
  model.value=same && next?.models.includes(model.value) ? model.value : next?.models[0] || "";
  guidance.value=same && next?.guidanceModes?.some(g=>g.value===guidance.value) ? guidance.value : next?.defaultGuidance || "";
  quality.value=same && next?.qualities?.includes(quality.value) ? quality.value : next?.defaultQuality || "";
  if (!same) manualStrength.value=null;
  if (angle.value==null) angle.value=next?.defaultAngle ?? catalog.value?.defaultAngle;
  if (next?.angleMode==="directions" && next.directions?.length && !next.directions.some(d=>d.angle===angle.value))
    angle.value=(next.directions.find(d=>d.angle===next.defaultAngle) ?? next.directions[0]).angle;
}
function changeProvider(e:Event){chooseProvider(catalog.value?.providers.find(p=>p.id===(e.target as HTMLSelectElement).value));}
async function refreshOptions() {
  checkingOptions.value = true;
  connectionError.value="";
  try {
    const next = await getOrbitProviders();
    catalog.value=next;
    chooseProvider(next.providers.find(p=>p.id===providerId.value) ?? next.providers.find(p=>p.id===next.defaultProvider) ?? next.providers[0]);
  } catch (e) {
    const missing = (e instanceof ApiError && e.status === 404) || e instanceof SyntaxError;
    const detail = missing ? "当前端口的场景服务缺少有效的视角选项接口，请更新并重启该服务。" : message(e);
    connectionError.value=detail;
    catalog.value=undefined;
  } finally {
    checkingOptions.value = false;
  }
}
onMounted(refreshOptions);
watch(()=>[props.scene?.code,props.draft] as const,async()=>{
  const token=++epoch;
  entries.value=[];chosen.value=[];error.value="";plan.value=undefined;report.value=undefined;reference.value="";select("");busy("");
  try{
    const [p,r]=await Promise.all([props.draft??(props.scene?.plan_url?getJson<Plan>(props.scene.plan_url):undefined),
      props.scene?.report_url?getJson<Report>(props.scene.report_url):undefined]);
    if(token!==epoch||!p)return;
    plan.value=p;report.value=r;reference.value=p.views[0]?.source||"";
    entries.value=(p.supplements||[]).map(s=>savedEntry(s,r?.supplements?.find(x=>x.source===s.source)));
    chosen.value=entries.value.map(e=>e.spec.source);select(chosen.value[chosen.value.length-1]||"");
    const requested=entries.value.find(e=>e.spec.source===props.initialView);
    if(requested?.quality?.camera){select(requested.spec.source);emit("camera",requested.quality.camera);}
    let id:string|null=null;try{id=localStorage.getItem(storageKey());}catch{/* no stored task */}
    if(id){busy("读取机位任务");await poll(id,token);}
  }catch(e){if(token===epoch)error.value=message(e);}
  finally{if(token===epoch)busy("");}
},{immediate:true});
onBeforeUnmount(()=>{epoch++;emit("busy",false);});
async function submit(image?:string,marks?:OrbitPoint[],from?:Supplement,file?:File){
  const targetAngle=from?.angle ?? angle.value;
  if(working.value||!plan.value||targetAngle==null||(!image&&!file&&!canGenerate.value))return;
  const token=epoch;error.value="";busy(image||file?"正在配准补充视图":model.value+" 正在生成机位");
  try{
    if(file) image=await uploadPhoto(file);
    if(token!==epoch)return;
    const p=provider.value;
    const job=await createOrbit({plan:plan.value,reference:from?.reference||reference.value,angle:targetAngle,image,points:marks,
      provider:image?from?.provider:p?.id,model:image?from?.model:model.value,registration:file?"measured":from?.registration,
      strength:!image && p?.supportsStrength ? strength.value:undefined,quality:!image && p?.qualities ? quality.value:undefined,
      guidance:!image && p?.guidanceModes ? guidance.value:undefined,instruction:!image && p?.acceptsInstruction ? instruction.value.trim()||undefined:undefined});
    if(token!==epoch)return;
    storeJob(job.taskId);await poll(job.taskId,token);
  }catch(e){if(token===epoch)error.value=message(e);}
  finally{if(token===epoch)busy("");}
}
function fileChanged(event:Event){const el=event.target as HTMLInputElement,f=el.files?.[0];el.value="";if(f)void submit(undefined,undefined,undefined,f);}
async function rebuild(){
  if(!plan.value||working.value)return;
  const token=epoch;error.value="";busy("正在重建建筑与地面");
  try{
    const keep=(plan.value.supplements||[]).map(s=>s.source).filter(id=>chosen.value.includes(id));
    const added=entries.value.filter(e=>chosen.value.includes(e.spec.source) && !keep.includes(e.spec.source));
    const result=await buildOrbit(plan.value,added.map(e=>e.spec.source.slice(4)),keep,added.some(e=>e.quality?.targetMismatch));
    if(token===epoch){storeJob("");emit("result",result,result.plan);}
  }catch(e){if(token===epoch)error.value=message(e);}
  finally{if(token===epoch)busy("");}
}
function openManual(){pairs.value=[...(active.value?.spec.points||[])];pending.value=undefined;holdout.value=false;manual.value=true;manualDialog.value?.showModal();}
function mark(event:MouseEvent,side:"guide"|"image"){
  if(pairs.value.length>=64)return;
  const rect=(event.currentTarget as HTMLElement).getBoundingClientRect();
  const uv:Pair=[Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width)),Math.max(0,Math.min(1,(event.clientY-rect.top)/rect.height))];
  if(side==="guide")pending.value=uv;
  else if(pending.value){pairs.value.push({guide:pending.value,image:uv,...(holdout.value?{check:true}:{})});pending.value=undefined;}
}
function registerMarks(){const e=active.value;if(!e)return;manualDialog.value?.close();void submit(e.spec.source,[...pairs.value],e.spec);}
</script>
<template>
  <div class="orbit-ui">
    <Teleport :to="previewTarget||'body'" :disabled="!previewTarget">
      <section v-if="preview" class="orbit-preview" :class="{large:!!previewTarget}">
        <header><strong>{{active?caption(active):'原始参考图'}}</strong><span>{{active?.spec.model || (active?'补充机位':'拍摄机位')}}</span></header>
        <img class="hero" :src="preview" :alt="active?'补充机位预览':'建模参考照片'"/>
        <nav v-if="active" aria-label="查看机位图层"><button :aria-pressed="layer==='image'" @click="layer='image'">生成图</button>
          <button v-if="active.guide" :aria-pressed="layer==='guide'" @click="layer='guide'">结构图</button>
          <button v-if="active.overlay" :aria-pressed="layer==='overlay'" @click="layer='overlay'">轮廓叠加</button></nav>
      </section>
    </Teleport>
    <div v-if="plan" class="orbit-content">
      <header><h3>补充视角建模</h3><span v-if="globalSurface?.retainedBuildings != null">{{globalSurface.closedAnchorBuildings + globalSurface.retainedBuildings}} 栋实体建筑</span><span v-else>{{plan.buildings.length}} 栋建筑<span v-if="infillCount"> + {{infillCount}} 栋补建</span></span></header>
      <p v-if="globalSurface || report?.orbitCoverage" class="info">
        <span v-if="globalSurface">全局融合网格 · {{globalSurface.triangles.toLocaleString()}} 个三角面</span>
        <br v-if="globalSurface && report?.orbitCoverage"/>
        <span v-if="report?.orbitCoverage">最大未采样方位跨度 {{report.orbitCoverage.largestGapDeg}}°</span>
      </p>
      <p v-if="surfaceDetail?.videoSurfaces?.buildings" class="info">视频复核 · {{surfaceDetail.videoSurfaces.roofTextures}} 片屋面、{{surfaceDetail.videoSurfaces.wallTextures}} 面墙采用视频纹理<span v-if="surfaceDetail.videoSurfaces.correctedFootprints">，{{surfaceDetail.videoSurfaces.correctedFootprints}} 栋校正檐口</span></p>
      <p v-else-if="surfaceDetail?.videoSurfaces?.reason" class="info">{{surfaceDetail.videoSurfaces.reason}}</p>
      <button v-if="!previewTarget && surfaceDetail?.videoSurfaces?.inspectionCamera" @click="emit('camera',surfaceDetail.videoSurfaces.inspectionCamera)">近看相邻屋顶接缝</button>
      <div class="filmstrip">
        <button :aria-pressed="!active" @click="select('')"><img :src="imageUrl(reference)" alt=""/><span>原图</span></button>
        <button v-for="e in entries" :key="e.spec.source" :aria-pressed="selected===e.spec.source" @click="select(e.spec.source)">
          <img :src="imageUrl(e.spec.source)" alt=""/><span>{{caption(e)}}</span></button>
      </div>
      <template v-if="active">
        <p v-if="active.quality" class="info">{{active.quality.status==='aligned'?'已对齐':active.quality.status==='needs-detail'?'需补充纹理':'待配准'}} · {{active.quality.inliers}} 个内点 · 配准点覆盖画幅 {{active.quality.coveragePercent}}%<br/>请求{{name(active.spec.angle)}}<span v-if="active.quality.actualAngle!=null"> · {{active.quality.status==='aligned'?'实测':'初估'}}{{name(active.quality.actualAngle)}}</span></p>
        <p v-if="surfaceDetail?.photoProjectors?.pairedAreaPercent?.[selected] != null" class="info">通过相邻照片复核的图域 {{surfaceDetail.photoProjectors.pairedAreaPercent[selected]}}%</p>
        <p v-else-if="active.quality?.newlyVisibleFaces" class="info">该机位可补充 {{active.quality.newlyVisibleFaces}} 个原图遮挡面</p>
        <p v-if="active.quality?.checkPoints" class="info">{{active.quality.checkPoints}} 个独立检查点 · 偏差中位数 {{active.quality.checkMedianPx}} px</p>
        <p v-if="active.strength!=null" class="info">本次生成强度 {{active.strength}}</p>
        <p v-if="globalSurface?.byView[selected] != null" class="info">此图为 {{globalSurface.byView[selected].toLocaleString()}} 个融合面提供纹理</p>
        <p v-if="surfaceDetail?.reviewedFacesByView?.[selected]" class="info">已逐面校正 {{surfaceDetail.reviewedFacesByView[selected]}} 个建筑面</p>
        <button v-if="!previewTarget && active.quality?.camera" @click="emit('camera',active.quality.camera)">从此机位查看 3D</button>
        <label class="use-view"><input type="checkbox" :checked="chosen.includes(active.spec.source)" :disabled="!!working || (active.quality && active.quality.status!=='aligned') || (!chosen.includes(active.spec.source)&&chosen.length>=maxSupplementViews)" @change="toggle(active.spec.source)"/> 采用这张补充视图</label>
        <p v-if="active.quality?.status==='needs-detail'" class="info">{{active.quality.message}}</p>
        <p v-if="active.quality?.targetMismatch && active.quality.status==='aligned'" class="info">{{ active.quality.message }}</p>
        <p v-if="active.quality?.status==='aligned' && !chosen.includes(active.spec.source) && chosen.length>=maxSupplementViews" class="info">已选 {{maxSupplementViews}} 个机位，取消一个后可采用当前视图。</p>
        <p v-if="active.quality?.status==='needs-points'" class="info">{{active.quality.message}}</p>
        <button v-if="active.guide" :disabled="!!working" @click="openManual">{{active.spec.points?.length?'查看或修改同名点':'标记同名点'}}</button>
      </template>
      <details v-if="surfaceDetail?.localMaterialReferences?.length" class="local-materials">
        <summary>局部材质 · {{surfaceDetail.localMaterialReferences.length}} 张参考图</summary>
        <figure v-for="m in surfaceDetail.localMaterialReferences" :key="m.source">
          <a :href="imageUrl(m.source)" target="_blank" rel="noopener" :aria-label="'打开请求'+name(m.angle)+'的材质参考图'">
            <img :src="imageUrl(m.source)" loading="lazy" :alt="name(m.angle)+'的生成图'"/>
          </a>
          <figcaption>
            <strong>请求{{name(m.angle)}}</strong><span v-if="m.model"> · {{m.model}}</span><br/>
            采用 {{m.walls}} 面墙、{{m.roofs}} 个屋顶、{{m.yardTexels.toLocaleString()}} 个院地纹理像素
            <template v-if="m.checkMedianPx!=null"><br/>整图独立检查中位偏差 {{m.checkMedianPx}} px</template>
          </figcaption>
        </figure>
      </details>
      <fieldset :disabled="!!working">
        <legend>生成新机位</legend>
        <label v-if="plan.views.length>1">原始参考图<select v-model="reference" @change="select('')"><option v-for="(v,i) in plan.views" :key="v.source" :value="v.source">参考图 {{i+1}}</option></select></label>
        <label v-if="catalog?.providers.length" class="direction">图像服务<select aria-label="图像服务" :value="providerId" @change="changeProvider"><option v-for="p in catalog.providers" :key="p.id" :value="p.id">{{p.label}}{{p.ready?'':' · 未就绪'}}</option></select></label>
        <label v-if="provider" class="direction">模型<select aria-label="生成模型" v-model="model"><option v-for="m in provider.models" :key="m">{{m}}</option></select></label>
        <template v-if="provider?.angleMode==='directions' && provider.directions?.length">
          <label class="direction">拍摄方向<select aria-label="拍摄方向" v-model.number="angle"><option v-for="d in provider.directions" :key="d.angle" :value="d.angle">{{d.label}} · {{Math.abs(d.angle)}}° 档</option></select></label>
          <div v-if="direction && provider.supportsStrength" class="strength-control">
            <label class="strength-number">LoRA 强度<input aria-label="LoRA 强度数值" type="number" min="0" max="1" step="any"
              :value="manualStrength??direction.suggestedStrength" @input="setStrength"/></label>
            <input aria-label="调整 LoRA 强度" type="range" min="0" max="1" step=".01" :value="strength??direction.suggestedStrength" @input="setStrength"/>
            <div class="strength-mode"><span>{{manualStrength===null?'随角度':'手动'}} · 此方向建议 {{strengthFormat.format(direction.suggestedStrength)}}</span>
              <button v-if="manualStrength!==null" type="button" @click="manualStrength=null">恢复随角度</button></div>
            <p v-if="strength===undefined" class="problem" role="alert">请输入 0 到 1 之间的强度。</p>
          </div>
        </template>
        <div v-else-if="catalog" class="angle-fields"><label>拍摄方向<select aria-label="机位预设" :value="catalog.presets.some(p=>p.angle===angle)?angle:''" @change="e=>{const v=(e.target as HTMLSelectElement).value;if(v)angle=Number(v)}">
          <option value="">自定义</option><option v-for="p in catalog.presets" :key="p.angle" :value="p.angle">{{p.label}}</option></select></label>
          <label>角度 °<input aria-label="目标旋转角度" type="number" step="any" v-model.number="angle" :min="-catalog.maxAngle" :max="catalog.maxAngle"/></label></div>
        <div v-if="reference===plan.views[0]?.source && report?.orbitCoverage?.suggestedAnglesDeg?.length" class="angle-suggestions">
          <span>建议补拍方位</span>
          <template v-for="a in report.orbitCoverage.suggestedAnglesDeg" :key="a">
            <button v-if="catalog && (provider?.angleMode!=='directions' || provider.directions?.some(d=>d.angle===a))" type="button" :aria-label="'填入'+name(a)" @click="angle=a">{{name(a)}}</button>
            <span v-else>{{name(a)}}</span>
          </template>
        </div>
        <label v-if="provider?.guidanceModes" class="direction">机位引导<select aria-label="机位引导" v-model="guidance"><option v-for="g in provider.guidanceModes" :key="g.value" :value="g.value">{{g.label}}</option></select></label>
        <details v-if="provider?.qualities || provider?.acceptsInstruction" class="generation-settings"><summary>出图设置</summary>
          <label v-if="provider.qualities" class="direction">图像质量<select v-model="quality"><option v-for="q in provider.qualities" :key="q" :value="q">{{qualityLabels[q]||q}}</option></select></label>
          <label v-if="provider.acceptsInstruction">补充说明（可选）<textarea v-model="instruction" maxlength="1200" rows="2" placeholder="例如保留屋檐的颜色和材质"/></label></details>
        <button class="generate-view" :disabled="!canGenerate" @click="submit()">生成补充视角</button>
        <p v-if="checkingOptions" class="info" role="status">正在读取视角选项…</p>
        <div v-else-if="!provider?.ready" class="provider-status"><p class="info" role="status">{{connectionError || provider?.message || '部署端尚未配置图像服务，可导入已有旋转图。'}}</p>
          <button type="button" @click="refreshOptions">重新检测</button></div>
        <p v-else-if="!plan.buildings.length" class="info">请先在原图标出至少一栋建筑的轮廓。</p>
        <input type="file" ref="fileInput" accept="image/png,image/jpeg" hidden @change="fileChanged"/>
        <button :disabled="!validAngle || !plan.buildings.length" @click="fileInput?.click()">导入已有旋转图</button>
      </fieldset>
      <button class="rebuild" :disabled="!!working || (!chosen.length && !plan.supplements?.length)" @click="rebuild">使用 {{chosen.length}} 个补充机位重建</button>
      <p v-if="report?.mesh?.supplements" class="info">本版补充 {{report.mesh.supplements.buildingFaces}} 个建筑面<span v-if="report.mesh.ground?.syntheticTexels">、{{report.mesh.ground.syntheticTexels.toLocaleString()}} 个地面纹理像素</span><span v-if="report.mesh.supplements.openingFrames">、{{report.mesh.supplements.openingFrames}} 组窗部构件</span>。</p>
      <a class="tool-link" href="/api/builder/tools" target="_blank" rel="noopener">查看 Agent 工具接口 ↗</a>
    </div>
    <p v-if="working" class="info" role="status">{{working}}…</p><p v-if="error" class="problem" role="alert">{{error}}</p>
    <dialog ref="manualDialog" class="point-dialog" @close="manual=false">
      <template v-if="manual && active"><header><h3>同名点配准</h3><button @click="manualDialog?.close()">关闭</button></header>
        <p>先点左图的屋角，再点右图的相同位置。选至少六对，尽量分布在近处和远处的不同建筑。</p>
        <label class="use-view"><input type="checkbox" v-model="holdout"/> 下一对作为独立检查点</label>
        <div class="pair-grid"><div v-for="side in (['guide','image'] as const)" :key="side"><strong>{{side==='guide'?'目标结构图':'生成视图'}}</strong>
          <div class="pick" @click="mark($event,side)"><img :src="side==='guide'?active.guide:imageUrl(active.spec.source)" alt="选择对应点"/>
            <i v-for="(pt,i) in pairs" :key="i" :class="{check:pt.check}" :style="{left:pt[side][0]*100+'%',top:pt[side][1]*100+'%'}">{{pt.check?'检':''}}{{i+1}}</i>
            <i v-if="pending && side==='guide'" :style="{left:pending[0]*100+'%',top:pending[1]*100+'%'}">{{pairs.length+1}}</i></div></div></div>
        <footer><span>{{fitCount}} 对拟合点 · {{pairs.length-fitCount}} 对检查点</span><button @click="pending?pending=undefined:pairs.pop()">撤回一点</button>
          <button class="rebuild" :disabled="fitCount<6 || !!pending" @click="registerMarks">重新配准</button></footer>
      </template>
    </dialog>
  </div>
</template>
<style scoped>
.orbit-ui,.orbit-content,.orbit-content fieldset,.orbit-preview{display:grid;gap:10px;min-width:0}.orbit-ui,.orbit-preview,.point-dialog{font-size:13px;color:#e2e9dc}
header,footer,.angles,.orbit-preview nav{display:flex;align-items:center;justify-content:space-between;gap:6px}h3{margin:0;font-size:16px;letter-spacing:-.02em}
header span,.info,.problem{font-size:12px;color:#b3bea9;margin:0;line-height:1.6}.problem{color:#f4ab92}.hero{width:100%;display:block;border:1px solid #46513f;border-radius:5px}
button,select,input[type=number]{font:inherit;background:#232c21;color:#dce6d3;border:1px solid #46513f;padding:7px;border-radius:4px;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}
button[aria-pressed=true]{border-color:#add286;background:#33432c}.orbit-preview nav button,.angles button{flex:1;font-size:12px}.large{padding:15px}.large .hero{height:clamp(240px,52vh,650px);object-fit:contain;background:#101510}.large nav{width:min(330px,100%)}
.filmstrip{display:grid;grid-template-columns:repeat(auto-fit,minmax(48px,1fr));gap:5px}.filmstrip button{padding:2px;font-size:11px;min-width:0}.filmstrip img{display:block;width:100%;aspect-ratio:1.55;object-fit:cover}.filmstrip span{display:block;padding:4px 0}
.orbit-content fieldset{padding:10px 0 0;margin:4px 0 0;border:0;border-top:1px solid #43513e}.orbit-content legend{font-weight:650;padding:0 6px 0 0}.angles label{display:flex;align-items:center;gap:4px;white-space:nowrap;font-size:11px}.angles input{width:55px}
.use-view{display:flex;align-items:center;gap:5px}.use-view input{accent-color:#b5dd84}.orbit-ui .rebuild{background:#bfdc87;color:#192512;font-weight:700}
.direction,.strength-number,.strength-mode,.provider-status{display:flex;align-items:center;justify-content:space-between;gap:8px}
.strength-control{display:grid;gap:6px}.strength-control input[type=range]{width:100%;margin:0;accent-color:#b5dd84}.strength-number input{width:75px}
.strength-mode{color:#b3bea9;font-size:12px}.strength-mode button,.provider-status button{font-size:12px;padding:4px 7px;white-space:nowrap}
.direction select{max-width:72%}.angle-fields{display:grid;grid-template-columns:1.35fr 1fr;gap:8px}.angle-fields label,.generation-settings label{display:grid;gap:4px}.angle-fields input{min-width:0;width:100%}.generation-settings summary{cursor:pointer;color:#c5d6b2}.generation-settings[open]>*+*{margin-top:8px}.generation-settings textarea{width:100%;resize:vertical;background:#232c21;border:1px solid #46513f;border-radius:4px;color:inherit;font:inherit;padding:6px}.generate-view{font-weight:650;border-color:#89a779}.tool-link{justify-self:start;color:#b5cd9d;font-size:12px}
.point-dialog{width:min(1100px,95vw);max-height:92vh;background:#192018;border:1px solid #758767;border-radius:8px;padding:18px}.point-dialog::backdrop{background:#101510c9}.pair-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0}
.pick{position:relative;cursor:crosshair}.pick img{width:100%;height:auto;display:block}.pick i{position:absolute;font:11px monospace;transform:translate(-50%,-50%);border-radius:100%;padding:2px 4px;background:#d3ee88;color:#192512;pointer-events:none}
.pick i.check{background:#fcbe76}
.local-materials summary{color:#c5d6b2;cursor:pointer}
.local-materials figure{margin:10px 0 0;display:grid;grid-template-columns:minmax(70px,32%) 1fr;align-items:start;gap:9px}
.local-materials a{display:block}.local-materials img{display:block;width:100%;border:1px solid #46513f;border-radius:3px}
.local-materials figcaption{font-size:12px;line-height:1.5;color:#bfccb5}.local-materials strong{color:#e2e9dc}
.angle-suggestions{display:flex;align-items:center;flex-wrap:wrap;gap:5px;font-size:12px;color:#bfccb5}
.angle-suggestions button{padding:3px 6px}
@media(max-width:650px){.pair-grid{grid-template-columns:1fr}}
</style>
