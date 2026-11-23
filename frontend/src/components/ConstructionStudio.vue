<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { getJson } from "../api/http";
import { fetchSceneMeshes, type SceneMeshInfo } from "../api/scene3d";
import { buildPlan, checkPlan, editablePlan, examplePlan, getTools, imageUrl, proposePlan, uploadPhoto,
  type Building, type BuildResult, type Observation, type Pair, type Plan, type Report, type View } from "../api/builder";
import PhotoCanvas, { type Marker } from "./PhotoCanvas.vue";
import GroundPanel from "./GroundPanel.vue";
import OrbitPanel from "./OrbitPanel.vue";

const props = defineProps<{ embedded?: boolean }>();
const emit = defineEmits<{ open: [code?: string, panel?: "ground" | "orbit"] }>();
const plan = ref<Plan>();
const currentView = ref(0), currentBuilding = ref(0);
const mode = ref<Marker["kind"]>("roof"), marker = ref<Marker | null>(null);
const tab = ref<"edit" | "quality" | "agent" | "ground" | "orbit">("edit");
const report = ref<Report>(), built = ref<SceneMeshInfo>();
const working = ref(""), error = ref(""), notice = ref("");
const planner = ref(false), instruction = ref("辨认图中主要建筑，按屋檐轮廓生成可编辑草案。");
const proposed = ref<Plan>(), scenes = ref<SceneMeshInfo[]>([]);
const showLibrary = ref(false), spanX = ref(20), spanZ = ref(10);
const photoInput = ref<HTMLInputElement>(), importInput = ref<HTMLInputElement>();
const adding = ref(false);
const groundBusy = ref(false), maskInput = ref<HTMLInputElement>();
const view = computed(() => plan.value?.views[currentView.value]);
const building = computed(() => plan.value?.buildings[currentBuilding.value]);
const obs = computed(() => building.value?.observations.find(o => o.source === view.value?.source));
const control = computed(() => marker.value?.kind === "ground" ? view.value?.controls[marker.value.index] : undefined);
const viewReport = computed(() => report.value?.views.find(v => v.source === view.value?.source));
const buildingReport = computed(() => report.value?.buildings.find(b => b.id === building.value?.id));
const savedKey = "scene.builder.draft.v1";
const heightLabels: Record<string, string> = { estimated: "估算檐高", measured: "输入实测檐高", vertical: "墙脚与檐角求高", multiview: "多视图交会" };
const message = (e: unknown) => e instanceof Error ? e.message : String(e);
const fmt = (n: number | null | undefined) => n == null ? "—" : n.toFixed(2);

function install(value: Plan) {
  plan.value = value;
  currentView.value = currentBuilding.value = 0;
  marker.value = null;
  mode.value = value.views[0].controls.length < 4 ? "ground" : "roof";
  tab.value = "edit";
  error.value = notice.value = "";
  proposed.value = undefined;
}
watch(plan, value => {
  report.value = undefined; built.value = undefined; proposed.value = undefined;
  if (value) { try { localStorage.setItem(savedKey, JSON.stringify(value)); } catch { /* JSON export still works. */ } }
}, { deep: true, flush: "sync" });
async function refreshLibrary() { scenes.value = await fetchSceneMeshes(); }
onMounted(async () => {
  try {
    let draft: unknown;
    try { draft = JSON.parse(localStorage.getItem(savedKey) || "null"); } catch { /* Load the bundled example. */ }
    const sample = await examplePlan();
    // Upgrade an untouched 1.2 village draft; preserve user-edited outlines and bounds.
    const earlierSample = { ...sample, ground: { ...sample.ground, marginPercent: 42 } };
    delete earlierSample.supplements;
    const unchangedExample = JSON.stringify(draft) === JSON.stringify(earlierSample);
    install(editablePlan(draft) && !unchangedExample ? draft : sample);
    const tools = await getTools();
    planner.value = tools.planner;
    await refreshLibrary();
    if (JSON.stringify(plan.value) === JSON.stringify(sample)) built.value = scenes.value.find(s => s.code === "VILLAGE_CALIBRATED");
  } catch (e) { error.value = message(e); }
});
function setView(i: number) { currentView.value = i; marker.value = null; }
function setTool(tool: Marker["kind"]) { mode.value = tool; marker.value = null; tab.value = "edit"; }
function setCalibration(approximate: boolean) {
  if (!view.value || !plan.value) return;
  if (approximate) {
    if (plan.value.views.length > 1 || view.value.controls.length) return;
    view.value.approxPose = { heightM: 15, pitchDeg: 25 };
    plan.value.scaleBasis = "estimated";
  } else delete view.value.approxPose;
}
function selectBuilding(id: string) {
  currentBuilding.value = Math.max(0, plan.value?.buildings.findIndex(b => b.id === id) ?? 0);
  setTool("roof");
}
function addBuilding() {
  if (!plan.value || plan.value.buildings.length >= 150) return;
  let i = 1;
  while (plan.value.buildings.some(b => b.id === "building_" + i)) i++;
  const b: Building = { id: "building_" + i, heightM: 6, heightSource: "estimated", observations: [] };
  plan.value.buildings.push(b);
  selectBuilding(b.id);
}
function removeBuilding() {
  if (!plan.value || !building.value) return;
  plan.value.buildings.splice(currentBuilding.value, 1);
  currentBuilding.value = Math.max(0, Math.min(currentBuilding.value, plan.value.buildings.length - 1));
  marker.value = null;
}
function mark(uv: Pair) {
  if (!plan.value || !view.value) return;
  error.value = "";
  if (mode.value === "ground") {
    if (view.value.approxPose) { notice.value = "请先切换到地面控制点标定，或在屋檐轮廓工具中描点。"; return; }
    const points = view.value.controls;
    if (points.length >= 64) return;
    const corners: Pair[] = [[0, 0], [spanX.value, 0], [spanX.value, spanZ.value], [0, spanZ.value]];
    points.push({ uv, xz: corners[points.length] || [0, 0] });
    marker.value = { kind: "ground", index: points.length - 1 };
    return;
  }
  if (!building.value) addBuilding();
  if (!building.value) return;
  let observation: Observation | undefined = obs.value;
  if (!observation) {
    building.value.observations.push({ source: view.value.source, roof: [] });
    observation = obs.value!;
  }
  if (mode.value === "base") {
    if (!observation.roof.length) { error.value = "请先标出第一个檐角"; return; }
    observation.base = uv; marker.value = { kind: "base", index: 0 }; return;
  }
  if (observation.roof.length >= 16) { error.value = "每个轮廓最多 16 个点"; return; }
  observation.roof.push(uv); marker.value = { kind: "roof", index: observation.roof.length - 1 };
}
function move(point: Marker, uv: Pair) {
  if (point.kind === "ground" && view.value?.controls[point.index]) view.value.controls[point.index].uv = uv;
  else if (point.kind === "roof" && obs.value?.roof[point.index]) obs.value.roof[point.index] = uv;
  else if (point.kind === "base" && obs.value) obs.value.base = uv;
}
function removePoint() {
  const point = marker.value;
  if (!point) return;
  if (point.kind === "ground") view.value?.controls.splice(point.index, 1);
  else if (point.kind === "base" && obs.value) delete obs.value.base;
  else if (obs.value) {
    obs.value.roof.splice(point.index, 1);
    if (point.index === 0) delete obs.value.base;
    if (!obs.value.roof.length && building.value) building.value.observations = building.value.observations.filter(o => o !== obs.value);
  }
  marker.value = null;
}
function requestPhotos(append: boolean) { adding.value = append; photoInput.value?.click(); }
async function readPhotos(e: Event) {
  const input = e.target as HTMLInputElement, files = Array.from(input.files || []);
  input.value = "";
  if (!files.length) return;
  working.value = "正在导入照片"; error.value = notice.value = "";
  try {
    if (files.length + (adding.value ? plan.value?.views.length || 0 : 0) > 6) throw new Error("每个场景最多 6 张参考图");
    const ids: string[] = [];
    for (const f of files) {
      if (f.size > 16 * 1024 * 1024) throw new Error(f.name + " 超过 16 MB");
      ids.push(await uploadPhoto(f));
    }
    const views: View[] = ids.map(source => ({ source, fovY: 45, controls: [] }));
    const all = [...(adding.value && plan.value ? plan.value.views : []), ...views];
    if (new Set(all.map(v => v.source)).size !== all.length) throw new Error("同一张参考图只能添加一次");
    if (!adding.value || !plan.value) {
      if (views.length === 1) views[0].approxPose = { heightM: 15, pitchDeg: 25 };
      install({ version: 1, name: files[0].name.replace(/\.[^.]+$/, "") || "新场景", scaleBasis: "estimated", views, buildings: [] });
    } else {
      for (const v of plan.value.views) delete v.approxPose;
      setView(plan.value.views.length); plan.value.views.push(...views);
    }
    setTool("ground");
    notice.value = all.length === 1 ? "可先估算相机高度与俯角，再描屋檐生成草模。已有实测坐标时可切换标定方式。" : "已导入照片，请在同一地面坐标系内标定各个视图。";
  } catch (e) { error.value = message(e); } finally { working.value = ""; }
}
function removeView() {
  if (!plan.value || !view.value || plan.value.views.length < 2) return;
  const source = view.value.source;
  for (const b of plan.value.buildings) b.observations = b.observations.filter(o => o.source !== source);
  if (plan.value.supplements) plan.value.supplements = plan.value.supplements.filter(s => s.reference !== source);
  plan.value.views.splice(currentView.value, 1); setView(0);
}
function downloadPlan() {
  if (!plan.value) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(plan.value, null, 2)], { type: "application/json" }));
  const a = document.createElement("a"); a.href = url; a.download = plan.value.name + ".json"; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function importPlan(e: Event) {
  const input = e.target as HTMLInputElement, f = input.files?.[0]; input.value = "";
  if (!f) return;
  try {
    const value: unknown = JSON.parse(await f.text());
    if (!editablePlan(value)) throw new Error("文件不是可编辑的 v1 场景方案，请核对视图和轮廓字段");
    install(value); notice.value = "已导入方案。参考图使用当前服务中的图片编号。";
  } catch (e) { error.value = message(e); }
}
async function openPlan(url?: string) {
  try {
    install(url ? await getJson<Plan>(url) : await examplePlan());
    built.value = scenes.value.find(s => s.plan_url === (url || "/api/builder/example"));
    showLibrary.value = false;
  }
  catch (e) { error.value = message(e); }
}
function onGroundResult(result: BuildResult, source: Plan) {
  plan.value = source; report.value = result.report; built.value = result.scene;
  scenes.value = [result.scene, ...scenes.value.filter(s => s.code !== result.scene.code)];
  notice.value = "已保存新版场景，可进入 3D 编辑器查看。";
}
async function readMask(e: Event) {
  const input = e.target as HTMLInputElement, file = input.files?.[0], current = view.value;
  input.value = "";
  if (!file || !current) return;
  working.value = "正在导入地面选区"; error.value = "";
  try { current.groundMask = await uploadPhoto(file); notice.value = "地面选区已保存，重新构建后生效。"; }
  catch (e) { error.value = message(e); } finally { working.value = ""; }
}
async function run(build: boolean) {
  if (!plan.value || working.value) return;
  tab.value = "quality"; error.value = notice.value = ""; working.value = build ? "正在生成三维场景" : "正在校验几何";
  try {
    if (build) {
      const result = await buildPlan(plan.value);
      report.value = result.report; built.value = result.scene;
      await refreshLibrary();
    } else report.value = await checkPlan(plan.value);
  } catch (e) { error.value = message(e); } finally { working.value = ""; }
}
async function propose() {
  if (!plan.value) return;
  working.value = "正在生成模型草案"; error.value = notice.value = ""; proposed.value = undefined;
  try { proposed.value = await proposePlan(plan.value, instruction.value); }
  catch (e) { error.value = message(e); } finally { working.value = ""; }
}
function applyProposal() {
  if (!plan.value || !proposed.value) return;
  plan.value.buildings = proposed.value.buildings;
  currentBuilding.value = 0; setTool("roof"); notice.value = "草案已应用，请核对檐角和高度后校验。";
}
</script>

<template>
  <main class="studio">
    <header v-if="!props.embedded" class="studio-head">
      <div class="brand"><i aria-hidden="true"></i><span>场景工作台</span></div>
      <div class="head-actions"><button @click="showLibrary = !showLibrary" :aria-expanded="showLibrary">场景库</button>
        <button :disabled="!!working" @click="requestPhotos(false)">＋ 导入新场景</button><button class="editor-link" @click="emit('open', built?.code)">进入 3D 编辑器 ↗</button></div>
    </header>
    <input ref="photoInput" type="file" accept="image/png,image/jpeg" multiple hidden @change="readPhotos"/>
    <input ref="importInput" type="file" accept=".json,application/json" hidden @change="importPlan"/>
    <input ref="maskInput" type="file" accept="image/png,image/jpeg" hidden @change="readMask"/>
    <section v-if="showLibrary" class="library" aria-label="场景库">
      <div class="section-title"><h2>场景库</h2><button @click="showLibrary = false" aria-label="收起场景库">关闭</button></div>
      <div class="library-item" v-for="s in scenes" :key="s.code"><strong>{{ s.label }}</strong>
        <button v-if="s.plan_url || s.code === 'VILLAGE_COMPLETE'" :disabled="!!working" @click="openPlan(s.plan_url)">编辑构建方案</button>
        <button @click="emit('open', s.code)">打开 3D ↗</button></div>
    </section>
    <div class="title-row"><h1>照片构建场景</h1>
      <div class="actions">
        <button v-if="props.embedded" type="button" @click="showLibrary = !showLibrary" :aria-expanded="showLibrary">场景库</button>
        <button v-if="props.embedded" type="button" :disabled="!!working" @click="requestPhotos(false)">＋ 导入照片</button>
        <button :disabled="!plan || !!working || groundBusy" @click="run(false)">校验几何</button>
        <button class="primary" :disabled="!plan || !!working || groundBusy" @click="run(true)">{{ working || '构建 3D 场景' }} <span aria-hidden="true">↗</span></button></div></div>
    <p v-if="error" role="alert" class="message error">{{ error }}</p>
    <p v-else-if="notice" role="status" class="message">{{ notice }}</p>
    <div v-if="plan && view" class="workspace">
      <section class="viewer">
        <div v-if="tab === 'ground'" id="construction-ground-map"></div>
        <div v-else-if="tab === 'orbit'" id="construction-orbit-view"></div>
        <template v-else>
        <div class="viewer-head"><div><span class="square"></span><strong>参考图</strong><span class="count mono">{{ currentView + 1 }} / {{ plan.views.length }}</span></div>
          <button :disabled="!!working || plan.views.length >= 6" @click="requestPhotos(true)">＋ 增加视图</button></div>
        <div v-if="plan.views.length > 1" class="view-strip" aria-label="参考视图">
          <button v-for="(v,i) in plan.views" :key="v.source" :aria-pressed="currentView === i" @click="setView(i)">
            <img :src="imageUrl(v.source)" alt=""/>视图 {{ i+1 }}</button>
          <button :disabled="!!working" @click="removeView">移除当前视图</button></div>
        <PhotoCanvas :view="view" :buildings="plan.buildings" :selected="building?.id || ''"
          :mode="mode" :marker="marker" :locked="!!working" @mark="mark" @move="move" @select-point="marker = $event" @select-building="selectBuilding"/>
        <div class="draw-tools" aria-label="标注工具"><button :aria-pressed="mode === 'ground'" @click="setTool('ground')">地面控制点</button>
          <button :aria-pressed="mode === 'roof'" @click="setTool('roof')">屋檐轮廓</button><button :aria-pressed="mode === 'base'" @click="setTool('base')">墙脚测高</button>
          <button class="delete-point" :disabled="!marker || !!working" @click="removePoint">移除选中点</button></div>
        </template>
      </section>
      <aside class="inspector">
        <nav class="tabs" aria-label="构建面板"><button v-for="[key,label] in [['edit','标注'],['orbit','补充视角'],['ground','地面补全'],['quality','质量报告'],['agent','模型草案']]"
          :key="key" :aria-pressed="tab === key" @click="tab = key as typeof tab">{{ label }}</button></nav>
        <div v-if="tab === 'orbit'" class="panel-content">
          <OrbitPanel :scene="built" :draft="plan" preview-target="#construction-orbit-view" @result="onGroundResult" @busy="groundBusy = $event"/>
          <button v-if="built" class="primary full" @click="emit('open', built.code, 'orbit')">查看 3D 场景 ↗</button>
        </div>
        <div v-if="tab === 'ground'" class="panel-content">
          <GroundPanel :scene="built" :draft="plan" preview-target="#construction-ground-map" @result="onGroundResult" @busy="groundBusy = $event"/>
          <details class="more"><summary>当前参考图的地面选区</summary><p class="help">上传与参考图对齐的黑白图。白色表示可以采样的地面，黑色排除树冠、屋顶等物体。</p>
            <a v-if="view.groundMask" class="good" :href="imageUrl(view.groundMask)" target="_blank" rel="noopener">查看当前选区 ↗</a>
            <div class="links"><button :disabled="!!working || groundBusy" @click="maskInput?.click()">{{ view.groundMask ? '替换选区' : '导入选区' }}</button>
              <button v-if="view.groundMask" :disabled="!!working || groundBusy" @click="delete view.groundMask">移除选区</button></div>
          </details>
          <button v-if="built" class="primary full" @click="emit('open', built.code, 'ground')">打开 3D 场景 ↗</button>
        </div>
        <fieldset v-if="tab === 'edit'" :disabled="!!working">
          <label class="name-field"><small>场景名称</small><input v-model="plan.name" maxlength="100" aria-label="场景名称"/></label>
          <template v-if="mode === 'ground'">
            <div class="section-title"><h2>相机标定</h2><span class="count mono">{{ view.approxPose ? '单图草模' : view.controls.length + ' 点' }}</span></div>
            <label>标定方式<select :value="view.approxPose ? 'pose' : 'controls'" @change="setCalibration(($event.target as HTMLSelectElement).value === 'pose')">
              <option value="controls">地面控制点</option><option value="pose" :disabled="plan.views.length > 1 || view.controls.length > 0">快速估算相机</option></select></label>
            <p v-if="!view.approxPose" class="help">在同一平地上标出至少四点，填写各点的 X、Z 坐标。另加已知点可检查拟合精度。</p>
            <div class="two"><label>垂直视场角 °<input type="number" v-model.number="view.fovY" min="12" max="120" step=".1"/></label>
              <label>地面尺度<select v-model="plan.scaleBasis" :disabled="!!view.approxPose"><option value="estimated">估算</option><option value="measured">现场实测</option></select></label></div>
            <template v-if="view.approxPose"><div class="two"><label>相机离地（m）<input type="number" v-model.number="view.approxPose.heightM" min=".5" max="2000" step=".5"/></label>
              <label>向下俯角 °<input type="number" v-model.number="view.approxPose.pitchDeg" min="1" max="89"/></label></div>
              <label>画面顺时针倾斜 °<input type="number" :value="view.approxPose.rollDeg || 0" min="-60" max="60" step=".5" @change="view.approxPose.rollDeg = ($event.target as HTMLInputElement).valueAsNumber"/></label>
              <button @click="setTool('roof')">描绘屋檐轮廓 →</button></template>
            <details v-else class="more"><summary>矩形标定辅助</summary><p class="help">前四点按左远、右远、右近、左近点选，依次预填矩形四角坐标。</p>
              <div class="two"><label>宽 X（m）<input type="number" min=".1" v-model.number="spanX"/></label><label>深 Z（m）<input type="number" min=".1" v-model.number="spanZ"/></label></div></details>
            <label v-if="view.controls.length">编辑控制点<select :value="control ? marker?.index : ''" @change="marker = {kind:'ground',index:Number(($event.target as HTMLSelectElement).value)}">
              <option value="" disabled>选择图中点位</option><option v-for="(c,i) in view.controls" :key="i" :value="i">点 {{ i+1 }} · {{ c.check ? '独立检查' : '参与拟合' }}</option></select></label>
            <template v-if="control"><div class="two"><label>X（m）<input v-model.number="control.xz[0]" type="number" step=".1"/></label>
              <label>Z（m）<input v-model.number="control.xz[1]" type="number" step=".1"/></label></div>
              <label class="check"><input v-model="control.check" type="checkbox"/> 留作独立检查点</label></template>
          </template>
          <template v-else>
            <div class="section-title"><h2>建筑轮廓</h2><span class="count mono">{{ plan.buildings.length }} 栋</span></div>
            <div class="select-build"><select aria-label="当前建筑" :value="building?.id" @change="selectBuilding(($event.target as HTMLSelectElement).value)">
              <option v-if="!building" value="">尚无建筑</option><option v-for="b in plan.buildings" :key="b.id">{{ b.id }}</option></select><button @click="addBuilding">＋ 建筑</button></div>
            <template v-if="building">
              <div class="two"><label>檐高（m）<input :value="building.heightM" type="number" min=".11" max="300" step=".1"
                @change="building.heightM = ($event.target as HTMLInputElement).value === '' ? undefined : ($event.target as HTMLInputElement).valueAsNumber"/></label>
                <label>高度来源<select v-model="building.heightSource"><option value="estimated">估算</option><option value="measured">实测</option></select></label></div>
              <label>屋脊高出屋檐（m）<input type="number" :value="building.ridgeRiseM || 0" min="0" max="60" step=".1"
                @change="building.ridgeRiseM = Number(($event.target as HTMLInputElement).value)"/></label>
              <p class="help">{{ mode === 'base' ? '选择第 1 个檐角垂直下方可见的墙脚。单图构建时将用这两个点估高。' : '沿同一方向标 3–16 个檐角。多图从同一个屋角开始，按相同顺序描点。' }}</p>
              <div class="object-info"><span>{{ obs?.roof.length || 0 }} 个檐角 · {{ building.observations.length }} 个视图</span><button @click="removeBuilding">移除建筑</button></div>
            </template><p v-else class="help">添加建筑后，沿图中屋檐依次点选轮廓。</p>
          </template>
        </fieldset>
        <div v-if="tab === 'quality'" class="panel-content">
          <div class="section-title"><h2>几何质量</h2><span v-if="report" class="good">已校验</span></div>
          <p v-if="working" role="status" class="help">{{ working }}…</p>
          <p v-else-if="!report" class="help">完成标注后点「校验几何」，查看当前视图和建筑的误差。</p>
          <template v-if="report && viewReport">
            <div class="metrics"><div><small>控制点拟合</small><b>{{ fmt(viewReport.fitRmsPx) }} <em>px</em></b></div>
              <div><small>独立检查</small><b>{{ fmt(viewReport.checkRmsM) }} <em>m</em></b></div></div>
            <p class="help">{{ viewReport.cameraMethod === 'estimated' ? '当前视图使用估算相机，暂无米制精度验证。' : '当前视图使用 ' + viewReport.fitPoints + ' 个控制点，另有 ' + viewReport.checkPoints + ' 个独立检查点。' }}</p>
            <template v-if="buildingReport"><h3>{{ buildingReport.id }}</h3><dl><dt>檐高依据</dt><dd>{{ heightLabels[buildingReport.heightMethod] }}</dd><dt>檐高 / 占地</dt><dd>{{ fmt(buildingReport.heightM) }} m / {{ fmt(buildingReport.areaM2) }} m²</dd>
              <template v-if="buildingReport.minRayAngleDeg != null"><dt>最小交角</dt><dd>{{ fmt(buildingReport.minRayAngleDeg) }}°</dd>
                <dt>屋角重投影</dt><dd>{{ fmt(buildingReport.roofRmsPx) }} px</dd></template></dl></template>
            <dl v-if="report.mesh"><dt>照片贴面</dt><dd>{{ report.mesh.photoFaces }} 面</dd><dt>局部贴面</dt><dd>{{ report.mesh.partialFaces ?? 0 }} 面</dd>
              <dt>推断补面（含底面）</dt><dd>{{ report.mesh.inferredFaces }} 面</dd></dl>
            <ul class="notes"><li v-for="n in report.notes" :key="n">{{ n }}</li></ul>
            <template v-if="built"><button class="primary full" @click="emit('open', built.code)">打开构建结果 ↗</button>
              <div class="links"><a :href="built.url" :download="plan.name+'.glb'">下载 GLB</a><a :href="built.report_url" download="report.json">下载报告</a></div></template>
          </template>
        </div>
        <div v-if="tab === 'agent'" class="panel-content"><div class="section-title"><h2>视觉模型草案</h2><span class="count">{{ planner ? '已配置' : '待配置' }}</span></div>
          <p class="help">{{ planner ? '生成时将当前参考图与要求发送至已配置的视觉模型，返回的轮廓可继续编辑。' : '服务端配置 SCENE_PLANNER_URL 和 SCENE_PLANNER_MODEL 后，可从参考图生成建筑轮廓草案。' }}</p>
          <label>建模要求<textarea v-model="instruction" rows="3" maxlength="3000"/></label>
          <button class="full" :disabled="!planner || !!working || !instruction.trim()" @click="propose">{{ working || '生成轮廓草案' }}</button>
          <div v-if="proposed" class="proposal"><p>模型返回 {{ proposed.buildings.length }} 栋建筑。应用后替换当前建筑轮廓。</p><button @click="applyProposal">应用建筑草案</button></div>
          <div class="links"><a href="/api/builder/schema" target="_blank" rel="noopener">方案 JSON Schema ↗</a><a href="/api/builder/tools" target="_blank" rel="noopener">工具接口 ↗</a></div>
        </div>
        <div class="plan-files"><button :disabled="!!working" @click="importInput?.click()">导入方案</button><button @click="downloadPlan">保存方案 JSON ↓</button></div>
      </aside>
    </div>
    <p v-else-if="!error" role="status" class="help">正在打开构建工作台…</p>
  </main>
</template>

<style scoped src="./construction.css"></style>
