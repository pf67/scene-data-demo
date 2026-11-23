<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { getJson } from "../api/http";
import { applyGround, buildPlan, type BuildResult, type GroundSettings, type Plan, type Report } from "../api/builder";
import type { SceneMeshInfo } from "../api/scene3d";

const props = defineProps<{ scene?: SceneMeshInfo; draft?: Plan; previewTarget?: string }>();
const emit = defineEmits<{ result: [result: BuildResult, plan: Plan]; busy: [value: boolean] }>();
const plan = ref<Plan>(), report = ref<Report>();
const surface = ref<"local" | "fields">("local"), margin = ref(25), relief = ref(0);
const mode = ref<"texture" | "coverage">("texture");
const working = ref(""), error = ref(""), upload = ref<HTMLInputElement>();
const ground = computed(() => report.value?.mesh?.ground);
const base = computed(() => ground.value ? props.scene?.ground_url : undefined);
const changed = computed(() => margin.value !== (plan.value?.ground?.marginPercent ?? 25)
  || relief.value !== (plan.value?.ground?.reliefM ?? 0) || surface.value !== (plan.value?.ground?.surface ?? "local"));
const editable = computed(() => !!base.value && !changed.value && !working.value);
const types = [
  { key: "observed", label: "照片覆盖", color: "#71b28e" }, { key: "occluded", label: "几何遮挡", color: "#d49c59" },
  { key: "outside", label: "画面以外", color: "#57658b" }, { key: "unresolved", label: "采样不足", color: "#977792" }
] as const;
function percent(key: typeof types[number]["key"]) {
  const g = ground.value;
  return g ? (100 * g[key] / (g.size[0]*g.size[1])).toFixed(1) + "%" : "—";
}
function setBusy(text: string) { working.value = text; emit("busy", !!text); }
const explain = (e: unknown) => e instanceof Error ? e.message : String(e);
let revision = 0;
onBeforeUnmount(() => { revision++; emit("busy", false); });
watch(() => [props.scene?.code, props.draft] as const, async () => {
  const rev = ++revision, scene = props.scene;
  plan.value = undefined; report.value = undefined; error.value = ""; setBusy("");
  try {
    const [source, summary] = await Promise.all([
      props.draft ?? (scene?.plan_url ? getJson<Plan>(scene.plan_url) : undefined),
      scene?.report_url ? getJson<Report>(scene.report_url) : undefined
    ]);
    if (rev !== revision || !source) return;
    plan.value = source; report.value = summary;
    margin.value = source.ground?.marginPercent ?? 25; relief.value = source.ground?.reliefM ?? 0; surface.value = source.ground?.surface ?? "local";
  } catch (e) { if (rev === revision) error.value = explain(e); }
  finally { if (rev === revision) setBusy(""); }
}, { immediate: true });

async function operate(action: "build" | "import", file?: File) {
  if (!plan.value || working.value) return;
  const rev = revision, source = plan.value, url = base.value;
  error.value = "";
  setBusy(action === "build" ? "正在融合照片并构建地面" : "正在写入地面补图");
  try {
    if (action === "build") {
      const settings: GroundSettings = { ...source.ground, marginPercent: margin.value, reliefM: relief.value, surface: surface.value };
      const updated = { ...source, ground: settings };
      const result = await buildPlan(updated);
      if (rev === revision) emit("result", result, updated);
    } else if (url && action === "import" && file) {
      if (file.size > 16*1024*1024) throw new Error("补图最多 16 MB");
      const result = await applyGround(url, file);
      if (rev === revision) emit("result", result, source);
    }
  } catch (e) { if (rev === revision) error.value = explain(e); }
  finally { if (rev === revision) setBusy(""); }
}
function readFile(event: Event) {
  const input = event.target as HTMLInputElement, file = input.files?.[0];
  input.value = "";
  if (file) void operate("import", file);
}
</script>

<template>
  <div class="terrain">
    <Teleport :to="previewTarget || 'body'" :disabled="!previewTarget">
      <section v-if="base && ground" class="atlas" :class="{ wide: !!previewTarget }" aria-label="地面俯视预览">
        <div class="atlas-head"><strong>地面俯视图</strong><span>{{ ground.size.join(' × ') }} px</span></div>
        <div class="atlas-picture"><img :src="base + '/' + mode" :alt="mode === 'texture' ? '场景地面正射纹理' : '地面照片覆盖和补全分区图'"/></div>
        <div class="atlas-buttons" aria-label="地面预览图层">
          <button type="button" :aria-pressed="mode === 'texture'" @click="mode = 'texture'">地面纹理</button>
          <button type="button" :aria-pressed="mode === 'coverage'" @click="mode = 'coverage'">覆盖分区</button>
        </div>
        <div class="legend"><span v-for="t in types" :key="t.key"><i :style="{background:t.color}"></i>{{ t.label }} <b>{{ percent(t.key) }}</b></span></div>
      </section>
      <p v-else-if="previewTarget" class="atlas-empty">构建地面后，可在这里对照俯视纹理和照片覆盖范围。</p>
    </Teleport>
    <fieldset :disabled="!!working || !plan">
      <legend>地面补全</legend>
      <label>缺失区域材质<select v-model="surface"><option value="local">延续邻近地面</option><option value="fields">田块与乡间道路</option></select></label>
      <div class="dimensions"><label>每侧外扩（%）<input v-model.number="margin" type="number" min="0" max="120" step="1"/></label>
        <label>地形起伏（米）<input v-model.number="relief" type="number" min="0" max="6" step=".1"/></label></div>
      <button class="build-ground" type="button" @click="operate('build')">离线补全并构建</button>
    </fieldset>
    <template v-if="base && ground">
      <div class="ground-files"><a :href="base+'/texture'" download="ground-texture.png">下载纹理</a><a :href="base+'/mask'" download="ground-mask.png">下载补全掩膜</a></div>
      <p class="caption">白色区域可补图，黑色区域保留照片纹理。导入同方向、同宽高比的俯视图即可生成新版场景。</p>
      <input ref="upload" type="file" accept="image/png,image/jpeg" hidden @change="readFile"/>
      <button :disabled="!editable" @click="upload?.click()">导入地面补图</button>
      <p v-if="changed" class="caption">构建后可对新范围导入补图。</p>
      <p v-if="ground.method !== 'local'" class="caption">当前地面来自{{ ground.method === 'imported' ? '导入补图' : ' Qwen-Image-2.1 重绘' }}。</p>
    </template>
    <p v-if="working" role="status" class="caption">{{ working }}…</p>
    <p v-if="error" role="alert" class="ground-error">{{ error }}</p>
  </div>
</template>

<style scoped>
.terrain,.terrain fieldset,.terrain details[open]{display:grid;gap:10px;min-width:0}
.terrain{color:#e4e9dd;font-size:13px}.terrain fieldset{border:0;margin:0;padding:0}
.terrain legend{font-size:16px;font-weight:650;margin-bottom:11px}.terrain label{display:grid;gap:5px;font-size:12px;color:#bdc7b5}
.terrain select,.terrain input:not([hidden]),.terrain textarea,.terrain button,.atlas-buttons button{font:inherit;color:#e2e8db;background:#222920;border:1px solid #485243;border-radius:4px;padding:7px 9px;min-width:0}
.terrain select,.terrain input:not([hidden]),.terrain textarea{width:100%;background:#141913;box-sizing:border-box}.terrain button,.atlas-buttons button,.terrain summary{cursor:pointer}
.terrain .build-ground{background:#bedb81;color:#1b2413;font-weight:700}.terrain button:disabled{opacity:.5;cursor:not-allowed}.dimensions{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.ground-files,.atlas-head,.atlas-buttons,.legend{display:flex;gap:10px;align-items:center}.ground-files a{color:#c5e696;font-size:12px}.caption,.ground-error,.atlas-empty{font-size:12px;line-height:1.65;margin:0;color:#adb8a6}
.terrain details{border-top:1px solid #414a3e;padding-top:11px}.terrain summary{color:#d6e4c8}.terrain details[open] summary{margin-bottom:8px}.ground-error{color:#ffb49c}
.atlas{display:grid;gap:8px;color:#e0e8d9}.atlas-head{justify-content:space-between;font-size:13px}.atlas-head span,.legend b{font:11px/1.5 Consolas,monospace;color:#bcc5b7}
.atlas-picture{background:#141913;display:grid;place-items:center;overflow:hidden;border:1px solid #454e40;border-radius:5px}.atlas img{display:block;width:100%;height:170px;object-fit:contain}
.atlas-buttons{gap:5px}.atlas-buttons button{font-size:12px;flex:1;padding:5px}.atlas-buttons button[aria-pressed=true]{border-color:#93b177;background:#34452d}
.legend{flex-wrap:wrap;gap:4px 12px}.legend span{display:flex;gap:5px;align-items:center;color:#c4cdbe;font-size:11px}.legend i{width:9px;height:9px;border-radius:2px}
.wide{padding:14px 18px;gap:10px}.wide .atlas-picture{background:#0f140f}.wide img{height:clamp(230px,52vh,680px)}.wide .atlas-buttons{width:210px}.atlas-empty{padding:40px 20px}
</style>
