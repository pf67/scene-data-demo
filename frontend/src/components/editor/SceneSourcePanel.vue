<script setup lang="ts">
import { Plus } from "../icons";
import type { SceneMeshInfo } from "../../api/scene3d";
import {
  SCENE_PROPS,
  type LightingMode,
  type LoadedSceneMeta,
  type ScenePropDef,
  type WorkspaceRoute,
} from "../../types/workspace";

defineProps<{
  route: WorkspaceRoute;
  activeMesh?: SceneMeshInfo;
  routeScenes: SceneMeshInfo[];
  loadedMeta: LoadedSceneMeta;
  sceneMeshLoading: boolean;
  splatLoading: boolean;
  hasSplat: boolean;
  splatError: string;
  lightingMode: LightingMode;
  sceneTextureName: string;
  activeSceneCode: string;
}>();

const emit = defineEmits<{
  selectScene: [mesh: SceneMeshInfo];
  openCalibrated2D: [];
  setLightingMode: [mode: LightingMode];
  resetFlatGround: [];
  uploadGroundTexture: [file: File];
  placeProp: [prop: ScenePropDef];
}>();

function onFileChange(event: Event): void {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) emit("uploadGroundTexture", file);
  input.value = "";
}
</script>

<template>
  <section class="source-block">
    <template v-if="route === 'blender'">
      <h3 class="mono">主场景底座 · BLENDER PBR</h3>
      <div class="source-preview">
        <img
          :src="activeMesh?.source_url || loadedMeta.referenceImageUrl || '/scene-demo/village-source.png'"
          :alt="(activeMesh?.label || '村庄精模') + '参考图'"
        />
        <span class="badge-green">BLENDER 2K PBR</span>
      </div>
      <div class="source-meta-head">
        <strong>{{ activeMesh?.label || "村庄 · Blender 精模与多视角视频烘焙" }}</strong>
      </div>
      <div class="tag-row">
        <span class="tag">{{ loadedMeta.buildingCount || 263 }} 栋闭合建筑</span>
        <span class="tag">2K CC0 PBR</span>
        <span class="tag">全场景重光照</span>
      </div>
      <p class="hint">
        {{ loadedMeta.completionNote || "高精度 Blender 4.5 水密建筑网格 + Poly Haven 2K 材质图集 + 沥青路面与透贴树木。" }}
      </p>
      <div v-if="routeScenes.length > 1" class="prop-row" role="group" aria-label="精模场景切换">
        <button
          v-for="mesh in routeScenes"
          :key="mesh.code"
          type="button"
          class="prop-button"
          :class="{ active: activeSceneCode === mesh.code }"
          :disabled="sceneMeshLoading"
          @click="emit('selectScene', mesh)"
        >
          {{ mesh.label }}
        </button>
      </div>
    </template>

    <template v-else-if="route === 'calibrated'">
      <h3 class="mono">实验路线 · 照片标定重建</h3>
      <div class="source-preview">
        <img
          :src="activeMesh?.source_url || '/scene-demo/village-source.png'"
          :alt="(activeMesh?.label || '标定场景') + '参考图'"
        />
        <span class="badge-amber">照片标定</span>
      </div>
      <div class="source-meta-head">
        <strong>{{ activeMesh?.label || "选择标定场景" }}</strong>
        <a
          v-if="activeMesh?.report_url"
          :href="activeMesh.report_url"
          download="report.json"
          class="inline-link"
        >
          几何报告 ↓
        </a>
      </div>
      <div v-if="routeScenes.length" class="prop-row" role="group" aria-label="照片标定场景底座">
        <button
          v-for="mesh in routeScenes"
          :key="mesh.code"
          type="button"
          class="prop-button"
          :class="{ active: activeSceneCode === mesh.code }"
          :disabled="sceneMeshLoading"
          @click="emit('selectScene', mesh)"
        >
          {{ mesh.label }}
        </button>
      </div>
      <p v-else class="hint">暂无照片标定场景，可前往 2D 标定台构建。</p>
      <button type="button" class="secondary-action" @click="emit('openCalibrated2D')">
        ← 切换至 2D 照片标定与构建
      </button>
    </template>

    <template v-else-if="route === '3dgs'">
      <h3 class="mono">实验路线 · 3DGS 高斯混合</h3>
      <p class="hint tight">
        高斯泼溅点云背景（.ply/.splat）与网格阴影代理实时融合，用于对比纯网格 PBR 方案。
      </p>
      <div v-if="routeScenes.length" class="prop-row" role="group" aria-label="3DGS 场景">
        <button
          v-for="mesh in routeScenes"
          :key="mesh.code"
          type="button"
          class="prop-button"
          :class="{ active: activeSceneCode === mesh.code }"
          :disabled="sceneMeshLoading || splatLoading"
          @click="emit('selectScene', mesh)"
        >
          {{ mesh.label }}
        </button>
      </div>
      <div class="mode-row" role="group" aria-label="3DGS 渲染模式">
        <button
          type="button"
          :class="{ active: lightingMode === 'splat_hybrid' }"
          :disabled="!hasSplat || splatLoading"
          @click="emit('setLightingMode', 'splat_hybrid')"
        >
          3DGS 混合显示
        </button>
        <button
          type="button"
          :class="{ active: lightingMode !== 'splat_hybrid' }"
          @click="emit('setLightingMode', 'match_object')"
        >
          仅代理网格
        </button>
      </div>
      <p class="hint mono">
        状态：{{ splatLoading ? "正在加载高斯点云…" : hasSplat ? "3DGS 已就绪" : "未加载点云（已回退代理网格）" }}
      </p>
      <p v-if="splatError" class="hint warn">{{ splatError }}</p>
    </template>

    <template v-else>
      <h3 class="mono">实验路线 · 平地自由沙盘</h3>
      <p class="hint tight">
        基于空白网格地面或自定义俯视贴图，配合基础几何物件快速搭建试验沙盘。
      </p>
      <div class="sandbox-controls">
        <button
          type="button"
          class="prop-button"
          :class="{ active: !activeSceneCode && !sceneTextureName }"
          @click="emit('resetFlatGround')"
        >
          重置空白平地
        </button>
        <label class="file-button">
          <Plus :size="14" />
          <span>上传地面背景贴图</span>
          <input type="file" accept="image/*" hidden @change="onFileChange" />
        </label>
      </div>
      <p v-if="sceneTextureName" class="hint mono">当前地面：{{ sceneTextureName }}</p>
      <div class="sub-label mono">场景基础物件（可与装备混排）</div>
      <div class="prop-row" role="group" aria-label="沙盘基础物件">
        <button
          v-for="prop in SCENE_PROPS"
          :key="prop.code"
          type="button"
          class="prop-button"
          :title="`放置${prop.label}（投射/接收阴影）`"
          @click="emit('placeProp', prop)"
        >
          ＋ {{ prop.label }}
        </button>
      </div>
    </template>
  </section>
</template>

<style scoped>
.source-block {
  display: grid;
  gap: 10px;
}

.source-block h3,
.sub-label {
  margin: 0;
  font-size: 10px;
  letter-spacing: 0.14em;
  color: var(--text-muted);
}

.sub-label {
  margin-top: 6px;
}

.source-preview {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid var(--line);
}

.source-preview img {
  display: block;
  width: 100%;
  height: 144px;
  object-fit: cover;
}

.source-preview span {
  position: absolute;
  left: 8px;
  bottom: 8px;
  font-size: 9px;
  letter-spacing: 0.14em;
  padding: 3px 7px;
  border-radius: 4px;
  font-weight: 600;
}

.badge-green {
  background: rgba(24, 34, 13, 0.88);
  color: var(--signal);
  border: 1px solid rgba(201, 255, 87, 0.45);
}

.badge-amber {
  background: rgba(34, 26, 13, 0.88);
  color: var(--amber);
  border: 1px solid rgba(243, 191, 79, 0.45);
}

.source-meta-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.source-meta-head strong {
  font-size: 13px;
  color: var(--text-primary);
}

.inline-link {
  font-size: 11px;
  color: var(--signal);
  text-decoration: none;
  white-space: nowrap;
}

.tag-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.tag {
  font-size: 11px;
  padding: 3px 8px;
  border-radius: 99px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid var(--line);
  color: var(--text-secondary);
}

.hint {
  margin: 0;
  font-size: 11.5px;
  line-height: 1.55;
  color: var(--text-muted);
}

.hint.tight {
  margin-top: -2px;
}

.hint.warn {
  color: var(--warning);
}

.prop-row {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.prop-button {
  padding: 6px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
}

.prop-button:hover:enabled {
  border-color: var(--amber);
  color: var(--text-primary);
}

.prop-button.active {
  border-color: var(--signal);
  color: var(--signal-ink);
  background: var(--signal);
  font-weight: 600;
}

.secondary-action {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-primary);
  font-size: 12px;
  cursor: pointer;
}

.secondary-action:hover {
  border-color: var(--signal);
  color: var(--signal);
}

.sandbox-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.file-button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px dashed var(--line-strong);
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
}

.file-button:hover {
  border-color: var(--signal);
  color: var(--signal);
}

.mode-row {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px;
}

.mode-row button {
  padding: 7px 8px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
}

.mode-row button.active {
  border-color: var(--signal);
  background: var(--signal);
  color: var(--signal-ink);
  font-weight: 600;
}

.mode-row button:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
</style>
