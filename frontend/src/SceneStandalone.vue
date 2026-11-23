<script setup lang="ts">
import { computed, defineAsyncComponent, onMounted, ref } from "vue";
import { getJson } from "./api/http";
import { fetchSceneMeshes } from "./api/scene3d";
import {
  classifySceneMeshes,
  resolveInitialRoute,
  ROUTE_META,
  type CalibratedSubView,
  type WorkspaceRoute,
} from "./types/workspace";
import ConstructionStudio from "./components/ConstructionStudio.vue";
import ExperimentalRoutesMenu from "./components/editor/ExperimentalRoutesMenu.vue";
import { Star } from "./components/icons";

const SceneEditor = defineAsyncComponent(() => import("./components/SceneEditor.vue"));

const query = new URLSearchParams(location.search);
const resolved = resolveInitialRoute(location.search);

const route = ref<WorkspaceRoute>(resolved.route);
const calibratedSubView = ref<CalibratedSubView>(resolved.calibratedSubView);
const initialScene = ref(query.get("scene") || "");
const initialView = ref(query.get("view") || undefined);
const requestedPanel = query.get("panel");
const initialPanel = ref<"ground" | "orbit" | undefined>(
  requestedPanel === "ground" || requestedPanel === "orbit" ? requestedPanel : undefined,
);

const downloads = ref<Array<{ task_id: string; frames: number; url: string }>>([]);
const latestExport = ref("");
const routeCounts = ref({ calibrated: 0, splat: 0 });

const show2DStudio = computed(() => route.value === "calibrated" && calibratedSubView.value === "2d");
const show3DEditor = computed(() => !show2DStudio.value);
const editorMounted = ref(show3DEditor.value);
if (show3DEditor.value) editorMounted.value = true;

function syncUrl(nextRoute: WorkspaceRoute, subView: CalibratedSubView): void {
  const url = new URL(location.href);
  url.searchParams.delete("mode");
  url.searchParams.delete("view3d");
  if (nextRoute === "blender") {
    url.searchParams.delete("route");
    url.searchParams.delete("scene");
    url.searchParams.delete("panel");
    url.searchParams.delete("view");
  } else {
    url.searchParams.set("route", nextRoute);
    if (nextRoute === "calibrated" && subView === "3d") {
      url.searchParams.set("view3d", "1");
    }
  }
  history.replaceState(null, "", url.pathname + url.search);
}

function setRoute(nextRoute: WorkspaceRoute, subView?: CalibratedSubView): void {
  if (nextRoute !== "calibrated") {
    initialScene.value = "";
    initialPanel.value = undefined;
    initialView.value = undefined;
  }
  route.value = nextRoute;
  calibratedSubView.value = subView ?? "2d";
  if (nextRoute !== "calibrated" || calibratedSubView.value === "3d") {
    editorMounted.value = true;
  }
  syncUrl(nextRoute, calibratedSubView.value);
}

function setCalibratedSubView(subView: CalibratedSubView): void {
  calibratedSubView.value = subView;
  if (subView === "3d") {
    editorMounted.value = true;
  }
  syncUrl("calibrated", subView);
}

function openCalibrated3D(code?: string, panel?: "ground" | "orbit"): void {
  initialView.value = undefined;
  initialScene.value = code || "";
  initialPanel.value = panel;
  setRoute("calibrated", "3d");
}

async function refreshExports(): Promise<void> {
  try {
    const result = await getJson<{ data: { exports: typeof downloads.value } }>("/api/scene/exports");
    downloads.value = result.data.exports;
  } catch {
    downloads.value = [];
  }
}

async function refreshCounts(): Promise<void> {
  try {
    const groups = classifySceneMeshes(await fetchSceneMeshes());
    routeCounts.value = { calibrated: groups.calibrated.length, splat: groups.splat.length };
  } catch {
    routeCounts.value = { calibrated: 0, splat: 0 };
  }
}

async function onExported(id: string): Promise<void> {
  latestExport.value = id;
  await refreshExports();
}

onMounted(() => {
  void refreshExports();
  void refreshCounts();
});
</script>

<template>
  <template v-if="show2DStudio">
    <header class="calibrated-topbar">
      <div class="bar-left">
        <span class="bar-mark"><Star :size="16" /></span>
        <div class="bar-title">
          <span class="mono">3D SCENE WORKSPACE</span>
          <strong>3D 场景编辑器</strong>
        </div>
        <span class="route-pill exp">{{ ROUTE_META.calibrated.badge }}</span>
      </div>
      <div class="bar-right">
        <div class="subview-switch" role="group" aria-label="照片标定视图">
          <button type="button" class="active">2D 照片标定</button>
          <button type="button" @click="setCalibratedSubView('3d')">3D 标定预览</button>
        </div>
        <button type="button" class="back-btn" @click="setRoute('blender')">
          ← 返回 Blender 主工作台
        </button>
        <ExperimentalRoutesMenu
          :active-route="route"
          :counts="routeCounts"
          @select="setRoute($event, '2d')"
        />
      </div>
    </header>
    <ConstructionStudio embedded @open="openCalibrated3D" />
  </template>

  <SceneEditor
    v-if="editorMounted"
    :visible="show3DEditor"
    :route="route"
    :initial-scene-code="initialScene"
    :initial-panel="initialPanel"
    :initial-view="initialView"
    :downloads="downloads"
    standalone
    @close="setRoute('blender')"
    @change-route="setRoute($event, '2d')"
    @open-calibrated2d="setCalibratedSubView('2d')"
    @exported="onExported"
  />

  <a v-if="latestExport" class="export-toast" :href="`/api/scene/exports/${latestExport}/download`">
    导出完成，下载 PNG + VOC →
  </a>
</template>

<style scoped>
.calibrated-topbar {
  position: sticky;
  top: 0;
  z-index: 40;
  height: 54px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 0 16px;
  border-bottom: 1px solid var(--line);
  background: var(--ink-raised);
}

.bar-left,
.bar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.bar-mark {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border-radius: 8px;
  background: var(--signal);
  color: var(--signal-ink);
  flex-shrink: 0;
}

.bar-title span.mono {
  display: block;
  font-size: 9px;
  letter-spacing: 0.14em;
  color: var(--text-muted);
}

.bar-title strong {
  display: block;
  font-size: 15px;
  font-weight: 650;
  line-height: 1.2;
}

.route-pill.exp {
  padding: 3px 9px;
  border-radius: 99px;
  font-size: 11px;
  font-weight: 600;
  background: rgba(243, 191, 79, 0.14);
  color: var(--amber);
  border: 1px solid rgba(243, 191, 79, 0.38);
  white-space: nowrap;
}

.subview-switch {
  display: inline-flex;
  padding: 2px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  background: var(--ink);
}

.subview-switch button {
  padding: 4px 10px;
  border: 0;
  border-radius: 4px;
  background: transparent;
  color: var(--text-muted);
  font-size: 12px;
  cursor: pointer;
}

.subview-switch button.active {
  background: var(--signal);
  color: var(--signal-ink);
  font-weight: 650;
}

.back-btn {
  padding: 6px 10px;
  border: 1px solid rgba(201, 255, 87, 0.4);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--signal);
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
}

.export-toast {
  position: fixed;
  z-index: 2000;
  bottom: 18px;
  left: 20px;
  padding: 12px 18px;
  background: var(--ink);
  border: 1px solid var(--signal);
  border-radius: 8px;
  color: var(--signal);
  font-size: 13px;
  text-decoration: none;
}
</style>
