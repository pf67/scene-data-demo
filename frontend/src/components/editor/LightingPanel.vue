<script setup lang="ts">
import type { SceneLightingState } from "../../api/scene3d";
import type { LightingMode, WorkspaceRoute } from "../../types/workspace";

defineProps<{
  route: WorkspaceRoute;
  lighting: SceneLightingState;
  lightingMode: LightingMode;
}>();

const emit = defineEmits<{
  preset: [preset: "day" | "sunset" | "overcast"];
  updateLighting: [patch: Partial<SceneLightingState>];
  setLightingMode: [mode: LightingMode];
}>();

function onRange(key: keyof SceneLightingState, event: Event): void {
  const val = (event.target as HTMLInputElement).valueAsNumber;
  if (Number.isFinite(val)) emit("updateLighting", { [key]: val });
}
</script>

<template>
  <section class="light-block">
    <h3 class="mono">实时重光照</h3>
    <p class="hint">
      {{
        route === "calibrated" && lightingMode === "match_object"
          ? "当前保留参考照片原光影，太阳方向仅作用于新放入的装备与阴影。"
          : route === "3dgs" && lightingMode === "splat_hybrid"
            ? "3DGS 高斯背景保持原采光，放置装备按太阳方向投射实时阴影。"
            : "物理太阳、天光 IBL 与地面接触阴影随以下参数实时联调。"
      }}
    </p>

    <div class="mode-row">
      <button type="button" @click="emit('preset', 'day')">晴昼</button>
      <button type="button" @click="emit('preset', 'sunset')">日落</button>
      <button type="button" @click="emit('preset', 'overcast')">阴天</button>
    </div>

    <div v-if="route === 'calibrated'" class="mode-row two" role="group" aria-label="标定光照策略">
      <button
        type="button"
        :class="{ active: lightingMode === 'match_object' }"
        @click="emit('setLightingMode', 'match_object')"
      >
        保留原图光影
      </button>
      <button
        type="button"
        :class="{ active: lightingMode === 'relight_all' }"
        @click="emit('setLightingMode', 'relight_all')"
      >
        全场景重光照
      </button>
    </div>

    <div class="sliders">
      <label class="slider-row">
        <span>太阳方位</span>
        <input :value="lighting.azimuth" type="range" min="0" max="360" @input="onRange('azimuth', $event)" />
        <em class="mono">{{ lighting.azimuth }}°</em>
      </label>
      <label class="slider-row">
        <span>太阳高度</span>
        <input :value="lighting.elevation" type="range" min="5" max="85" @input="onRange('elevation', $event)" />
        <em class="mono">{{ lighting.elevation }}°</em>
      </label>
      <label class="slider-row">
        <span>光照强度</span>
        <input :value="lighting.intensity" type="range" min="0.2" max="6" step="0.1" @input="onRange('intensity', $event)" />
        <em class="mono">{{ lighting.intensity.toFixed(1) }}</em>
      </label>
      <label class="slider-row">
        <span>环境补光</span>
        <input :value="lighting.ambient" type="range" min="0" max="1.5" step="0.05" @input="onRange('ambient', $event)" />
        <em class="mono">{{ lighting.ambient.toFixed(2) }}</em>
      </label>
      <label class="slider-row">
        <span>太阳色温</span>
        <input :value="lighting.temperature" type="range" min="2800" max="9000" step="100" @input="onRange('temperature', $event)" />
        <em class="mono">{{ lighting.temperature }}K</em>
      </label>
    </div>
  </section>
</template>

<style scoped>
.light-block {
  display: grid;
  gap: 14px;
}

.light-block h3 {
  margin: 0;
  font-size: 10px;
  letter-spacing: 0.14em;
  color: var(--text-muted);
}

.hint {
  margin: 0;
  font-size: 11.5px;
  line-height: 1.55;
  color: var(--text-muted);
}

.mode-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}

.mode-row.two {
  grid-template-columns: repeat(2, 1fr);
}

.mode-row button {
  padding: 7px 6px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
}

.mode-row button:hover {
  border-color: var(--amber);
  color: var(--text-primary);
}

.mode-row button.active {
  border-color: var(--signal);
  background: var(--signal);
  color: var(--signal-ink);
  font-weight: 600;
}

.sliders {
  display: grid;
  gap: 12px;
  padding-top: 4px;
}

.slider-row {
  display: grid;
  grid-template-columns: 62px minmax(0, 1fr) 48px;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-secondary);
}

.slider-row input {
  width: 100%;
  accent-color: var(--signal);
}

.slider-row em {
  font-style: normal;
  font-size: 11px;
  text-align: right;
  color: var(--text-muted);
}
</style>
