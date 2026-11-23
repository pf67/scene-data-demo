<script setup lang="ts">
import type { TransformMode } from "../../types/workspace";

defineProps<{
  placed: Array<{ id: string; label: string; kind: "equipment" | "scene_prop" }>;
  selectedId: string;
  transformMode: TransformMode;
  inspector: { x: number; y: number; z: number; yaw: number; scale: number };
  groundAlign: boolean;
}>();

const emit = defineEmits<{
  select: [id: string];
  deselect: [];
  setMode: [mode: TransformMode];
  updateField: [field: "x" | "y" | "z" | "yaw" | "scale", value: number];
  snap: [];
  "update:groundAlign": [value: boolean];
  remove: [];
}>();

const modeLabels: Record<TransformMode, string> = {
  translate: "平移",
  rotate: "旋转",
  scale: "缩放",
};

function onNumChange(field: "x" | "y" | "z" | "yaw" | "scale", event: Event): void {
  const value = (event.target as HTMLInputElement).valueAsNumber;
  if (Number.isFinite(value)) emit("updateField", field, value);
}
</script>

<template>
  <section class="transform-block">
    <h3 class="mono">已放置对象 ({{ placed.length }})</h3>
    <ul v-if="placed.length" class="placed-list">
      <li v-for="item in placed" :key="item.id">
        <button
          type="button"
          class="placed-row"
          :class="{ active: item.id === selectedId }"
          @click="emit('select', item.id)"
        >
          <span>{{ item.label }}</span>
          <span class="mono id-tag">{{ item.id.slice(-4) }}</span>
        </button>
      </li>
    </ul>

    <div class="mode-row" role="group" aria-label="变换模式">
      <button
        v-for="mode in (['translate', 'rotate', 'scale'] as TransformMode[])"
        :key="mode"
        type="button"
        :class="{ active: transformMode === mode }"
        @click="emit('setMode', mode)"
      >
        {{ modeLabels[mode] }}
      </button>
    </div>

    <template v-if="selectedId">
      <p class="hint">拖动 3D 视口中的坐标轴，或修改下方精确数值：</p>
      <div class="transform-fields">
        <label>
          X (m)
          <input :value="inspector.x" type="number" step="0.1" @change="onNumChange('x', $event)" />
        </label>
        <label>
          Y (m)
          <input :value="inspector.y" type="number" step="0.1" @change="onNumChange('y', $event)" />
        </label>
        <label>
          Z (m)
          <input :value="inspector.z" type="number" step="0.1" @change="onNumChange('z', $event)" />
        </label>
        <label>
          朝向 (°)
          <input :value="inspector.yaw" type="number" step="5" @change="onNumChange('yaw', $event)" />
        </label>
        <label>
          比例
          <input :value="inspector.scale" type="number" min="0.05" step="0.1" @change="onNumChange('scale', $event)" />
        </label>
      </div>
      <div class="action-pair">
        <button type="button" class="prop-button" @click="emit('snap')">贴合下方表面</button>
        <button type="button" class="prop-button" @click="emit('deselect')">完成调整（隐藏坐标轴）</button>
      </div>
      <label class="check-row">
        <input
          :checked="groundAlign"
          type="checkbox"
          @change="emit('update:groundAlign', ($event.target as HTMLInputElement).checked)"
        />
        <span>移动后自动贴地</span>
      </label>
      <button type="button" class="danger-button" @click="emit('remove')">移除选中对象</button>
    </template>
    <p v-else class="hint">请从「场景与装备」拖入模型，或在 3D 画面中点击已有对象。</p>
  </section>
</template>

<style scoped>
.transform-block {
  display: grid;
  gap: 12px;
}

.transform-block h3 {
  margin: 0;
  font-size: 10px;
  letter-spacing: 0.14em;
  color: var(--text-muted);
}

.placed-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
  max-height: 160px;
  overflow: auto;
}

.placed-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-primary);
  font-size: 12.5px;
  cursor: pointer;
}

.placed-row.active {
  border-color: var(--signal);
  background: rgba(201, 255, 87, 0.1);
}

.id-tag {
  font-size: 10px;
  color: var(--text-muted);
}

.mode-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
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

.mode-row button.active {
  border-color: var(--signal);
  background: var(--signal);
  color: var(--signal-ink);
  font-weight: 600;
}

.transform-fields {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
}

.transform-fields label {
  font-size: 10.5px;
  color: var(--text-muted);
  display: grid;
  gap: 4px;
}

.transform-fields input {
  width: 100%;
  padding: 7px 6px;
  border-radius: 5px;
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--text-primary);
  font-size: 12px;
}

.prop-button {
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-primary);
  font-size: 12px;
  cursor: pointer;
}

.prop-button:hover {
  border-color: var(--signal);
}

.action-pair {
  display: grid;
  grid-template-columns: 1fr 1.25fr;
  gap: 6px;
}

.check-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-secondary);
  cursor: pointer;
}

.check-row input {
  accent-color: var(--signal);
}

.danger-button {
  padding: 8px 10px;
  border: 1px solid rgba(255, 118, 87, 0.45);
  border-radius: var(--radius-sm);
  background: rgba(255, 118, 87, 0.12);
  color: var(--danger);
  font-size: 12px;
  cursor: pointer;
}

.danger-button:hover {
  background: rgba(255, 118, 87, 0.22);
}

.hint {
  margin: 0;
  font-size: 11.5px;
  line-height: 1.5;
  color: var(--text-muted);
}
</style>
