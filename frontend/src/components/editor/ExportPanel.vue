<script setup lang="ts">
import { Download, Save, Sparkles } from "../icons";
import type { SceneSessionSummary } from "../../api/scene3d";

defineProps<{
  standalone: boolean;
  sessionTitle: string;
  sessionId: string;
  sessions: SceneSessionSummary[];
  sessionBusy: boolean;
  sessionMessage: string;
  exportViewCount: number;
  exportResolution: number;
  exporting: boolean;
  exportMessage: string;
  placedCount: number;
  downloads: Array<{ task_id: string; frames: number; url: string }>;
  batchBusy: boolean;
  batchMessage: string;
  trainingSets: Array<{ name: string; frames?: number; has_masks?: boolean }>;
}>();

const emit = defineEmits<{
  "update:sessionTitle": [title: string];
  saveSession: [];
  newSession: [];
  openSession: [id: string];
  snapshot: [];
  openRedraw: [];
  "update:exportViewCount": [count: number];
  "update:exportResolution": [res: number];
  runExport: [];
  runBatch: [];
}>();

function onSessionChange(event: Event): void {
  const id = (event.target as HTMLSelectElement).value;
  if (id) emit("openSession", id);
}
</script>

<template>
  <div class="export-stack">
    <section class="card">
      <h3 class="mono">场景会话</h3>
      <label class="field-row">
        <span>名称</span>
        <input
          :value="sessionTitle"
          type="text"
          class="text-input"
          @input="emit('update:sessionTitle', ($event.target as HTMLInputElement).value)"
        />
      </label>
      <div class="btn-row">
        <button type="button" class="btn" :disabled="sessionBusy" @click="emit('saveSession')">
          <Save :size="14" /> 保存
        </button>
        <button type="button" class="btn" :disabled="sessionBusy" @click="emit('newSession')">
          新建空白
        </button>
      </div>
      <select
        class="select-input"
        :disabled="sessionBusy || !sessions.length"
        :value="sessionId"
        @change="onSessionChange"
      >
        <option value="" disabled>选择已保存会话 ({{ sessions.length }})</option>
        <option v-for="item in sessions" :key="item.id" :value="item.id">
          {{ item.title }}（{{ item.object_count }} 个对象）
        </option>
      </select>
      <p v-if="sessionMessage" class="hint mono">{{ sessionMessage }}</p>
    </section>

    <section class="card">
      <h3 class="mono">画面与数据集导出</h3>
      <button type="button" class="btn full" @click="emit('snapshot')">
        <Download :size="14" /> 下载当前视角 PNG 截图
      </button>
      <button type="button" class="btn full redraw-accent" @click="emit('openRedraw')">
        <Sparkles :size="14" /> 截图并渲染（MiniMax-H3-Image）
      </button>
      <div class="divider" />
      <label class="slider-row">
        <span>环绕视角数</span>
        <input
          :value="exportViewCount"
          type="range"
          min="4"
          max="24"
          step="1"
          @input="emit('update:exportViewCount', ($event.target as HTMLInputElement).valueAsNumber)"
        />
        <em class="mono">{{ exportViewCount }}</em>
      </label>
      <label class="slider-row">
        <span>输出分辨率</span>
        <select
          class="select-input"
          :value="exportResolution"
          @change="emit('update:exportResolution', Number(($event.target as HTMLSelectElement).value))"
        >
          <option :value="512">512 × 512</option>
          <option :value="768">768 × 768</option>
          <option :value="1024">1024 × 1024</option>
          <option :value="1536">1536 × 1536</option>
        </select>
      </label>
      <button
        type="button"
        class="export-primary"
        :disabled="exporting || !placedCount"
        @click="emit('runExport')"
      >
        <Download :size="15" />
        {{ exporting ? "导出中…" : "导出多角度 PNG + VOC ZIP" }}
      </button>
      <p v-if="!placedCount" class="hint">请先在场景中放置至少 1 个装备后解锁多角度标注导出。</p>
      <p v-if="exportMessage" class="hint mono">{{ exportMessage }}</p>
    </section>

    <section class="card">
      <h3 class="mono">已导出压缩包 ({{ downloads.length }})</h3>
      <ul v-if="downloads.length" class="dl-list">
        <li v-for="entry in downloads" :key="entry.task_id">
          <a :href="entry.url">{{ entry.task_id.slice(0, 22) }}… · {{ entry.frames }} 帧 ZIP ↓</a>
        </li>
      </ul>
      <p v-else class="hint">暂无历史导出记录，导出后的 ZIP 将列在此处。</p>
    </section>

    <section v-if="!standalone" class="card">
      <h3 class="mono">三期 · 批量训练集</h3>
      <p class="hint">按「场景 + 摆放 + 光照」配方批量出 PNG + VOC + mask + index</p>
      <button type="button" class="export-primary" :disabled="batchBusy" @click="emit('runBatch')">
        {{ batchBusy ? "批量生产中…" : "批量出训练集" }}
      </button>
      <p v-if="batchMessage" class="hint mono">{{ batchMessage }}</p>
      <ul v-if="trainingSets.length" class="dl-list">
        <li v-for="item in trainingSets" :key="item.name">
          <span>{{ item.name }}（{{ item.frames || 0 }} 帧）</span>
        </li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.export-stack {
  display: grid;
  gap: 16px;
}

.card {
  display: grid;
  gap: 10px;
  padding-bottom: 14px;
  border-bottom: 1px solid var(--line);
}

.card:last-child {
  border-bottom: 0;
  padding-bottom: 0;
}

.card h3 {
  margin: 0;
  font-size: 10px;
  letter-spacing: 0.14em;
  color: var(--text-muted);
}

.field-row {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: var(--text-secondary);
}

.text-input,
.select-input {
  flex: 1;
  min-width: 0;
  width: 100%;
  padding: 7px 9px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-primary);
  font-size: 12px;
}

.btn-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 7px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-primary);
  font-size: 12px;
  cursor: pointer;
}

.btn.full {
  width: 100%;
}

.btn.redraw-accent {
  border-color: rgba(201, 255, 87, 0.5);
  background: rgba(201, 255, 87, 0.12);
  color: var(--signal);
  font-weight: 600;
}

.btn:hover:enabled {
  border-color: var(--signal);
  color: var(--signal);
}

.divider {
  border-top: 1px solid var(--line);
  margin: 2px 0;
}

.slider-row {
  display: grid;
  grid-template-columns: 74px minmax(0, 1fr) auto;
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
  color: var(--text-muted);
}

.export-primary {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  width: 100%;
  padding: 9px 12px;
  border: 1px solid var(--signal);
  border-radius: var(--radius-sm);
  background: var(--signal);
  color: var(--signal-ink);
  font-size: 12.5px;
  font-weight: 700;
  cursor: pointer;
}

.export-primary:disabled {
  opacity: 0.42;
  cursor: not-allowed;
}

.hint {
  margin: 0;
  font-size: 11.5px;
  line-height: 1.5;
  color: var(--text-muted);
}

.dl-list {
  display: grid;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
  max-height: 140px;
  overflow: auto;
}

.dl-list a {
  font-size: 11.5px;
  color: var(--signal);
  text-decoration: none;
}

.dl-list a:hover {
  text-decoration: underline;
}
</style>
