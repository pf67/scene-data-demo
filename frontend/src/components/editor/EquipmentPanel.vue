<script setup lang="ts">
import { Plus } from "../icons";
import type { Asset3D } from "../../api/scene3d";

defineProps<{
  assets: Asset3D[];
  loading: boolean;
  error: string;
  clickPlaceMode: boolean;
  pendingPlaceCode: string;
}>();

const emit = defineEmits<{
  dragAsset: [event: DragEvent, asset: Asset3D];
  dragEnd: [];
  toggleClickPlace: [asset: Asset3D];
  cancelClickPlace: [];
}>();
</script>

<template>
  <section class="equip-block">
    <div class="equip-head">
      <h3 class="mono">装备库 ({{ assets.length }})</h3>
      <button
        v-if="clickPlaceMode"
        type="button"
        class="cancel-place"
        @click="emit('cancelClickPlace')"
      >
        退出点放
      </button>
    </div>

    <p v-if="clickPlaceMode" class="place-banner">
      点击左侧路面、院落或屋顶表面放置已选装备（点击上方退出点放）
    </p>
    <p v-else class="hint">拖入场景表面，或点击右侧「点放」按钮放置。</p>

    <p v-if="loading" class="hint">加载中…</p>
    <p v-else-if="error" class="hint warn">{{ error }}</p>
    <ul v-else class="asset-list">
      <li v-for="asset in assets" :key="asset.code">
        <button
          type="button"
          class="asset-row"
          :class="{ pending: clickPlaceMode && pendingPlaceCode === asset.code }"
          draggable="true"
          :disabled="!asset.has_model && asset.category === 'missing'"
          @dragstart="emit('dragAsset', $event, asset)"
          @dragend="emit('dragEnd')"
          @click="emit('toggleClickPlace', asset)"
        >
          <span class="asset-name">{{ asset.label }}</span>
          <span class="mono asset-meta">
            {{ asset.dimensions.length }}×{{ asset.dimensions.width }}×{{ asset.dimensions.height }}m
            <em v-if="!asset.has_model"> · 占位</em>
          </span>
          <Plus :size="14" />
        </button>
        <button
          type="button"
          class="place-btn"
          :class="{ active: clickPlaceMode && pendingPlaceCode === asset.code }"
          title="点击场景表面放置"
          @click="emit('toggleClickPlace', asset)"
        >
          点放
        </button>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.equip-block {
  display: grid;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
}

.equip-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.equip-head h3 {
  margin: 0;
  font-size: 10px;
  letter-spacing: 0.14em;
  color: var(--text-muted);
}

.cancel-place {
  padding: 3px 8px;
  border: 1px solid var(--amber);
  border-radius: 4px;
  background: rgba(243, 191, 79, 0.14);
  color: var(--amber);
  font-size: 11px;
  cursor: pointer;
}

.place-banner {
  margin: 0;
  padding: 7px 9px;
  border-radius: 6px;
  border-left: 3px solid var(--signal);
  background: rgba(201, 255, 87, 0.1);
  color: var(--signal);
  font-size: 11.5px;
  line-height: 1.5;
}

.hint {
  margin: 0;
  font-size: 11.5px;
  line-height: 1.5;
  color: var(--text-muted);
}

.hint.warn {
  color: var(--warning);
}

.asset-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin: 0;
  padding: 0;
  list-style: none;
  max-height: 260px;
  overflow: auto;
}

.asset-list li {
  display: flex;
  gap: 6px;
}

.asset-row {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-primary);
  font-size: 12.5px;
  cursor: grab;
  text-align: left;
}

.asset-row:hover,
.asset-row.pending {
  border-color: var(--signal);
  background: rgba(201, 255, 87, 0.08);
}

.asset-row:active {
  cursor: grabbing;
}

.asset-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.asset-meta {
  font-size: 10px;
  color: var(--text-muted);
  white-space: nowrap;
}

.asset-meta em {
  font-style: normal;
  color: var(--warning);
}

.place-btn {
  flex-shrink: 0;
  padding: 6px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 11px;
  cursor: pointer;
}

.place-btn:hover,
.place-btn.active {
  border-color: var(--signal);
  background: var(--signal);
  color: var(--signal-ink);
  font-weight: 600;
}
</style>
