<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from "vue";
import { ROUTE_META, type WorkspaceRoute } from "../../types/workspace";

const props = defineProps<{
  activeRoute: WorkspaceRoute;
  counts: { calibrated: number; splat: number };
}>();

const emit = defineEmits<{
  select: [route: WorkspaceRoute];
}>();

const open = ref(false);
const menuHost = ref<HTMLElement | null>(null);

function choose(route: WorkspaceRoute): void {
  open.value = false;
  emit("select", route);
}

function handleDocumentPointerDown(event: PointerEvent): void {
  if (!open.value) return;
  const target = event.target as Node | null;
  if (menuHost.value && target && !menuHost.value.contains(target)) {
    open.value = false;
  }
}

function handleKeyDown(event: KeyboardEvent): void {
  if (open.value && event.key === "Escape") {
    open.value = false;
  }
}

onMounted(() => {
  document.addEventListener("pointerdown", handleDocumentPointerDown);
  document.addEventListener("keydown", handleKeyDown);
});

onBeforeUnmount(() => {
  document.removeEventListener("pointerdown", handleDocumentPointerDown);
  document.removeEventListener("keydown", handleKeyDown);
});

const experimentalRoutes: Array<{ route: WorkspaceRoute; countText?: () => string }> = [
  { route: "calibrated", countText: () => `${props.counts.calibrated} 个场景` },
  { route: "3dgs", countText: () => `${props.counts.splat} 个场景` },
  { route: "sandbox", countText: () => "自由地坪" },
];
</script>

<template>
  <div ref="menuHost" class="routes-menu">
    <button
      type="button"
      class="menu-trigger"
      :class="{ active: activeRoute !== 'blender' }"
      aria-haspopup="menu"
      :aria-expanded="open"
      @click="open = !open"
    >
      <span>其他实验路线</span>
      <span class="caret">{{ open ? "▴" : "▾" }}</span>
    </button>

    <div v-if="open" class="menu-popover" role="menu" aria-label="3D 场景技术路线">
      <button
        type="button"
        role="menuitemradio"
        :aria-checked="activeRoute === 'blender'"
        class="route-item"
        :class="{ selected: activeRoute === 'blender' }"
        @click="choose('blender')"
      >
        <div class="route-head">
          <strong>{{ ROUTE_META.blender.title }}</strong>
          <span class="pill primary">主工作台</span>
        </div>
        <p>{{ ROUTE_META.blender.description }}</p>
      </button>

      <div class="menu-divider">
        <span class="mono">EXPERIMENTAL ROUTES · 实验路线</span>
      </div>

      <button
        v-for="item in experimentalRoutes"
        :key="item.route"
        type="button"
        role="menuitemradio"
        :aria-checked="activeRoute === item.route"
        class="route-item"
        :class="{ selected: activeRoute === item.route }"
        @click="choose(item.route)"
      >
        <div class="route-head">
          <strong>{{ ROUTE_META[item.route].title }}</strong>
          <span class="pill">{{ item.countText ? item.countText() : "实验" }}</span>
        </div>
        <p>{{ ROUTE_META[item.route].description }}</p>
      </button>
    </div>
  </div>
</template>

<style scoped>
.routes-menu {
  position: relative;
}

.menu-trigger {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 11px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 550;
  cursor: pointer;
  white-space: nowrap;
}

.menu-trigger:hover {
  border-color: var(--amber);
  color: var(--text-primary);
}

.menu-trigger.active {
  border-color: var(--amber);
  background: rgba(243, 191, 79, 0.14);
  color: var(--amber);
}

.caret {
  font-size: 10px;
  opacity: 0.85;
}

.menu-popover {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  z-index: 90;
  width: 316px;
  padding: 8px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-md);
  background: var(--ink-raised);
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.55);
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.menu-divider {
  padding: 6px 8px 2px;
  border-top: 1px solid var(--line);
  margin-top: 2px;
}

.menu-divider span {
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--text-muted);
}

.route-item {
  width: 100%;
  text-align: left;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: 7px;
  background: var(--ink-soft);
  color: var(--text-primary);
  cursor: pointer;
  display: grid;
  gap: 4px;
}

.route-item:hover {
  border-color: var(--line-strong);
  background: #2b322b;
}

.route-item.selected {
  border-color: var(--signal);
  background: rgba(201, 255, 87, 0.08);
}

.route-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.route-head strong {
  font-size: 12.5px;
  font-weight: 600;
}

.route-item p {
  margin: 0;
  font-size: 11px;
  line-height: 1.5;
  color: var(--text-muted);
}

.pill {
  font-size: 10px;
  padding: 2px 7px;
  border-radius: 99px;
  background: rgba(255, 255, 255, 0.08);
  color: var(--text-secondary);
  white-space: nowrap;
}

.pill.primary {
  background: rgba(201, 255, 87, 0.16);
  color: var(--signal);
}
</style>
