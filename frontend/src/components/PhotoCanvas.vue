<script setup lang="ts">
import { computed, ref, watch } from "vue";
import { imageUrl, type Building, type Pair, type View } from "../api/builder";

export type Marker = { kind: "ground" | "roof" | "base"; index: number };
const props = defineProps<{
  view: View; buildings: Building[]; selected: string; mode: Marker["kind"];
  marker: Marker | null; locked: boolean;
}>();
const emit = defineEmits<{
  selectBuilding: [id: string]; mark: [point: Pair]; selectPoint: [marker: Marker]; move: [marker: Marker, point: Pair];
}>();
const svg = ref<SVGSVGElement>();
const size = ref<Pair>([1536, 1024]);
const missing = ref(false);
const loaded = ref(false);
const drawingH = computed(() => 1000 * size.value[1] / size.value[0]);
const zoom = ref(1), center = ref<Pair>([.5, .5]);
const box = computed(() => {
  const w = 1 / zoom.value;
  return [(center.value[0] - w/2)*1000, (center.value[1] - w/2)*drawingH.value, w*1000, w*drawingH.value].join(" ");
});
function setCenter(u: number, v: number) {
  const edge = .5 / zoom.value;
  center.value = [u,v].map(x => Math.max(edge,Math.min(1-edge,x))) as Pair;
}
function magnify(next: number, focus: Pair = center.value) {
  const prev = zoom.value;
  zoom.value = Math.min(12, Math.max(1, next));
  setCenter(focus[0] + (center.value[0] - focus[0])*prev/zoom.value, focus[1] + (center.value[1] - focus[1])*prev/zoom.value);
}
const shapes = computed(() => props.buildings.flatMap(b => {
  const obs = b.observations.find(o => o.source === props.view.source);
  return obs?.roof.length ? [{ id: b.id, roof: obs.roof }] : [];
}));
const chosen = computed(() => props.buildings.find(b => b.id === props.selected)?.observations.find(o => o.source === props.view.source));
const points = computed(() => {
  if (props.mode === "ground") return props.view.controls.map((c, index) => ({ uv: c.uv, index, label: String(index + 1), check: c.check, kind: "ground" as const }));
  const roof = (chosen.value?.roof || []).map((uv, index) => ({ uv, index, label: String(index + 1), check: false, kind: "roof" as const }));
  return chosen.value?.base ? [...roof, { uv: chosen.value.base, index: 0, label: "脚", check: false, kind: "base" as const }] : roof;
});
let dragging: Marker | null = null;
let start: Pair | null = null;
let pan: {center: Pair; pixels: number} | null = null;
let generation = 0;
watch(() => props.view.source, id => {
  const token = ++generation;
  magnify(1);
  loaded.value = missing.value = false;
  const photo = new Image();
  photo.onload = () => { if (token === generation) { size.value = [photo.naturalWidth, photo.naturalHeight]; loaded.value = true; } };
  photo.onerror = () => { if (token === generation) missing.value = true; };
  photo.src = imageUrl(id);
}, { immediate: true });
function polygon(points: Pair[]) {
  return points.map(([u, v]) => (u * 1000) + "," + (v * drawingH.value)).join(" ");
}
function locate(e: PointerEvent | WheelEvent): Pair | null {
  const matrix = svg.value?.getScreenCTM();
  if (!matrix) return null;
  const pt = new DOMPoint(e.clientX, e.clientY).matrixTransform(matrix.inverse());
  const u = pt.x / 1000, v = pt.y / drawingH.value;
  if (u < 0 || v < 0 || u > 1 || v > 1) return null;
  return [Number(u.toFixed(7)), Number(v.toFixed(7))];
}
function down(e: PointerEvent, marker?: Marker) {
  if (props.locked || !loaded.value || (e.button !== 0 && e.button !== 1)) return;
  svg.value?.focus({ preventScroll: true });
  svg.value?.setPointerCapture(e.pointerId);
  dragging = marker || null;
  start = [e.clientX, e.clientY];
  if (e.shiftKey || e.button === 1) {
    e.preventDefault();
    pan = {center: [...center.value], pixels: svg.value?.getScreenCTM()?.a || 1};
    dragging = null;
    return;
  }
  if (marker) emit("selectPoint", marker);
}
function move(e: PointerEvent) {
  if (pan && start) {
    setCenter(pan.center[0] - (e.clientX-start[0])/pan.pixels/1000, pan.center[1] - (e.clientY-start[1])/pan.pixels/drawingH.value);
    return;
  }
  if (props.locked || !dragging || !start) return;
  const point = locate(e);
  if (point) emit("move", dragging, point);
}
function up(e: PointerEvent) {
  if (!props.locked && !pan && !dragging && start && Math.hypot(e.clientX - start[0], e.clientY - start[1]) < 5) {
    const point = locate(e);
    if (point) emit("mark", point);
  }
  pan = dragging = start = null;
}
function nudge(e: KeyboardEvent) {
  if (!props.marker || props.locked) return;
  const dirs: Record<string, Pair> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  const dir = dirs[e.key];
  const current = points.value.find(p => p.kind === props.marker?.kind && p.index === props.marker.index);
  if (!dir || !current) return;
  e.preventDefault();
  const point = current.uv.map((v, i) => Math.max(0, Math.min(1, v + dir[i] * (e.shiftKey ? 10 : 1) / size.value[i]))) as Pair;
  emit("move", props.marker, point);
}
</script>

<template>
  <div class="drawing">
    <div v-if="missing" class="photo-error" role="alert">参考图不可用，请在当前服务重新上传对应照片。</div>
    <svg ref="svg" :viewBox="box" tabindex="0" role="group" aria-label="参考图标注画布，用方向键微调选中点"
      @wheel.prevent="magnify(zoom * ($event.deltaY < 0 ? 1.2 : 1/1.2),locate($event) || center)"
      @pointerdown="down($event)" @pointermove="move" @pointerup="up" @pointercancel="pan = dragging = start = null" @keydown="nudge">
      <image v-if="!missing" :href="imageUrl(view.source)" x="0" y="0" width="1000" :height="drawingH"/>
      <g v-if="loaded">
        <polygon v-for="s in shapes" :key="s.id" :points="polygon(s.roof)" :class="['outline', { chosen: s.id === selected && mode !== 'ground', passive: mode !== 'roof' }]"
          @pointerdown.stop="$event.shiftKey || $event.button === 1 ? down($event) : !locked && emit('selectBuilding', s.id)"/>
        <template v-if="mode === 'base' && chosen?.base && chosen.roof.length">
          <polyline class="plumb" :points="polygon([chosen.roof[0], chosen.base])"/>
        </template>
        <g v-for="p in points" :key="p.kind + p.index" :transform="'translate('+p.uv[0]*1000+','+p.uv[1]*drawingH+') scale('+1/zoom+')'"
          :class="['handle', p.kind, {held: marker?.kind === p.kind && marker.index === p.index, check:p.check}]"
          @pointerdown.stop="down($event, { kind: p.kind, index: p.index })">
          <circle r="9" class="hit"/><circle r="4.2"/><text x="7" y="-5">{{ p.label }}</text>
        </g>
      </g>
    </svg>
    <div class="zoom-bar" aria-label="画布缩放"><button aria-label="缩小参考图" :disabled="zoom <= 1" @click="magnify(zoom / 1.5)">−</button>
      <button title="适配整图" aria-label="适配整图" @click="magnify(1)">{{ Math.round(zoom*100) }}%</button><button aria-label="放大参考图" :disabled="zoom >= 12" @click="magnify(zoom * 1.5)">＋</button></div>
  </div>
  <div class="photo-foot"><span>{{ mode === "ground" ? "点地面添加控制点" : mode === "base" ? "点第 1 个檐角正下方的墙脚" : "沿屋檐依次点选角点" }} · Shift 拖动平移</span>
    <span class="mono">{{ loaded ? size.join(" × ") : "…" }} px</span></div>
</template>

<style scoped>
.drawing{position:relative;min-height:270px;height:clamp(350px,55vh,720px);background-color:#101410;background-image:radial-gradient(#343a33 .7px,transparent .9px);background-size:12px 12px}
svg{display:block;width:100%;height:100%;touch-action:none;cursor:crosshair;outline-offset:-3px}
.outline{fill:rgba(240,246,218,.025);stroke:rgba(250,255,229,.57);stroke-width:1;vector-effect:non-scaling-stroke;pointer-events:stroke}
.outline.chosen{fill:#c9ff5728;stroke:#d4ff67;stroke-width:2;pointer-events:none}.outline.passive{pointer-events:none}
.handle{cursor:grab}.handle:active{cursor:grabbing}.handle circle{fill:#d5ff69;stroke:#263318;stroke-width:1.6}
.handle.ground circle{fill:#8fe2ff;stroke:#103749}.handle.check circle,.handle.base circle{fill:#ffd28c}.handle.held circle{stroke:white;stroke-width:2.3}
.handle circle.hit{fill:transparent;stroke:none}.handle text{font:700 12px Consolas,monospace;fill:white;paint-order:stroke;stroke:#161c16;stroke-width:3;pointer-events:none}
.plumb{fill:none;stroke:#ffd28c;stroke-width:2;stroke-dasharray:4 3}
.zoom-bar{position:absolute;bottom:10px;right:12px;display:flex;background:#11180fee;border:1px solid #6f776a;border-radius:4px;overflow:hidden}
.zoom-bar button{color:#f2f4eb;background:none;border:none;min-width:28px;padding:5px 6px;cursor:pointer;font:12px Consolas,monospace}.zoom-bar button:disabled{opacity:.4}.zoom-bar button:hover:enabled{background:#33402e}
.photo-foot{display:flex;justify-content:space-between;gap:10px;padding:12px 16px;border-top:1px solid var(--line);font-size:12px;color:var(--paper-dim)}
.photo-foot .mono{white-space:nowrap}.photo-error{position:absolute;inset:40% 0 auto;text-align:center;color:var(--ember)}
@media(max-width:760px){.drawing{height:clamp(260px,52vw,450px);min-height:0}.photo-foot{padding:9px;font-size:11px}}
</style>
