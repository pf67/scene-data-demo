<script setup lang="ts">
import {
  computed,
  markRaw,
  nextTick,
  onBeforeUnmount,
  reactive,
  ref,
  shallowRef,
  watch,
} from "vue";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { Sky } from "three/examples/jsm/objects/Sky.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { GTAOPass } from "three/examples/jsm/postprocessing/GTAOPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { readPhotoProjectors } from "../photoProjectors";
import {
  createSceneSession,
  exportSceneFrames,
  fetchAssets3D,
  fetchSceneMeshes,
  getSceneSession,
  listSceneSessions,
  listTrainingSets,
  runSceneBatch,
  saveSceneSession,
  type Asset3D,
  type SceneFrame,
  type SceneLightingState,
  type SceneMeshInfo,
  type SceneObjectBBox,
  type SceneObjectState,
  type SceneSessionState,
  type SceneSessionSummary,
} from "../api/scene3d";
import type { BuildResult, OrbitCamera } from "../api/builder";
import {
  classifySceneMeshes,
  DEMO_ASSETS,
  ROUTE_META,
  SCENE_PROPS,
  type InspectorTab,
  type LightingMode,
  type LoadedSceneMeta,
  type ScenePropDef,
  type TransformMode,
  type WorkspaceRoute,
} from "../types/workspace";
import { Download, Sparkles, Star, X } from "./icons";
import GroundPanel from "./GroundPanel.vue";
import OrbitPanel from "./OrbitPanel.vue";
import ExperimentalRoutesMenu from "./editor/ExperimentalRoutesMenu.vue";
import SceneSourcePanel from "./editor/SceneSourcePanel.vue";
import EquipmentPanel from "./editor/EquipmentPanel.vue";
import TransformPanel from "./editor/TransformPanel.vue";
import LightingPanel from "./editor/LightingPanel.vue";
import ExportPanel from "./editor/ExportPanel.vue";
import RedrawModal, { type RedrawCaptureBundle } from "./editor/RedrawModal.vue";

const props = withDefaults(
  defineProps<{
    visible: boolean;
    standalone?: boolean;
    route?: WorkspaceRoute;
    initialSceneCode?: string;
    initialPanel?: "ground" | "orbit";
    initialView?: string;
    downloads?: Array<{ task_id: string; frames: number; url: string }>;
  }>(),
  { visible: false, standalone: false, route: "blender", downloads: () => [] },
);

const emit = defineEmits<{
  close: [];
  exported: [taskId: string];
  changeRoute: [route: WorkspaceRoute];
  openCalibrated2D: [];
}>();

type PlacedItem = {
  id: string;
  code: string;
  label: string;
  kind: "equipment" | "scene_prop";
  object3d: THREE.Object3D;
};

const currentRoute = computed<WorkspaceRoute>(() => props.route || "blender");
const routeInfo = computed(() => ROUTE_META[currentRoute.value]);

const canvasHost = shallowRef<HTMLDivElement | null>(null);
const assets = ref<Asset3D[]>([]);
const assetsLoading = ref(false);
const assetsError = ref("");
const placed = ref<PlacedItem[]>([]);
const selectedId = ref("");
const transformMode = ref<TransformMode>("translate");
const exporting = ref(false);
const exportMessage = ref("");
const exportViewCount = ref(8);
const exportResolution = ref(1024);
const sceneTextureName = ref("");
const sessionId = ref("");
const sessionTitle = ref("未命名场景");
const sessions = ref<SceneSessionSummary[]>([]);
const sessionMessage = ref("");
const sessionBusy = ref(false);
const batchBusy = ref(false);
const batchMessage = ref("");
const trainingSets = ref<
  Array<{ name: string; frames?: number; classes?: string[]; has_masks?: boolean; path: string }>
>([]);
const sceneMeshes = ref<SceneMeshInfo[]>([]);
const activeSceneCode = ref("");
const sceneMeshLoading = ref(false);
const hasSplat = ref(false);
const splatLoading = ref(false);
const splatError = ref("");
const loadedMeta = ref<LoadedSceneMeta>({});

const lighting = reactive<SceneLightingState>({
  azimuth: 52,
  elevation: 26,
  intensity: 4.3,
  ambient: 0.06,
  temperature: 5300,
});
const lightingMode = ref<LightingMode>("relight_all");
const clickPlaceMode = ref(false);
const groundAlign = ref(true);
const pendingPlaceCode = ref("");
const dropActive = ref(false);
const placementBusy = ref(false);
const autoOrbit = ref(false);
const sceneReady = ref(false);
const inspectorTab = ref<InspectorTab>("scene");
const inspector = reactive({ x: 0, y: 0, z: 0, yaw: 0, scale: 1 });
const redrawModalOpen = ref(false);
const redrawCapture = ref<RedrawCaptureBundle | null>(null);

const classifiedScenes = computed(() => classifySceneMeshes(sceneMeshes.value));
const routeScenes = computed<SceneMeshInfo[]>(() => {
  if (currentRoute.value === "blender") return classifiedScenes.value.blender;
  if (currentRoute.value === "calibrated") return classifiedScenes.value.calibrated;
  if (currentRoute.value === "3dgs") return classifiedScenes.value.splat;
  return [];
});
const routeCounts = computed(() => ({
  calibrated: classifiedScenes.value.calibrated.length,
  splat: classifiedScenes.value.splat.length,
}));
const activeMesh = computed(() => sceneMeshes.value.find((m) => m.code === activeSceneCode.value));
const activeSceneLabel = computed(
  () =>
    activeMesh.value?.label ||
    (currentRoute.value === "sandbox" ? "平地自由沙盘" : ROUTE_META[currentRoute.value].shortLabel),
);
const availableTabs = computed<Array<{ key: InspectorTab; label: string }>>(() => {
  const firstLabel =
    currentRoute.value === "3dgs"
      ? "3DGS 与装备"
      : currentRoute.value === "sandbox"
        ? "沙盘与装备"
        : "场景与装备";
  const tabs: Array<{ key: InspectorTab; label: string }> = [{ key: "scene", label: firstLabel }];
  if (currentRoute.value === "calibrated" && activeMesh.value?.plan_url) {
    tabs.push({ key: "ground", label: "地面补全" });
    tabs.push({ key: "orbit", label: "补充视角" });
  }
  tabs.push(
    { key: "transform", label: "变换" },
    { key: "light", label: "光照" },
    { key: "export", label: "保存导出" },
  );
  return tabs;
});

const selectedPlaced = computed(() => placed.value.find((item) => item.id === selectedId.value) ?? null);
const placedSummaries = computed(() =>
  placed.value.map((item) => ({ id: item.id, label: item.label, kind: item.kind })),
);

let renderer: THREE.WebGLRenderer | null = null;
let scene: THREE.Scene | null = null;
let camera: THREE.PerspectiveCamera | null = null;
let orbit: OrbitControls | null = null;
let gizmo: TransformControls | null = null;
let gizmoHelper: THREE.Object3D | null = null;
let sun: THREE.DirectionalLight | null = null;
let ambientLight: THREE.AmbientLight | null = null;
let ground: THREE.Mesh | null = null;
let groundMaterial: THREE.MeshStandardMaterial | null = null;
let sceneBase: THREE.Object3D | null = null;
let sparkMod: typeof import("@sparkjsdev/spark") | null = null;
let sparkRenderer: (THREE.Object3D & { dispose?: () => void }) | null = null;
let sceneSplat: (THREE.Object3D & { initialized?: Promise<unknown>; dispose?: () => void }) | null = null;
let shadowCatcherMat: THREE.ShadowMaterial | null = null;
let sceneLoadSeq = 0;
let splatLoadSeq = 0;
let photoProjectors: Awaited<ReturnType<typeof readPhotoProjectors>> = null;
let resizeObserver: ResizeObserver | null = null;
let animationHandle = 0;
let itemSeq = 0;
let gridHelper: THREE.GridHelper | null = null;
let hemisphere: THREE.HemisphereLight | null = null;
let skyDome: Sky | null = null;
let pmremGenerator: THREE.PMREMGenerator | null = null;
let envRenderTarget: THREE.WebGLRenderTarget | null = null;
let lastEnvKey = "";
let composer: EffectComposer | null = null;
let composerTarget: THREE.WebGLRenderTarget | null = null;
let renderPass: RenderPass | null = null;
let gtaoPass: GTAOPass | null = null;
let outputPass: OutputPass | null = null;
let photoGradePass: ShaderPass | null = null;

class SceneGTAOPass extends GTAOPass {
  render(
    rendererRef: THREE.WebGLRenderer,
    writeBuffer: THREE.WebGLRenderTarget,
    readBuffer: THREE.WebGLRenderTarget,
    deltaTime: number,
    maskActive: boolean,
  ): void {
    const hidden: THREE.Object3D[] = [];
    this.scene.traverse((obj) => {
      if (!obj.visible) return;
      if (
        obj === skyDome ||
        obj === gizmoHelper ||
        obj === gridHelper ||
        obj === sparkRenderer ||
        obj.userData.gtaoExclude === true
      ) {
        obj.visible = false;
        hidden.push(obj);
      }
    });
    try {
      super.render(rendererRef, writeBuffer, readBuffer, deltaTime, maskActive);
    } finally {
      for (const obj of hidden) obj.visible = true;
    }
  }
}

const PHOTO_GRADE_SHADER = {
  uniforms: {
    tDiffuse: { value: null as THREE.Texture | null },
    uResolution: { value: new THREE.Vector2(1600, 950) },
    uSharpness: { value: 0.25 },
    uWarmth: { value: 0.026 },
    uContrast: { value: 1.12 },
    uSaturation: { value: 1.07 },
    uVignette: { value: 0.08 },
  },
  vertexShader: /* glsl */ `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse;
    uniform vec2 uResolution;
    uniform float uSharpness;
    uniform float uWarmth;
    uniform float uContrast;
    uniform float uSaturation;
    uniform float uVignette;
    varying vec2 vUv;

    void main() {
      vec2 texel = 1.0 / max(uResolution, vec2(1.0));
      vec4 base = texture2D(tDiffuse, vUv);
      vec3 c = base.rgb;

      if (uSharpness > 0.001) {
        vec3 n = texture2D(tDiffuse, vUv + vec2(0.0, texel.y)).rgb;
        vec3 s = texture2D(tDiffuse, vUv - vec2(0.0, texel.y)).rgb;
        vec3 e = texture2D(tDiffuse, vUv + vec2(texel.x, 0.0)).rgb;
        vec3 w = texture2D(tDiffuse, vUv - vec2(texel.x, 0.0)).rgb;
        vec3 blur = 0.25 * (n + s + e + w);
        vec3 highFreq = c - blur;
        float diffLuma = abs(dot(highFreq, vec3(0.299, 0.587, 0.114)));
        float clarityWeight = clamp(diffLuma * 8.0, 0.0, 1.0);
        c = clamp(c + highFreq * (uSharpness * (0.5 + 0.5 * clarityWeight)), 0.0, 1.0);
      }

      c = clamp(c * vec3(1.038, 1.002, 0.948), 0.0, 1.0);
      float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
      c = clamp((c - 0.5) * uContrast + 0.5, 0.0, 1.0);
      luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
      c = mix(vec3(luma), c, uSaturation);

      vec3 splitTint = mix(vec3(-0.004, 0.0, 0.008), vec3(uWarmth, uWarmth * 0.45, -uWarmth * 0.55), smoothstep(0.18, 0.88, luma));
      c = clamp(c + splitTint, 0.0, 1.0);

      vec2 centered = vUv - 0.5;
      float vig = 1.0 - dot(centered, centered) * uVignette * 2.2;
      c *= clamp(vig, 0.85, 1.0);

      gl_FragColor = vec4(c, base.a);
    }
  `,
};

function renderFrame(activeCam: THREE.PerspectiveCamera): void {
  if (!scene || !renderer) return;
  const splatMode = lightingMode.value === "splat_hybrid" && !!sceneSplat && hasSplat.value;
  if (splatMode || !composer) {
    renderer.render(scene, activeCam);
    return;
  }
  if (renderPass) renderPass.camera = activeCam;
  if (gtaoPass) gtaoPass.camera = activeCam;
  if (photoGradePass && composerTarget) {
    photoGradePass.uniforms.uResolution.value.set(composerTarget.width, composerTarget.height);
  }
  composer.render();
}

type PhotoCalibration = {
  up: number[];
  size: number[];
  principal: number[];
  focalRatio: number;
  center: THREE.Vector3;
  radius: number;
  bounds: THREE.Box3;
};
let sceneCamera: { position: number[]; target: number[]; fov: number; calibration?: PhotoCalibration } | null = null;
let matchPhoto = false;
let orbitCalibration: OrbitCamera | undefined;
const requestedView = ref<string>();
let cameraRoll = 0;
const worldUp = new THREE.Vector3(0, 1, 0);
const viewForward = new THREE.Vector3(), viewRight = new THREE.Vector3(), viewUp = new THREE.Vector3();

function alignOrbitUp(captureRoll = false): void {
  if (!camera || !orbit || !matchPhoto) return;
  viewForward.subVectors(orbit.target, camera.position).normalize();
  viewRight.crossVectors(viewForward, worldUp);
  if (viewRight.lengthSq() < 1e-12) return;
  viewRight.normalize();
  viewUp.crossVectors(viewRight, viewForward);
  if (captureRoll) cameraRoll = Math.atan2(camera.up.dot(viewRight), camera.up.dot(viewUp));
  camera.up.copy(viewUp).multiplyScalar(Math.cos(cameraRoll)).addScaledVector(viewRight, Math.sin(cameraRoll));
  camera.lookAt(orbit.target);
}

function updateViewportProjection(): void {
  if (!camera) return;
  camera.updateProjectionMatrix();
  const cal = matchPhoto && (orbitCalibration || sceneCamera?.calibration);
  if (!cal) return;
  const ratio = cal.size[0] / cal.size[1];
  const fit = Math.min(1, camera.aspect / ratio);
  const m = camera.projectionMatrix.elements;
  m[0] *= cal.focalRatio * fit;
  m[5] *= fit;
  m[8] = (1 - 2 * cal.principal[0]) * fit * ratio / camera.aspect;
  m[9] = (2 * cal.principal[1] - 1) * fit;
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}

const litMaterials = new WeakMap<THREE.Mesh, THREE.Material | THREE.Material[]>();
const unlitMaterials = new WeakMap<THREE.Mesh, THREE.Material | THREE.Material[]>();
const pointerStart = new THREE.Vector2();
let shadowRadius = 120;

function kelvinToHex(kelvin: number): number {
  const t = Math.min(40000, Math.max(1000, kelvin)) / 100;
  let r: number;
  let g: number;
  let b: number;
  if (t <= 66) {
    r = 255;
    g = 99.4708 * Math.log(t) - 161.1196;
    b = t <= 19 ? 0 : 138.5177 * Math.log(t - 10) - 305.0448;
  } else {
    r = 329.6987 * Math.pow(t - 60, -0.1332);
    g = 288.1222 * Math.pow(t - 60, -0.0755);
    b = 255;
  }
  const clamp = (value: number) => Math.min(255, Math.max(0, Math.round(value)));
  return (clamp(r) << 16) | (clamp(g) << 8) | clamp(b);
}

function shouldCastSceneShadow(name: string): boolean {
  const lower = name.toLowerCase();
  return (
    !lower.includes("ground") &&
    !lower.includes("foundation") &&
    !lower.includes("horizon") &&
    !lower.includes("skirt") &&
    !lower.includes("road_asphalt") &&
    !lower.includes("road_concrete")
  );
}

function updateSkyAndEnvironment(): void {
  if (!scene || !renderer) return;
  const azRad = (lighting.azimuth * Math.PI) / 180;
  const elRad = (lighting.elevation * Math.PI) / 180;
  const sunVec = new THREE.Vector3(
    Math.cos(elRad) * Math.cos(azRad),
    Math.sin(elRad),
    Math.cos(elRad) * Math.sin(azRad),
  ).normalize();

  const horizonHex = lighting.elevation < 28 ? 0xd9cdb8 : lighting.intensity < 1.2 ? 0xb8c1c6 : 0xcbd1cf;
  scene.background = new THREE.Color(horizonHex);
  if (scene.fog instanceof THREE.Fog) scene.fog.color.setHex(horizonHex);

  const configureSkyUniforms = (targetSky: Sky, showDisc: number) => {
    const u = targetSky.material.uniforms;
    if (u.turbidity) u.turbidity.value = lighting.intensity < 1.2 ? 8.5 : lighting.elevation < 28 ? 7.2 : 4.8;
    if (u.rayleigh) u.rayleigh.value = lighting.elevation < 28 ? 1.95 : 1.35;
    if (u.mieCoefficient) u.mieCoefficient.value = 0.006;
    if (u.mieDirectionalG) u.mieDirectionalG.value = 0.84;
    if (u.cloudCoverage) u.cloudCoverage.value = 0.28;
    if (u.cloudDensity) u.cloudDensity.value = 0.28;
    if (u.showSunDisc) u.showSunDisc.value = showDisc;
    if (u.sunPosition) (u.sunPosition.value as THREE.Vector3).copy(sunVec).multiplyScalar(1000);
  };

  if (skyDome) {
    skyDome.visible = lightingMode.value !== "splat_hybrid" || !hasSplat.value;
    configureSkyUniforms(skyDome, 0);
  }

  if (pmremGenerator) {
    const envKey = `${Math.round(lighting.azimuth / 5)}_${Math.round(lighting.elevation / 5)}_${Math.round(lighting.temperature / 300)}_${Math.round(lighting.intensity * 2)}_${lighting.intensity < 1.2 ? 1 : 0}`;
    if (envKey !== lastEnvKey) {
      lastEnvKey = envKey;
      const envScene = new THREE.Scene();
      const envSky = new Sky();
      envSky.scale.setScalar(120);
      configureSkyUniforms(envSky, 1);
      envScene.add(envSky);

      const earthGeo = new THREE.CircleGeometry(100, 24);
      earthGeo.rotateX(-Math.PI / 2);
      const earthMat = new THREE.MeshBasicMaterial({ color: 0x2b251e, side: THREE.DoubleSide });
      const earthDisc = new THREE.Mesh(earthGeo, earthMat);
      earthDisc.position.y = -0.5;
      envScene.add(earthDisc);

      const hdrBoost = lighting.intensity < 1.3 ? 1.8 : 12.0 * (lighting.intensity / 4.0);
      const sunDiscGeo = new THREE.SphereGeometry(7.5, 16, 12);
      const sunDiscMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(kelvinToHex(lighting.temperature)).multiplyScalar(hdrBoost),
      });
      const sunDiscMesh = new THREE.Mesh(sunDiscGeo, sunDiscMat);
      sunDiscMesh.position.copy(sunVec).multiplyScalar(92);
      envScene.add(sunDiscMesh);

      envRenderTarget?.dispose();
      envRenderTarget = pmremGenerator.fromScene(envScene, 0.012);
      scene.environment = envRenderTarget.texture;

      sunDiscGeo.dispose();
      sunDiscMat.dispose();
      envSky.geometry.dispose();
      envSky.material.dispose();
      earthGeo.dispose();
      earthMat.dispose();
    }
    scene.environmentIntensity = lightingMode.value === "match_object" ? 0.14 : lighting.intensity < 1.3 ? 0.55 : 0.92;
  }
}

function applyLighting(): void {
  if (!sun || !ambientLight) return;
  const azimuth = (lighting.azimuth * Math.PI) / 180;
  const elevation = (lighting.elevation * Math.PI) / 180;
  const radius = shadowRadius * 2.5;
  sun.position.set(
    sun.target.position.x + radius * Math.cos(elevation) * Math.cos(azimuth),
    sun.target.position.y + radius * Math.sin(elevation),
    sun.target.position.z + radius * Math.cos(elevation) * Math.sin(azimuth),
  );
  sun.intensity = lighting.intensity;
  sun.color.setHex(kelvinToHex(lighting.temperature));
  const splatMode = lightingMode.value === "splat_hybrid" && !!sceneSplat && hasSplat.value;
  if (sceneSplat) sceneSplat.visible = splatMode;
  if (renderer) {
    renderer.toneMapping = THREE.AgXToneMapping;
    renderer.toneMappingExposure = splatMode ? 1.08 : 0.94;
  }
  updateSkyAndEnvironment();
  const envUnlit = lightingMode.value === "match_object";
  if (gtaoPass) gtaoPass.enabled = !splatMode && !envUnlit;
  if (photoGradePass) photoGradePass.enabled = !splatMode && !envUnlit;
  ambientLight.intensity = lighting.ambient;
  if (hemisphere) hemisphere.intensity = Math.max(0.06, lighting.ambient * 0.85);
  if (sceneBase) {
    const originalColors = sceneBase.userData.global_surface === true || !!photoProjectors;
    sceneBase.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      if (!litMaterials.has(mesh)) litMaterials.set(mesh, mesh.material);
      if (splatMode && shadowCatcherMat) {
        mesh.material = shadowCatcherMat;
        mesh.receiveShadow = true;
        mesh.castShadow = false;
        mesh.renderOrder = 20;
      } else {
        if (envUnlit && !unlitMaterials.has(mesh)) {
          const originals = litMaterials.get(mesh)!;
          let receiver: THREE.Object3D | null = mesh;
          while (receiver && receiver.userData.photo_projectors !== false) receiver = receiver.parent;
          const convert = (mat: THREE.Material) => {
            const source = mat as THREE.MeshStandardMaterial;
            const material = new THREE.MeshBasicMaterial({
              map: source.map || null,
              color: source.color || 0xffffff,
              vertexColors: source.vertexColors,
              side: source.side,
              toneMapped: !originalColors,
            });
            if (!receiver) photoProjectors?.bind(material);
            return material;
          };
          unlitMaterials.set(mesh, Array.isArray(originals) ? originals.map(convert) : convert(originals));
        }
        mesh.material = (envUnlit ? unlitMaterials : litMaterials).get(mesh)!;
        mesh.renderOrder = 0;
        mesh.receiveShadow = !envUnlit;
        mesh.castShadow = shouldCastSceneShadow(mesh.name);
      }
    });
  }
  if (ground) ground.receiveShadow = true;
  for (const item of placed.value) {
    item.object3d.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    });
  }
}

function lightingPreset(preset: "day" | "sunset" | "overcast"): void {
  if (currentRoute.value === "blender" || currentRoute.value === "sandbox") {
    lightingMode.value = "relight_all";
  }
  Object.assign(
    lighting,
    preset === "day"
      ? { azimuth: 52, elevation: 26, intensity: 4.3, ambient: 0.06, temperature: 5300 }
      : preset === "sunset"
        ? { azimuth: 245, elevation: 14, intensity: 4.0, ambient: 0.05, temperature: 3200 }
        : { azimuth: 130, elevation: 62, intensity: 1.0, ambient: 0.42, temperature: 7000 },
  );
}

function syncGizmoVisibility(): void {
  if (!gizmo || !gizmoHelper) return;
  const target = inspectorTab.value === "transform" ? selectedPlaced.value?.object3d ?? null : null;
  if (target) {
    gizmo.attach(target);
    gizmo.enabled = true;
    gizmoHelper.visible = true;
    syncInspector();
  } else {
    gizmo.detach();
    gizmo.enabled = false;
    gizmoHelper.visible = false;
  }
}

function deselectItem(): void {
  selectedId.value = "";
  syncGizmoVisibility();
}

watch(lighting, applyLighting, { deep: true });
watch(lightingMode, applyLighting);
watch([inspectorTab, selectedId], syncGizmoVisibility);

function disposeObject(root: THREE.Object3D): void {
  root.traverse((child) => {
    if (child === sparkRenderer || child === sceneSplat || child.userData.isSparkObject === true) return;
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.geometry?.dispose();
    const materials = new Set(
      [mesh.material, litMaterials.get(mesh), unlitMaterials.get(mesh)].flat().filter(Boolean) as THREE.Material[],
    );
    for (const material of materials) {
      if (material === shadowCatcherMat) continue;
      for (const key of ["map", "normalMap", "roughnessMap", "metalnessMap", "aoMap"] as const) {
        const texture = (material as THREE.MeshStandardMaterial)[key];
        texture?.dispose();
      }
      material.dispose();
    }
  });
}

function clearSceneSplat(): void {
  if (sceneSplat && scene) scene.remove(sceneSplat);
  sceneSplat?.dispose?.();
  sceneSplat = null;
  hasSplat.value = false;
  splatLoading.value = false;
  splatError.value = "";
}

async function ensureSparkRenderer(): Promise<typeof import("@sparkjsdev/spark") | null> {
  if (!scene || !renderer) return null;
  if (!sparkMod) sparkMod = await import("@sparkjsdev/spark");
  if (!sparkRenderer) {
    const inst = markRaw(new sparkMod.SparkRenderer({ renderer }));
    inst.userData.isSparkObject = true;
    scene.add(inst);
    sparkRenderer = inst;
  }
  return sparkMod;
}

function buildGround(): void {
  if (!scene) return;
  sceneLoadSeq += 1;
  splatLoadSeq += 1;
  clearSceneSplat();
  photoProjectors?.dispose();
  photoProjectors = null;
  loadedMeta.value = {};
  if (ground) {
    scene.remove(ground);
    disposeObject(ground);
  }
  if (sceneBase) {
    scene.remove(sceneBase);
    disposeObject(sceneBase);
    sceneBase = null;
  }
  const geometry = new THREE.PlaneGeometry(60, 60);
  geometry.rotateX(-Math.PI / 2);
  groundMaterial = new THREE.MeshStandardMaterial({
    color: 0x2a3128,
    roughness: 0.92,
    metalness: 0.0,
  });
  ground = new THREE.Mesh(geometry, groundMaterial);
  ground.receiveShadow = true;
  ground.name = "ground";
  scene.add(ground);
}

function tuneSceneMaterialPBR(
  mesh: THREE.Mesh,
  mat: THREE.MeshStandardMaterial,
  maxAniso: number,
  isEquipment = false,
): void {
  if (mat.map) {
    mat.map.colorSpace = THREE.SRGBColorSpace;
    mat.map.anisotropy = maxAniso;
  }
  if (mat.normalMap) mat.normalMap.anisotropy = maxAniso;
  if (mat.roughnessMap) mat.roughnessMap.anisotropy = maxAniso;
  if (mat.metalnessMap) mat.metalnessMap.anisotropy = maxAniso;
  if (mat.aoMap) mat.aoMap.anisotropy = maxAniso;

  if (isEquipment) {
    mat.roughness = mat.roughnessMap ? 0.58 : Math.min(mat.roughness, 0.38);
    mat.metalness = mat.metalnessMap ? 1.0 : Math.max(mat.metalness, 0.35);
    mat.envMapIntensity = 1.95;
    mat.needsUpdate = true;
    return;
  }

  if (mat.alphaTest >= 0.35) {
    mat.transparent = false;
    mat.depthWrite = true;
    mat.side = THREE.DoubleSide;
    mat.envMapIntensity = 0.16;
    mat.roughness = Math.max(mat.roughness, 0.84);
    mesh.userData.gtaoExclude = true;
    mat.needsUpdate = true;
    return;
  }

  const lower = (mat.name || "").toLowerCase();
  if (
    lower.includes("glass") ||
    lower.includes("solarcollector") ||
    lower.includes("headlight") ||
    lower.includes("taillight")
  ) {
    mat.roughness = mat.roughnessMap ? 0.16 : Math.min(mat.roughness, 0.12);
    mat.metalness = Math.max(mat.metalness, 0.18);
    mat.envMapIntensity = 2.4;
  } else if (
    lower.includes("car_paint") ||
    lower.includes("roofsilver") ||
    lower.includes("roofblue") ||
    lower.includes("windowframe") ||
    lower.includes("solartank") ||
    lower.includes("shopshutter") ||
    lower.includes("steel") ||
    lower.includes("irongate")
  ) {
    mat.envMapIntensity = 1.8;
    if (lower.includes("roofblue") || lower.includes("roofsilver")) {
      mat.roughness = mat.roughnessMap ? 0.4 : Math.min(mat.roughness, 0.34);
      mat.metalness = Math.max(mat.metalness, 0.45);
    }
  } else {
    mat.envMapIntensity = 0.18;
  }
  mat.needsUpdate = true;
}

/** Load a GLB scene base (and optionally lazy-load 3DGS splats only when in the 3DGS route). */
async function loadSceneMesh(mesh: SceneMeshInfo, keepView = false, loadSplat = currentRoute.value === "3dgs"): Promise<void> {
  if (!scene) return;
  const targetScene = scene;
  const myLoadSeq = ++sceneLoadSeq;
  sceneMeshLoading.value = true;
  exportMessage.value = "";
  try {
    const gltf = await new GLTFLoader().loadAsync(mesh.url);
    const root = gltf.scene;
    const nextProjectors = await readPhotoProjectors(gltf.parser, root.userData.photo_projectors);
    if (scene !== targetScene || myLoadSeq !== sceneLoadSeq) {
      nextProjectors?.dispose();
      disposeObject(root);
      return;
    }
    photoProjectors?.dispose();
    photoProjectors = nextProjectors;
    splatLoadSeq += 1;
    clearSceneSplat();
    if (ground) {
      scene.remove(ground);
      disposeObject(ground);
      ground = null;
      groundMaterial = null;
    }
    if (sceneBase) {
      scene.remove(sceneBase);
      disposeObject(sceneBase);
      sceneBase = null;
    }
    const maxAniso = Math.min(16, renderer?.capabilities.getMaxAnisotropy() || 1);
    root.traverse((child) => {
      const m = child as THREE.Mesh;
      if (m.isMesh) {
        for (const material of Array.isArray(m.material) ? m.material : [m.material]) {
          tuneSceneMaterialPBR(m, material as THREE.MeshStandardMaterial, maxAniso, false);
        }
        m.castShadow = shouldCastSceneShadow(m.name);
        m.receiveShadow = true;
      }
    });

    const box = new THREE.Box3().setFromObject(root);
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const completed = mesh.kind === "photo_completed_village" || root.userData.preserve_scale === true;
    const uniform = completed ? 1 : 80 / maxDim;
    root.scale.setScalar(uniform);
    const scaled = new THREE.Box3().setFromObject(root);
    const center = scaled.getCenter(new THREE.Vector3());
    if (!completed) {
      root.position.x -= center.x;
      root.position.z -= center.z;
      root.position.y -= scaled.min.y;
    }
    root.name = `scene_${mesh.code}`;
    scene.add(root);
    sceneBase = markRaw(root);
    activeSceneCode.value = mesh.code;
    sceneTextureName.value = `scene:${mesh.code}`;
    loadedMeta.value = {
      buildingCount: Number(root.userData.building_count) || undefined,
      referenceImageUrl:
        typeof root.userData.reference_image_url === "string" ? root.userData.reference_image_url : undefined,
      completionNote:
        typeof root.userData.completion_note === "string" ? root.userData.completion_note : undefined,
    };

    sceneCamera = completed
      ? {
          position: root.userData.camera_position || [6.3, 37.6, 87.8],
          target: root.userData.camera_target || [0, 2, -10],
          fov: Number(root.userData.camera_fov) || 40,
        }
      : { position: [45, 45, 55], target: [0, 5, 0], fov: 45 };
    if (mesh.kind === "calibrated_building_scene" && root.userData.reference_size) {
      const meta = root.userData;
      const fit = scaled.clone();
      const footprint = meta.footprint_bounds_xz;
      if (
        Array.isArray(footprint) &&
        footprint.length === 2 &&
        footprint.every((p) => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))
      ) {
        fit.min.x = footprint[0][0];
        fit.min.z = footprint[0][1];
        fit.max.x = footprint[1][0];
        fit.max.z = footprint[1][1];
      }
      sceneCamera.calibration = {
        up: meta.camera_up,
        size: meta.reference_size,
        principal: meta.principal,
        focalRatio: meta.focal_ratio,
        center: fit.getCenter(new THREE.Vector3()),
        bounds: fit,
        radius: Math.max(1, fit.getSize(new THREE.Vector3()).length() / 2),
      };
    }
    photoProjectors?.center.copy(sceneCamera.calibration?.center || new THREE.Vector3().fromArray(sceneCamera.target));
    if (orbit && camera) {
      const radius = sceneCamera.calibration?.radius;
      const distance = new THREE.Vector3().fromArray(sceneCamera.position).distanceTo(center);
      orbit.minDistance = radius ? Math.min(5, radius / 5) : 5;
      const halfFov = THREE.MathUtils.degToRad(sceneCamera.fov / 2);
      orbit.maxDistance = radius
        ? Math.max(650, 8 * radius, 2 * distance, (2 * radius) / (Math.tan(halfFov) * Math.min(1, camera.aspect)))
        : 650;
      camera.far = Math.max(2400, orbit.maxDistance * 2, orbit.maxDistance + size.length());
      if (scene.fog instanceof THREE.Fog) {
        scene.fog.near = radius ? Math.max(175, 3 * radius) : 175;
        scene.fog.far = radius ? Math.max(620, orbit.maxDistance + 2.5 * radius) : 620;
      }
    }
    if (gridHelper) gridHelper.visible = false;

    if (loadSplat && mesh.splat_url) {
      const token = ++splatLoadSeq;
      splatLoading.value = true;
      splatError.value = "";
      try {
        const spark = await ensureSparkRenderer();
        if (!spark || scene !== targetScene || token !== splatLoadSeq || myLoadSeq !== sceneLoadSeq) return;
        const nextSplat = markRaw(new spark.SplatMesh({ url: mesh.splat_url }));
        nextSplat.userData.isSparkObject = true;
        const tf = mesh.splat_transform;
        if (tf?.position?.length === 3) nextSplat.position.fromArray(tf.position);
        if (tf?.rotation?.length === 3) nextSplat.rotation.set(tf.rotation[0], tf.rotation[1], tf.rotation[2]);
        if (tf?.scale?.length === 3) nextSplat.scale.fromArray(tf.scale);
        nextSplat.name = `splat_${mesh.code}`;
        scene.add(nextSplat);
        await nextSplat.initialized;
        if (scene !== targetScene || token !== splatLoadSeq || myLoadSeq !== sceneLoadSeq) {
          targetScene.remove(nextSplat);
          nextSplat.dispose();
          return;
        }
        sceneSplat = nextSplat;
        hasSplat.value = true;
        if (!keepView) lightingMode.value = "splat_hybrid";
      } catch (err) {
        console.warn("3DGS 加载失败，回退至网格底座:", err);
        splatError.value = `3DGS 点云未就绪（已回退代理网格）：${err instanceof Error ? err.message : String(err)}`;
        if (!keepView) {
          lightingMode.value = root.userData.global_surface === true || photoProjectors ? "match_object" : "relight_all";
        }
      } finally {
        if (token === splatLoadSeq) splatLoading.value = false;
      }
    } else if (currentRoute.value === "calibrated") {
      if (!keepView) {
        lightingMode.value = root.userData.global_surface === true || photoProjectors ? "match_object" : "relight_all";
      }
    } else {
      lightingMode.value = "relight_all";
    }

    sceneReady.value = true;
    applyLighting();
    if (!keepView) cameraView("photo");
    fitShadowCamera();
  } catch (error) {
    if (myLoadSeq === sceneLoadSeq) {
      exportMessage.value = `场景底座加载失败：${error instanceof Error ? error.message : String(error)}`;
    }
  } finally {
    if (myLoadSeq === sceneLoadSeq) sceneMeshLoading.value = false;
  }
}

async function applyBuiltScene(result: BuildResult, message: string): Promise<void> {
  sceneMeshes.value = [result.scene, ...sceneMeshes.value.filter((s) => s.code !== result.scene.code)];
  await loadSceneMesh(result.scene, true, false);
  if (activeSceneCode.value === result.scene.code) exportMessage.value = message;
}

function resetFlatGround(): void {
  activeSceneCode.value = "";
  sceneTextureName.value = "";
  buildGround();
  lightingMode.value = "relight_all";
  sceneReady.value = true;
  sceneCamera = null;
  if (scene?.fog instanceof THREE.Fog) {
    scene.fog.near = 260;
    scene.fog.far = 680;
  }
  if (gridHelper) gridHelper.visible = true;
  applyLighting();
  cameraView("photo");
  fitShadowCamera();
}

function stopCameraMotion(): void {
  if (!orbit) return;
  autoOrbit.value = false;
  orbit.autoRotate = false;
  const damped = orbit.enableDamping;
  orbit.enableDamping = false;
  orbit.update();
  orbit.enableDamping = damped;
}

function showOrbitCamera(pose: OrbitCamera): void {
  if (!camera || !orbit) return;
  stopCameraMotion();
  orbitCalibration = pose;
  requestedView.value = undefined;
  matchPhoto = true;
  camera.position.fromArray(pose.position);
  camera.up.fromArray(pose.up);
  camera.fov = pose.fov;
  orbit.target.fromArray(pose.target);
  camera.lookAt(orbit.target);
  alignOrbitUp(true);
  updateViewportProjection();
  orbit.update();
}

function cameraView(view: "photo" | "back" | "top" | "focus"): void {
  if (!camera || !orbit) return;
  stopCameraMotion();
  orbitCalibration = undefined;
  const target = new THREE.Vector3().fromArray(sceneCamera?.target || [0, 1, 0]);
  const cal = sceneCamera?.calibration;
  matchPhoto = view === "photo" && !!cal;
  if (cal && (view === "back" || view === "top")) target.copy(cal.center);
  if (view === "photo") {
    camera.position.fromArray(sceneCamera?.position || [25, 25, 30]);
    camera.fov = sceneCamera?.fov || 45;
  } else if (view === "back" || view === "top") {
    const offset = view === "back" ? new THREE.Vector3(-45, 48, -85) : new THREE.Vector3(0, 155, 0.001);
    if (cal) {
      if (view === "back" && sceneCamera) {
        const front = new THREE.Vector3()
          .fromArray(sceneCamera.position)
          .sub(new THREE.Vector3().fromArray(sceneCamera.target));
        if (Math.hypot(front.x, front.z) > 0.001) offset.set(-front.x, front.y, -front.z);
      }
      const direction = offset.clone().normalize();
      const right = new THREE.Vector3(0, 1, 0).cross(direction).normalize();
      const up = direction.clone().cross(right);
      const tanY = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
      const tanX = tanY * camera.aspect;
      let fit = 1;
      for (const x of [cal.bounds.min.x, cal.bounds.max.x]) {
        for (const y of [cal.bounds.min.y, cal.bounds.max.y]) {
          for (const z of [cal.bounds.min.z, cal.bounds.max.z]) {
            const corner = new THREE.Vector3(x, y, z).sub(target);
            fit = Math.max(fit, corner.dot(direction) + Math.max(Math.abs(corner.dot(right)) / tanX, Math.abs(corner.dot(up)) / tanY));
          }
        }
      }
      offset.setLength(1.12 * fit);
    }
    camera.position.copy(target).add(offset);
  } else if (selectedPlaced.value) {
    const bounds = new THREE.Box3().setFromObject(selectedPlaced.value.object3d);
    bounds.getCenter(target);
    camera.position.copy(target).add(new THREE.Vector3(14, 12, 18));
  }
  camera.up.set(0, 1, 0);
  if (matchPhoto && cal) camera.up.fromArray(cal.up);
  orbit.target.copy(target);
  camera.lookAt(target);
  alignOrbitUp(true);
  updateViewportProjection();
  orbit.update();
}

function snapshot(): void {
  if (!renderer || !scene || !camera) return;
  if (gizmoHelper) gizmoHelper.visible = false;
  try {
    photoProjectors?.update(camera.position);
    renderFrame(camera);
    const link = document.createElement("a");
    link.download = `village-${Date.now()}.png`;
    link.href = renderer.domElement.toDataURL("image/png");
    link.click();
  } finally {
    syncGizmoVisibility();
  }
}

function createMaskReplacement(original: THREE.Material, isWhite: boolean, tempMats: THREE.Material[]): THREE.Material {
  if (isWhite) {
    const mat = new THREE.MeshBasicMaterial({ color: 0xffffff, side: original.side });
    tempMats.push(mat);
    return mat;
  }
  const std = original as THREE.MeshStandardMaterial;
  const useCutout = std.alphaTest > 0 && (std.map || std.alphaMap);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x000000,
    side: original.side,
    ...(useCutout
      ? {
          map: std.map ?? null,
          alphaMap: std.alphaMap ?? null,
          alphaTest: std.alphaTest,
        }
      : {}),
  });
  tempMats.push(mat);
  return mat;
}

function captureRedrawBundle(): RedrawCaptureBundle | null {
  if (!renderer || !scene || !camera) return null;
  if (gizmoHelper) gizmoHelper.visible = false;
  let screenshot = "";
  let environment: string | null = null;
  let protectionMask: string | null = null;
  const equipmentItems = placed.value.filter((item) => item.kind === "equipment");
  try {
    photoProjectors?.update(camera.position);
    renderFrame(camera);
    screenshot = renderer.domElement.toDataURL("image/png");

    if (equipmentItems.length > 0) {
      const prevVis = equipmentItems.map((item) => item.object3d.visible);
      try {
        for (const item of equipmentItems) item.object3d.visible = false;
        renderFrame(camera);
        environment = renderer.domElement.toDataURL("image/png");
      } finally {
        equipmentItems.forEach((item, index) => {
          item.object3d.visible = prevVis[index] ?? true;
        });
      }

      const prevBg = scene.background;
      const prevFog = scene.fog;
      const prevSky = skyDome?.visible ?? false;
      const prevGrid = gridHelper?.visible ?? false;
      const prevSplat = sceneSplat?.visible ?? false;
      const prevTone = renderer.toneMapping;
      const prevExp = renderer.toneMappingExposure;
      const savedMeshes: Array<{
        mesh: THREE.Mesh;
        material: THREE.Material | THREE.Material[];
        visible: boolean;
      }> = [];
      const tempMats: THREE.Material[] = [];

      try {
        const swapGroup = (root: THREE.Object3D | null | undefined, isWhite: boolean) => {
          if (!root) return;
          root.traverse((child) => {
            const m = child as THREE.Mesh;
            if (!m.isMesh) return;
            savedMeshes.push({ mesh: m, material: m.material, visible: m.visible });
            if (!isWhite && m.material === shadowCatcherMat) {
              m.visible = false;
              return;
            }
            m.material = Array.isArray(m.material)
              ? m.material.map((entry) => createMaskReplacement(entry, isWhite, tempMats))
              : createMaskReplacement(m.material, isWhite, tempMats);
          });
        };

        swapGroup(sceneBase, false);
        swapGroup(ground, false);
        for (const item of placed.value) {
          swapGroup(item.object3d, item.kind === "equipment");
        }

        scene.background = new THREE.Color(0x000000);
        scene.fog = null;
        if (skyDome) skyDome.visible = false;
        if (gridHelper) gridHelper.visible = false;
        if (sceneSplat) sceneSplat.visible = false;
        renderer.toneMapping = THREE.NoToneMapping;
        renderer.toneMappingExposure = 1;
        renderer.setRenderTarget(null);
        renderer.render(scene, camera);
        protectionMask = renderer.domElement.toDataURL("image/png");
      } finally {
        for (const entry of savedMeshes) {
          entry.mesh.material = entry.material;
          entry.mesh.visible = entry.visible;
        }
        for (const mat of tempMats) mat.dispose();
        scene.background = prevBg;
        scene.fog = prevFog;
        if (skyDome) skyDome.visible = prevSky;
        if (gridHelper) gridHelper.visible = prevGrid;
        if (sceneSplat) sceneSplat.visible = prevSplat;
        renderer.toneMapping = prevTone;
        renderer.toneMappingExposure = prevExp;
      }
    }

    renderFrame(camera);
    return {
      screenshot,
      environment,
      protectionMask,
      width: renderer.domElement.width,
      height: renderer.domElement.height,
      equipmentCount: equipmentItems.length,
    };
  } finally {
    syncGizmoVisibility();
  }
}

function openRedrawStudio(): void {
  const bundle = captureRedrawBundle();
  if (!bundle) return;
  redrawCapture.value = bundle;
  redrawModalOpen.value = true;
}

async function refreshSceneMeshes(): Promise<void> {
  try {
    sceneMeshes.value = await fetchSceneMeshes();
  } catch {
    sceneMeshes.value = [];
  }
}

async function applySceneTexture(file: File): Promise<void> {
  if (!groundMaterial) return;
  const url = URL.createObjectURL(file);
  try {
    const texture = await new THREE.TextureLoader().loadAsync(url);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = THREE.ClampToEdgeWrapping;
    texture.wrapT = THREE.ClampToEdgeWrapping;
    groundMaterial.map = texture;
    groundMaterial.color.setHex(0xffffff);
    groundMaterial.needsUpdate = true;
    sceneTextureName.value = file.name;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function handleUploadGroundTexture(file: File): void {
  resetFlatGround();
  void applySceneTexture(file);
}

function createPlaceholderMesh(asset: Asset3D, color = 0xb8a0ff): THREE.Object3D {
  const { length, width, height } = asset.dimensions;
  const geometry = new THREE.BoxGeometry(width, height, length);
  const material = new THREE.MeshStandardMaterial({ color, metalness: 0.35, roughness: 0.55 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  const group = new THREE.Group();
  group.add(mesh);
  mesh.position.y = height / 2;
  return group;
}

function createScenePropMesh(width: number, height: number, length: number): THREE.Object3D {
  const geometry = new THREE.BoxGeometry(width, height, length);
  const material = new THREE.MeshStandardMaterial({ color: 0x6b7368, metalness: 0.05, roughness: 0.9 });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  const group = new THREE.Group();
  group.add(mesh);
  mesh.position.y = height / 2;
  return group;
}

function fitShadowCamera(): void {
  if (!sun || !scene) return;
  const bounds = new THREE.Box3();
  if (sceneBase) {
    sceneBase.updateMatrixWorld(true);
    sceneBase.traverse((child) => {
      if (child.name.startsWith("house_")) bounds.expandByObject(child);
    });
    if (bounds.isEmpty()) {
      if (sceneCamera?.calibration) bounds.copy(sceneCamera.calibration.bounds);
      else bounds.expandByObject(sceneBase);
    }
  }
  for (const item of placed.value) bounds.expandByObject(item.object3d);
  if (ground) bounds.expandByObject(ground);
  if (bounds.isEmpty()) return;
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const radius = Math.max(size.x, size.y, size.z, 30) * 0.52 + 6;
  shadowRadius = radius;
  sun.target.position.copy(center);
  sun.target.updateMatrixWorld();
  const shadowCam = sun.shadow.camera as THREE.OrthographicCamera;
  shadowCam.left = -radius;
  shadowCam.right = radius;
  shadowCam.top = radius;
  shadowCam.bottom = -radius;
  shadowCam.near = Math.max(1, radius * 1.1);
  shadowCam.far = radius * 4.0;
  shadowCam.updateProjectionMatrix();
  applyLighting();
}

async function loadAssetObject(asset: Asset3D): Promise<THREE.Object3D> {
  if (!asset.has_model || !asset.url) return createPlaceholderMesh(asset);
  const gltf = await new GLTFLoader().loadAsync(asset.url);
  const root = gltf.scene;
  if (asset.up_axis.toLowerCase() === "z") root.rotation.x = -Math.PI / 2;
  if (asset.up_axis.toLowerCase() === "x") root.rotation.z = Math.PI / 2;
  root.updateMatrixWorld(true);
  const maxAniso = Math.min(16, renderer?.capabilities.getMaxAnisotropy() || 1);
  root.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.isMesh) {
      for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
        tuneSceneMaterialPBR(mesh, material as THREE.MeshStandardMaterial, maxAniso, true);
      }
      mesh.castShadow = true;
      mesh.receiveShadow = true;
    }
  });

  const box = new THREE.Box3().setFromObject(root);
  const size = box.getSize(new THREE.Vector3());
  const target = asset.dimensions;
  const group = new THREE.Group();
  group.add(root);

  const uniform = size.z > 1e-6 ? target.length / size.z : 1;
  root.scale.setScalar(uniform * (asset.default_scale || 1));

  const scaled = new THREE.Box3().setFromObject(root);
  root.position.y -= scaled.min.y;
  root.position.x -= (scaled.min.x + scaled.max.x) / 2;
  root.position.z -= (scaled.min.z + scaled.max.z) / 2;

  if (asset.yaw_offset) root.rotation.y = (asset.yaw_offset * Math.PI) / 180;
  return group;
}

async function placeAsset(
  asset: Asset3D,
  transform?: { id?: string; position?: number[]; rotation?: number[]; scale?: number[] },
  hit?: { point: THREE.Vector3; normal?: THREE.Vector3 },
): Promise<void> {
  if (!scene || !gizmo || placementBusy.value) return;
  placementBusy.value = true;
  exportMessage.value = `正在加载「${asset.label}」…`;
  try {
    const object3d = await loadAssetObject(asset);
    if (!scene || !gizmo) {
      disposeObject(object3d);
      return;
    }
    object3d.name = asset.code;
    if (transform?.position) {
      object3d.position.fromArray(transform.position);
    } else if (hit) {
      object3d.position.copy(hit.point);
      if (groundAlign.value && hit.normal) {
        const up = new THREE.Vector3(0, 1, 0);
        const quat = new THREE.Quaternion().setFromUnitVectors(up, hit.normal.clone().normalize());
        object3d.quaternion.copy(quat);
      }
    } else {
      object3d.position.set((itemSeq % 5) * 3 - 6, 0, ((itemSeq * 2) % 5) * 3 - 6);
      itemSeq += 1;
    }
    if (transform?.rotation) object3d.rotation.fromArray(transform.rotation as [number, number, number]);
    if (transform?.scale) object3d.scale.fromArray(transform.scale);
    scene.add(object3d);

    const item: PlacedItem = {
      id: transform?.id || `placed_${Date.now()}_${++itemSeq}`,
      code: asset.code,
      label: asset.label,
      kind: "equipment",
      object3d: markRaw(object3d),
    };
    placed.value = [...placed.value, item];
    if (!transform?.id) selectItem(item.id);
    fitShadowCamera();
    exportMessage.value = `已放置「${asset.label}」· 可拖动坐标轴调整位置`;
  } catch (error) {
    exportMessage.value = `装备加载失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    placementBusy.value = false;
  }
}

function placeSceneProp(
  prop: ScenePropDef,
  transform?: { id?: string; position?: number[]; rotation?: number[]; scale?: number[] },
): void {
  if (!scene) return;
  const [width, height, length] = prop.size;
  const object3d = createScenePropMesh(width, height, length);
  object3d.name = prop.code;
  if (transform?.position) {
    object3d.position.fromArray(transform.position);
  } else {
    object3d.position.set((itemSeq % 5) * 4 - 8, 0, ((itemSeq * 3) % 5) * 4 - 8);
    itemSeq += 1;
  }
  if (transform?.rotation) object3d.rotation.fromArray(transform.rotation as [number, number, number]);
  if (transform?.scale) object3d.scale.fromArray(transform.scale);
  scene.add(object3d);
  fitShadowCamera();
  const item: PlacedItem = {
    id: transform?.id || `prop_${Date.now()}_${itemSeq}`,
    code: prop.code,
    label: prop.label,
    kind: "scene_prop",
    object3d: markRaw(object3d),
  };
  placed.value = [...placed.value, item];
  if (!transform?.id) selectItem(item.id);
}

function selectItem(id: string): void {
  selectedId.value = id;
  inspectorTab.value = "transform";
  syncGizmoVisibility();
}

function toggleClickPlace(asset: Asset3D): void {
  pendingPlaceCode.value = asset.code;
  clickPlaceMode.value = true;
  exportMessage.value = `点击放置：在场景表面放下「${asset.label}」`;
}

function surfaceAt(clientX: number, clientY: number): { point: THREE.Vector3; normal: THREE.Vector3 } | null {
  if (!renderer || !camera) return null;
  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((clientX - rect.left) / rect.width) * 2 - 1,
    -((clientY - rect.top) / rect.height) * 2 + 1,
  );
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, camera);
  const targets: THREE.Object3D[] = [];
  if (sceneBase) targets.push(sceneBase);
  if (ground) targets.push(ground);
  for (const hit of ray.intersectObjects(targets, true)) {
    const normal = hit.face?.normal.clone().applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld)).normalize();
    if (normal && normal.y > 0.45) return { point: hit.point.clone(), normal };
  }
  return null;
}

function onAssetDrag(event: DragEvent, asset: Asset3D): void {
  event.dataTransfer?.setData("application/x-scene-asset", asset.code);
  if (event.dataTransfer) event.dataTransfer.effectAllowed = "copy";
  pendingPlaceCode.value = asset.code;
}

function onAssetDrop(event: DragEvent): void {
  event.preventDefault();
  dropActive.value = false;
  clickPlaceMode.value = false;
  const code = event.dataTransfer?.getData("application/x-scene-asset");
  const asset = assets.value.find((entry) => entry.code === code);
  const hit = surfaceAt(event.clientX, event.clientY);
  if (asset && hit) void placeAsset(asset, undefined, hit);
  else exportMessage.value = "请把装备拖到道路、院落或屋顶表面。";
}

function onCanvasPointerDown(event: PointerEvent): void {
  autoOrbit.value = false;
  pointerStart.set(event.clientX, event.clientY);
}

function onCanvasPointerUp(event: PointerEvent): void {
  if (!renderer || !camera || event.button !== 0 || gizmo?.dragging) return;
  if (pointerStart.distanceTo(new THREE.Vector2(event.clientX, event.clientY)) > 4) return;
  if (clickPlaceMode.value) {
    const asset = assets.value.find((item) => item.code === pendingPlaceCode.value);
    const hit = surfaceAt(event.clientX, event.clientY);
    if (asset && hit) {
      void placeAsset(asset, undefined, hit);
      clickPlaceMode.value = false;
    } else exportMessage.value = "请点击道路、院落或屋顶表面。";
    return;
  }
  if (gizmo?.axis) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const ray = new THREE.Raycaster();
  ray.setFromCamera(
    new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1),
    camera,
  );
  const hit = ray.intersectObjects(placed.value.map((item) => item.object3d), true)[0];
  if (hit) {
    const item = placed.value.find((entry) => {
      let node: THREE.Object3D | null = hit.object;
      while (node) {
        if (node === entry.object3d) return true;
        node = node.parent;
      }
      return false;
    });
    if (item) selectItem(item.id);
  } else {
    deselectItem();
  }
}

function syncInspector(): void {
  const obj = selectedPlaced.value?.object3d;
  if (!obj) return;
  Object.assign(inspector, {
    x: +obj.position.x.toFixed(2),
    y: +obj.position.y.toFixed(2),
    z: +obj.position.z.toFixed(2),
    yaw: +THREE.MathUtils.radToDeg(obj.rotation.y).toFixed(1),
    scale: +obj.scale.x.toFixed(2),
  });
}

function updateTransformField(field: "x" | "y" | "z" | "yaw" | "scale", value: number): void {
  inspector[field] = value;
  const obj = selectedPlaced.value?.object3d;
  if (!obj || Object.values(inspector).some((v) => !Number.isFinite(v))) return;
  obj.position.set(inspector.x, inspector.y, inspector.z);
  obj.rotation.y = THREE.MathUtils.degToRad(inspector.yaw);
  obj.scale.setScalar(Math.max(0.05, inspector.scale));
  fitShadowCamera();
}

function snapSelected(): void {
  const obj = selectedPlaced.value?.object3d;
  if (!obj) return;
  const ray = new THREE.Raycaster(new THREE.Vector3(obj.position.x, 500, obj.position.z), new THREE.Vector3(0, -1, 0));
  const hits = ray.intersectObjects([sceneBase, ground].filter(Boolean) as THREE.Object3D[], true);
  const hit = hits.find((h) => (h.face?.normal.clone().transformDirection(h.object.matrixWorld).y || 0) > 0.45);
  if (hit) obj.position.y = hit.point.y;
  syncInspector();
}

function removeSelected(): void {
  const item = selectedPlaced.value;
  if (!item || !scene || !gizmo) return;
  gizmo.detach();
  scene.remove(item.object3d);
  disposeObject(item.object3d);
  placed.value = placed.value.filter((entry) => entry.id !== item.id);
  selectedId.value = "";
  fitShadowCamera();
}

function setTransformMode(mode: TransformMode): void {
  transformMode.value = mode;
  gizmo?.setMode(mode);
}

function projectBBox(
  object3d: THREE.Object3D,
  cam: THREE.Camera,
  width: number,
  height: number,
): Pick<SceneObjectBBox, "xmin" | "ymin" | "xmax" | "ymax"> {
  const box = new THREE.Box3().setFromObject(object3d);
  const corners = [
    new THREE.Vector3(box.min.x, box.min.y, box.min.z),
    new THREE.Vector3(box.min.x, box.min.y, box.max.z),
    new THREE.Vector3(box.min.x, box.max.y, box.min.z),
    new THREE.Vector3(box.min.x, box.max.y, box.max.z),
    new THREE.Vector3(box.max.x, box.min.y, box.min.z),
    new THREE.Vector3(box.max.x, box.min.y, box.max.z),
    new THREE.Vector3(box.max.x, box.max.y, box.min.z),
    new THREE.Vector3(box.max.x, box.max.y, box.max.z),
  ];
  let xmin = Number.POSITIVE_INFINITY;
  let ymin = Number.POSITIVE_INFINITY;
  let xmax = Number.NEGATIVE_INFINITY;
  let ymax = Number.NEGATIVE_INFINITY;
  for (const corner of corners) {
    const projected = corner.clone().project(cam);
    const px = (projected.x * 0.5 + 0.5) * width;
    const py = (-projected.y * 0.5 + 0.5) * height;
    xmin = Math.min(xmin, px);
    xmax = Math.max(xmax, px);
    ymin = Math.min(ymin, py);
    ymax = Math.max(ymax, py);
  }
  const clamp = (value: number, max: number) => Math.min(max, Math.max(0, value));
  return {
    xmin: clamp(xmin, width),
    ymin: clamp(ymin, height),
    xmax: clamp(xmax, width),
    ymax: clamp(ymax, height),
  };
}

async function runExport(): Promise<void> {
  const rendererRef = renderer;
  const sceneRef = scene;
  const cam = camera;
  if (!rendererRef || !sceneRef || !cam || exporting.value) return;
  if (!placed.value.length) {
    exportMessage.value = "先放置至少一个装备再导出。";
    return;
  }
  exporting.value = true;
  exportMessage.value = "正在渲染多角度视图…";
  const size = exportResolution.value;
  const previousSize = rendererRef.getSize(new THREE.Vector2());
  const previousRatio = rendererRef.getPixelRatio();
  const previousPosition = cam.position.clone();
  const previousQuaternion = cam.quaternion.clone();
  const previousAspect = cam.aspect;
  try {
    rendererRef.setPixelRatio(1);
    rendererRef.setSize(size, size, false);
    if (composer) {
      composer.setPixelRatio(1);
      composer.setSize(size, size);
    }
    cam.aspect = 1;
    cam.updateProjectionMatrix();
    if (gizmoHelper) gizmoHelper.visible = false;

    const frames: SceneFrame[] = [];
    const count = exportViewCount.value;
    for (let index = 0; index < count; index += 1) {
      const azimuth = (index / count) * Math.PI * 2;
      const focus = new THREE.Box3();
      for (const item of placed.value) focus.expandByObject(item.object3d);
      const center = focus.getCenter(new THREE.Vector3());
      const radius = Math.max(28, focus.getSize(new THREE.Vector3()).length() * 1.2);
      const height = center.y + radius * 0.65;
      cam.position.set(center.x + radius * Math.cos(azimuth), height, center.z + radius * Math.sin(azimuth));
      cam.lookAt(center);
      cam.updateMatrixWorld();
      photoProjectors?.update(cam.position);
      renderFrame(cam);

      const dataUrl = rendererRef.domElement.toDataURL("image/png");
      const image_base64 = dataUrl.slice(dataUrl.indexOf(",") + 1);
      const objects: SceneObjectBBox[] = placed.value.map((item) => ({
        name: item.code,
        ...projectBBox(item.object3d, cam, size, size),
      }));
      frames.push({
        image_base64,
        width: size,
        height: size,
        objects,
        filename: `view_${String(index + 1).padStart(2, "0")}`,
        folder: "scene_export",
      });
    }

    exportMessage.value = "正在提交导出任务…";
    const result = await exportSceneFrames(frames);
    exportMessage.value = `已提交 ${frames.length} 帧，任务 ${result.task_id}`;
    emit("exported", result.task_id);
  } catch (error) {
    exportMessage.value = `导出失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    syncGizmoVisibility();
    rendererRef.setPixelRatio(previousRatio);
    rendererRef.setSize(previousSize.x, previousSize.y, false);
    if (composer) {
      composer.setPixelRatio(previousRatio);
      composer.setSize(previousSize.x, previousSize.y);
    }
    cam.position.copy(previousPosition);
    cam.quaternion.copy(previousQuaternion);
    cam.aspect = previousAspect;
    updateViewportProjection();
    exporting.value = false;
  }
}

function initThree(): void {
  if (renderer || !canvasHost.value) return;

  const width = canvasHost.value.clientWidth || 640;
  const height = canvasHost.value.clientHeight || 480;

  scene = markRaw(new THREE.Scene());
  scene.background = new THREE.Color(0xd8cbb4);
  scene.fog = new THREE.Fog(0xd8cbb4, 260, 860);

  camera = markRaw(new THREE.PerspectiveCamera(45, width / height, 0.1, 2400));
  camera.position.set(18, 14, 18);
  camera.lookAt(0, 1, 0);

  renderer = markRaw(new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true }));
  const pixelRatio = Math.min(window.devicePixelRatio, 2);
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(width, height, false);
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.AgXToneMapping;
  renderer.toneMappingExposure = 0.94;
  canvasHost.value.innerHTML = "";
  canvasHost.value.appendChild(renderer.domElement);
  renderer.domElement.addEventListener("pointerdown", onCanvasPointerDown);
  renderer.domElement.addEventListener("pointerup", onCanvasPointerUp);

  skyDome = markRaw(new Sky());
  skyDome.scale.setScalar(2600);
  skyDome.frustumCulled = false;
  skyDome.name = "atmosphere_sky";
  scene.add(skyDome);

  pmremGenerator = markRaw(new THREE.PMREMGenerator(renderer));
  pmremGenerator.compileEquirectangularShader();
  lastEnvKey = "";

  // Note: SparkRenderer is instantiated lazily only when route === '3dgs' so default Blender PBR triggers zero WASM.
  shadowCatcherMat = markRaw(
    new THREE.ShadowMaterial({
      opacity: 0.48,
      depthWrite: false,
      fog: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    }),
  );

  ambientLight = markRaw(new THREE.AmbientLight(0xffffff, lighting.ambient));
  scene.add(ambientLight);

  hemisphere = markRaw(new THREE.HemisphereLight(0x90b9eb, 0x32291f, 0.08));
  scene.add(hemisphere);

  sun = markRaw(new THREE.DirectionalLight(0xffffff, lighting.intensity));
  sun.castShadow = true;
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.bias = -0.00012;
  sun.shadow.normalBias = 0.012;
  sun.shadow.radius = 1.25;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 120;
  const shadowSize = 35;
  const shadowCam = sun.shadow.camera as THREE.OrthographicCamera;
  shadowCam.left = -shadowSize;
  shadowCam.right = shadowSize;
  shadowCam.top = shadowSize;
  shadowCam.bottom = -shadowSize;
  scene.add(sun);
  scene.add(sun.target);

  composerTarget = markRaw(
    new THREE.WebGLRenderTarget(Math.max(1, Math.round(width * pixelRatio)), Math.max(1, Math.round(height * pixelRatio)), {
      samples: 4,
      type: THREE.HalfFloatType,
    }),
  );
  composer = markRaw(new EffectComposer(renderer, composerTarget));
  composer.setPixelRatio(pixelRatio);
  renderPass = markRaw(new RenderPass(scene, camera));
  composer.addPass(renderPass);
  gtaoPass = markRaw(new SceneGTAOPass(scene, camera, width, height));
  gtaoPass.output = GTAOPass.OUTPUT.Default;
  gtaoPass.blendIntensity = 0.78;
  gtaoPass.updateGtaoMaterial({ radius: 2.1, distanceExponent: 1.35, thickness: 2.0, scale: 1.25, samples: 12 });
  gtaoPass.updatePdMaterial({ lumaPhi: 10, depthPhi: 2.5, normalPhi: 3.5, radius: 6, samples: 12 });
  composer.addPass(gtaoPass);
  outputPass = markRaw(new OutputPass());
  composer.addPass(outputPass);
  photoGradePass = markRaw(new ShaderPass(PHOTO_GRADE_SHADER));
  composer.addPass(photoGradePass);
  composer.setSize(width, height);

  buildGround();
  applyLighting();

  orbit = markRaw(new OrbitControls(camera, renderer.domElement));
  orbit.enableDamping = true;
  orbit.maxPolarAngle = Math.PI * 0.49;
  orbit.minDistance = 5;
  orbit.maxDistance = 650;
  orbit.autoRotateSpeed = 0.6;
  orbit.addEventListener("change", () => alignOrbitUp());
  orbit.target.set(0, 1, 0);

  gizmo = markRaw(new TransformControls(camera, renderer.domElement));
  gizmo.setMode(transformMode.value);
  gizmo.addEventListener("dragging-changed", (event) => {
    if (orbit) orbit.enabled = !(event as { value: boolean }).value;
    if (!(event as { value: boolean }).value) {
      if (groundAlign.value && transformMode.value === "translate") snapSelected();
      fitShadowCamera();
    }
  });
  gizmo.addEventListener("objectChange", syncInspector);
  gizmo.setSize(0.85);
  gizmoHelper = markRaw(gizmo.getHelper());
  scene.add(gizmoHelper);
  syncGizmoVisibility();

  const grid = new THREE.GridHelper(60, 30, 0x3a4438, 0x232823);
  gridHelper = markRaw(grid);
  (grid.material as THREE.Material).transparent = true;
  (grid.material as THREE.Material).opacity = 0.35;
  scene.add(grid);

  resizeObserver = new ResizeObserver(() => {
    if (!renderer || !camera || !canvasHost.value) return;
    const nextWidth = canvasHost.value.clientWidth || 640;
    const nextHeight = canvasHost.value.clientHeight || 480;
    camera.aspect = nextWidth / nextHeight;
    updateViewportProjection();
    renderer.setSize(nextWidth, nextHeight, false);
    composer?.setSize(nextWidth, nextHeight);
  });
  resizeObserver.observe(canvasHost.value);

  const tick = () => {
    animationHandle = requestAnimationFrame(tick);
    if (orbit) {
      orbit.autoRotate = autoOrbit.value;
      if (!exporting.value) orbit.update();
    }
    if (!exporting.value && scene && camera) {
      photoProjectors?.update(camera.position);
      renderFrame(camera);
    }
  };
  tick();
}

function teardownThree(): void {
  sceneLoadSeq += 1;
  splatLoadSeq += 1;
  clearSceneSplat();
  photoProjectors?.dispose();
  photoProjectors = null;
  cancelAnimationFrame(animationHandle);
  resizeObserver?.disconnect();
  resizeObserver = null;
  gizmo?.detach();
  gizmo?.dispose();
  orbit?.dispose();
  if (gizmoHelper && scene) scene.remove(gizmoHelper);
  gizmoHelper = null;
  if (sparkRenderer && scene) scene.remove(sparkRenderer);
  sparkRenderer?.dispose?.();
  sparkRenderer = null;
  if (scene) {
    for (const item of placed.value) {
      scene.remove(item.object3d);
      disposeObject(item.object3d);
    }
    disposeObject(scene);
  }
  shadowCatcherMat?.dispose();
  shadowCatcherMat = null;
  if (skyDome && scene) scene.remove(skyDome);
  skyDome?.geometry.dispose();
  skyDome?.material.dispose();
  skyDome = null;
  envRenderTarget?.dispose();
  envRenderTarget = null;
  pmremGenerator?.dispose();
  pmremGenerator = null;
  lastEnvKey = "";
  gtaoPass?.dispose();
  gtaoPass = null;
  renderPass?.dispose();
  renderPass = null;
  outputPass?.dispose();
  outputPass = null;
  photoGradePass?.dispose();
  photoGradePass = null;
  composer?.dispose();
  composer = null;
  composerTarget?.dispose();
  composerTarget = null;
  if (scene) scene.environment = null;
  ground = null;
  sceneBase = null;
  gridHelper = null;
  hemisphere = null;
  sceneReady.value = false;
  autoOrbit.value = false;
  groundMaterial = null;
  placed.value = [];
  selectedId.value = "";
  gizmo = null;
  orbit = null;
  sun = null;
  ambientLight = null;
  scene = null;
  camera = null;
  if (renderer) {
    renderer.dispose();
    renderer.domElement.remove();
    renderer = null;
  }
}

function captureObjects(): SceneObjectState[] {
  return placed.value.map((item, index) => ({
    id: item.id,
    asset_code: item.code,
    label: item.label,
    kind: item.kind,
    position: item.object3d.position.toArray(),
    rotation: [item.object3d.rotation.x, item.object3d.rotation.y, item.object3d.rotation.z],
    scale: item.object3d.scale.toArray(),
    sort_order: index,
  }));
}

function clearPlaced(): void {
  if (gizmo) gizmo.detach();
  for (const item of placed.value) {
    scene?.remove(item.object3d);
    disposeObject(item.object3d);
  }
  placed.value = [];
  selectedId.value = "";
}

async function applySessionState(state: SceneSessionState): Promise<void> {
  clearPlaced();
  sessionId.value = state.id;
  sessionTitle.value = state.title;
  Object.assign(lighting, state.lighting);
  sceneTextureName.value = state.ground_texture || "";
  if (state.ground_texture?.startsWith("scene:")) {
    const mesh = sceneMeshes.value.find((m) => m.code === state.ground_texture?.slice(6));
    if (mesh && mesh.code !== activeSceneCode.value) {
      await loadSceneMesh(mesh, false, currentRoute.value === "3dgs");
    }
  } else {
    resetFlatGround();
  }
  for (const obj of state.objects) {
    const transform = {
      id: obj.id || undefined,
      position: obj.position,
      rotation: obj.rotation,
      scale: obj.scale,
    };
    if (obj.kind === "scene_prop") {
      const prop = SCENE_PROPS.find((entry) => entry.code === obj.asset_code);
      if (prop) {
        placeSceneProp(prop, transform);
        continue;
      }
    }
    const asset =
      assets.value.find((entry) => entry.code === obj.asset_code) ??
      ({
        code: obj.asset_code,
        label: obj.label || obj.asset_code,
        category: "会话",
        has_model: false,
        url: null,
        dimensions: { length: 4, width: 2, height: 2 },
        origin: [0, 0, 0],
        up_axis: "y",
        yaw_offset: 0,
        default_scale: 1,
        thumbnail: null,
      } satisfies Asset3D);
    await placeAsset(asset, transform);
  }
  fitShadowCamera();
}

async function refreshSessions(): Promise<void> {
  try {
    sessions.value = await listSceneSessions();
  } catch {
    sessions.value = [];
  }
}

async function saveCurrentSession(): Promise<void> {
  if (sessionBusy.value) return;
  sessionBusy.value = true;
  sessionMessage.value = "保存中…";
  try {
    const payload = {
      title: sessionTitle.value || "未命名场景",
      lighting: { ...lighting },
      objects: captureObjects(),
      ground_texture: sceneTextureName.value || null,
    };
    if (sessionId.value) {
      const state = await saveSceneSession(sessionId.value, payload);
      sessionId.value = state.id;
    } else {
      const state = await createSceneSession(payload);
      sessionId.value = state.id;
      sessionTitle.value = state.title;
    }
    sessionMessage.value = "场景已保存";
    await refreshSessions();
  } catch (error) {
    sessionMessage.value = `保存失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    sessionBusy.value = false;
  }
}

async function openSession(id: string): Promise<void> {
  if (sessionBusy.value) return;
  sessionBusy.value = true;
  sessionMessage.value = "加载中…";
  try {
    const state = await getSceneSession(id);
    await applySessionState(state);
    sessionMessage.value = `已加载「${state.title}」`;
  } catch (error) {
    sessionMessage.value = `加载失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    sessionBusy.value = false;
  }
}

function startNewSession(): void {
  clearPlaced();
  sessionId.value = "";
  sessionTitle.value = "未命名场景";
  if (currentRoute.value === "sandbox") {
    resetFlatGround();
  }
  sessionMessage.value = "已新建空白放置会话";
}

async function activateRouteScene(route: WorkspaceRoute): Promise<void> {
  if (!scene) return;
  inspectorTab.value = "scene";
  exportMessage.value = "";
  splatError.value = "";

  const groups = classifiedScenes.value;
  if (route === "sandbox") {
    resetFlatGround();
    sessionTitle.value = "平地自由沙盘";
    return;
  }
  if (route === "3dgs") {
    const target =
      (props.initialSceneCode ? groups.splat.find((m) => m.code === props.initialSceneCode) : undefined) ||
      groups.splat[0];
    if (target) {
      await loadSceneMesh(target, false, true);
      sessionTitle.value = target.label;
    } else {
      resetFlatGround();
      splatError.value = "当前暂无可用 3DGS 点云场景";
    }
    return;
  }
  if (route === "calibrated") {
    const target =
      (props.initialSceneCode ? groups.calibrated.find((m) => m.code === props.initialSceneCode) : undefined) ||
      groups.calibrated.find((m) => m.code === "VILLAGE_CALIBRATED") ||
      groups.calibrated[0];
    if (target) {
      await loadSceneMesh(target, false, false);
      sessionTitle.value = target.label;
      if (props.initialPanel && target.plan_url) {
        requestedView.value = props.initialView;
        inspectorTab.value = props.initialPanel;
        if (props.initialPanel === "ground") cameraView("back");
      }
    } else {
      resetFlatGround();
    }
    return;
  }

  // Default route: blender
  const target =
    (props.initialSceneCode ? groups.blender.find((m) => m.code === props.initialSceneCode) : undefined) ||
    groups.blender.find((m) => m.code === "VILLAGE_BLENDER") ||
    groups.blender[0] ||
    sceneMeshes.value[0];
  if (target) {
    await loadSceneMesh(target, false, false);
    sessionTitle.value = target.label;
  }
}

async function bootstrap(): Promise<void> {
  await nextTick();
  sessionId.value = "";
  sessionMessage.value = "";
  exportMessage.value = "";
  inspectorTab.value = "scene";
  lightingMode.value = "relight_all";
  clickPlaceMode.value = false;
  activeSceneCode.value = "";
  initThree();
  lightingPreset("day");
  assetsLoading.value = true;
  assetsError.value = "";
  try {
    assets.value = await fetchAssets3D(true);
    assets.value.sort((a, b) => Number(b.code === "Tank_model1") - Number(a.code === "Tank_model1"));
    if (!assets.value.length) {
      assets.value = DEMO_ASSETS;
      assetsError.value = "资产库为空，已载入演示占位体。";
    }
    await refreshSessions();
    await refreshTrainingSets();
    await refreshSceneMeshes();
    await activateRouteScene(currentRoute.value);
  } catch (error) {
    assetsError.value = `资产加载失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    assetsLoading.value = false;
  }
}

watch(
  () => props.visible,
  (visible) => {
    if (visible) {
      void bootstrap();
    } else {
      if (sessionId.value && placed.value.length) {
        void saveCurrentSession();
      }
      teardownThree();
    }
  },
  { immediate: true },
);

watch(
  () => [props.route, props.initialSceneCode, props.initialPanel, props.initialView] as const,
  ([nextRoute]) => {
    if (props.visible && scene && nextRoute) {
      void activateRouteScene(nextRoute);
    }
  },
);

function onEditorKeyDown(event: KeyboardEvent): void {
  if (event.key === "Escape" && props.visible && !redrawModalOpen.value) {
    clickPlaceMode.value = false;
    deselectItem();
  }
}

window.addEventListener("keydown", onEditorKeyDown);

onBeforeUnmount(() => {
  window.removeEventListener("keydown", onEditorKeyDown);
  teardownThree();
});

async function runPhase3Batch(): Promise<void> {
  if (batchBusy.value) return;
  batchBusy.value = true;
  batchMessage.value = "批量渲染中…";
  try {
    const objects = captureObjects().map((item) => ({
      asset_code: item.asset_code,
      label: item.label,
      kind: item.kind,
      position: item.position,
      rotation: item.rotation,
      scale: item.scale,
    }));
    const baseObjects = objects.length
      ? objects
      : [{ asset_code: "K1", label: "K1坦克", kind: "equipment", position: [0, 0, 0], rotation: [0, 0, 0], scale: [1, 1, 1] }];
    const summary = await runSceneBatch({
      name: `scene_batch_${Date.now()}`,
      include_masks: true,
      defaults: {
        map_id: "A001",
        weather: "晴天",
        scene: "城乡结合部",
        camera: {
          mode: "drone",
          views: exportViewCount.value,
          resolution: Math.min(exportResolution.value, 768),
          radius: 24,
          height: 14,
          look_at: [0, 1.2, 0],
          fov: 45,
        },
      },
      jobs: [
        { id: "noon", lighting: { ...lighting, azimuth: 45, elevation: 70, temperature: 5800 }, objects: baseObjects },
        {
          id: "dusk",
          lighting: { ...lighting, azimuth: 250, elevation: 18, temperature: 3600, ambient: 0.22 },
          objects: baseObjects,
        },
      ],
    });
    batchMessage.value = `已生成 ${summary.frames} 帧 → ${summary.output_dir}`;
    trainingSets.value = await listTrainingSets();
  } catch (error) {
    batchMessage.value = `批量失败：${error instanceof Error ? error.message : String(error)}`;
  } finally {
    batchBusy.value = false;
  }
}

async function refreshTrainingSets(): Promise<void> {
  try {
    trainingSets.value = await listTrainingSets();
  } catch {
    trainingSets.value = [];
  }
}

function patchLighting(patch: Partial<SceneLightingState>): void {
  Object.assign(lighting, patch);
}

function setLightingModeSafe(mode: LightingMode): void {
  lightingMode.value = mode;
}
</script>

<template>
  <template v-if="props.standalone">
    <div v-if="props.visible" class="scene-shell">
      <header class="scene-header">
        <div class="scene-title">
          <span class="scene-mark"><Star :size="16" /></span>
          <div class="title-stack">
            <span class="mono">3D SCENE WORKSPACE</span>
            <h2 id="scene-editor-title" class="display-font">3D 场景编辑器</h2>
          </div>
          <span class="route-pill" :class="{ exp: routeInfo.isExperimental }">
            {{ routeInfo.badge }}
          </span>
        </div>

        <div class="header-actions">
          <div v-if="currentRoute === 'calibrated'" class="subview-switch" role="group" aria-label="照片标定视图">
            <button type="button" @click="emit('openCalibrated2D')">2D 照片标定</button>
            <button type="button" class="active">3D 标定预览</button>
          </div>
          <button
            v-if="currentRoute !== 'blender'"
            type="button"
            class="hdr-btn back"
            @click="emit('changeRoute', 'blender')"
          >
            ← 返回 Blender 主工作台
          </button>
          <span class="ready-badge">
            {{ sceneMeshLoading || splatLoading ? "加载三维场景…" : sceneReady ? "场景已就绪" : "准备场景" }}
          </span>
          <button class="hdr-btn" type="button" @click="snapshot">
            <Download :size="14" /> 保存当前画面
          </button>
          <button class="hdr-btn accent" type="button" @click="openRedrawStudio">
            <Sparkles :size="14" /> 截图并渲染
          </button>
          <ExperimentalRoutesMenu
            :active-route="currentRoute"
            :counts="routeCounts"
            @select="emit('changeRoute', $event)"
          />
        </div>
      </header>

      <div class="scene-body">
        <div
          class="scene-viewport"
          :class="{ 'drag-active': dropActive, placing: clickPlaceMode }"
          @dragover.prevent="dropActive = true"
          @dragleave="dropActive = false"
          @drop="onAssetDrop"
        >
          <div ref="canvasHost" class="scene-canvas" />
          <div class="stage-label">
            <span class="mono">{{ routeInfo.badge }}</span>
            <strong>{{ activeSceneLabel }}</strong>
          </div>
          <div class="viewport-toolbar" role="group" aria-label="场景视角">
            <button type="button" @click="cameraView('photo')">原图视角</button>
            <button type="button" @click="cameraView('back')">背面视角</button>
            <button type="button" @click="cameraView('top')">俯视</button>
            <button type="button" :class="{ active: autoOrbit }" @click="autoOrbit = !autoOrbit">
              {{ autoOrbit ? "暂停环绕" : "自动环绕" }}
            </button>
            <button type="button" :disabled="!selectedPlaced" @click="cameraView('focus')">聚焦装备</button>
          </div>
          <div v-if="dropActive" class="drop-prompt">松开，将装备放到场景表面</div>
          <div v-if="sceneMeshLoading || splatLoading || placementBusy" class="loading-indicator">
            <span />{{
              placementBusy
                ? "正在加载装备模型…"
                : splatLoading
                  ? "正在加载 3DGS 高斯点云…"
                  : "正在加载三维场景…"
            }}
          </div>
          <div class="viewport-status" role="status">
            <span>{{ exportMessage || (clickPlaceMode ? "点击场景表面放置装备" : "左键环绕 · 右键平移 · 滚轮缩放") }}</span>
            <span class="mono">{{ placed.length }} 个装备 · 实时光照</span>
          </div>
        </div>

        <aside class="scene-side">
          <nav class="inspector-tabs" aria-label="编辑工具">
            <button
              v-for="tab in availableTabs"
              :key="tab.key"
              type="button"
              :class="{ active: inspectorTab === tab.key }"
              @click="inspectorTab = tab.key"
            >
              {{ tab.label }}
            </button>
          </nav>

          <div class="side-scroll">
            <template v-if="inspectorTab === 'scene'">
              <SceneSourcePanel
                :route="currentRoute"
                :active-mesh="activeMesh"
                :route-scenes="routeScenes"
                :loaded-meta="loadedMeta"
                :scene-mesh-loading="sceneMeshLoading"
                :splat-loading="splatLoading"
                :has-splat="hasSplat"
                :splat-error="splatError"
                :lighting-mode="lightingMode"
                :scene-texture-name="sceneTextureName"
                :active-scene-code="activeSceneCode"
                @select-scene="loadSceneMesh($event, false, currentRoute === '3dgs')"
                @open-calibrated2d="emit('openCalibrated2D')"
                @set-lighting-mode="setLightingModeSafe"
                @reset-flat-ground="resetFlatGround"
                @upload-ground-texture="handleUploadGroundTexture"
                @place-prop="placeSceneProp($event)"
              />
              <EquipmentPanel
                :assets="assets"
                :loading="assetsLoading"
                :error="assetsError"
                :click-place-mode="clickPlaceMode"
                :pending-place-code="pendingPlaceCode"
                @drag-asset="onAssetDrag"
                @drag-end="dropActive = false"
                @toggle-click-place="toggleClickPlace"
                @cancel-click-place="clickPlaceMode = false"
              />
            </template>

            <section v-else-if="inspectorTab === 'ground' && currentRoute === 'calibrated' && activeMesh?.plan_url">
              <GroundPanel
                :scene="activeMesh"
                @result="applyBuiltScene($event, '地面已更新，新版本已保存到场景库。')"
              />
            </section>

            <section v-else-if="inspectorTab === 'orbit' && currentRoute === 'calibrated' && activeMesh?.plan_url">
              <OrbitPanel
                :scene="activeMesh"
                :initial-view="requestedView"
                @camera="showOrbitCamera"
                @result="applyBuiltScene($event, '补充视角已写入新版场景。')"
              />
            </section>

            <TransformPanel
              v-else-if="inspectorTab === 'transform'"
              :placed="placedSummaries"
              :selected-id="selectedId"
              :transform-mode="transformMode"
              :inspector="inspector"
              :ground-align="groundAlign"
              @select="selectItem"
              @deselect="deselectItem"
              @set-mode="setTransformMode"
              @update-field="updateTransformField"
              @snap="snapSelected"
              @update:ground-align="groundAlign = $event"
              @remove="removeSelected"
            />

            <LightingPanel
              v-else-if="inspectorTab === 'light'"
              :route="currentRoute"
              :lighting="lighting"
              :lighting-mode="lightingMode"
              @preset="lightingPreset"
              @update-lighting="patchLighting"
              @set-lighting-mode="setLightingModeSafe"
            />

            <ExportPanel
              v-else-if="inspectorTab === 'export'"
              :standalone="props.standalone"
              :session-title="sessionTitle"
              :session-id="sessionId"
              :sessions="sessions"
              :session-busy="sessionBusy"
              :session-message="sessionMessage"
              :export-view-count="exportViewCount"
              :export-resolution="exportResolution"
              :exporting="exporting"
              :export-message="exportMessage"
              :placed-count="placed.length"
              :downloads="props.downloads"
              :batch-busy="batchBusy"
              :batch-message="batchMessage"
              :training-sets="trainingSets"
              @update:session-title="sessionTitle = $event"
              @save-session="saveCurrentSession"
              @new-session="startNewSession"
              @open-session="openSession"
              @snapshot="snapshot"
              @open-redraw="openRedrawStudio"
              @update:export-view-count="exportViewCount = $event"
              @update:export-resolution="exportResolution = $event"
              @run-export="runExport"
              @run-batch="runPhase3Batch"
            />
          </div>
        </aside>
      </div>
    </div>
  </template>

  <Teleport v-else to="body">
    <div v-if="props.visible" class="scene-overlay" role="dialog" aria-modal="true" aria-labelledby="scene-editor-title">
      <button class="scene-scrim" type="button" aria-label="关闭场景编辑器" @click="emit('close')" />
      <section class="scene-shell modal-panel">
        <header class="scene-header">
          <div class="scene-title">
            <span class="scene-mark"><Star :size="16" /></span>
            <div class="title-stack">
              <span class="mono">3D SCENE WORKSPACE</span>
              <h2 id="scene-editor-title" class="display-font">3D 场景编辑器</h2>
            </div>
            <span class="route-pill" :class="{ exp: routeInfo.isExperimental }">{{ routeInfo.badge }}</span>
          </div>
          <div class="header-actions">
            <span class="ready-badge">
              {{ sceneMeshLoading || splatLoading ? "加载三维场景…" : sceneReady ? "场景已就绪" : "准备场景" }}
            </span>
            <button class="hdr-btn" type="button" @click="snapshot"><Download :size="14" /> 保存当前画面</button>
            <button class="hdr-btn accent" type="button" @click="openRedrawStudio"><Sparkles :size="14" /> 截图并渲染</button>
            <button class="icon-button" type="button" aria-label="关闭" @click="emit('close')"><X :size="18" /></button>
          </div>
        </header>

        <div class="scene-body">
          <div
            class="scene-viewport"
            :class="{ 'drag-active': dropActive, placing: clickPlaceMode }"
            @dragover.prevent="dropActive = true"
            @dragleave="dropActive = false"
            @drop="onAssetDrop"
          >
            <div ref="canvasHost" class="scene-canvas" />
            <div class="stage-label">
              <span class="mono">{{ routeInfo.badge }}</span>
              <strong>{{ activeSceneLabel }}</strong>
            </div>
            <div class="viewport-toolbar" role="group" aria-label="场景视角">
              <button type="button" @click="cameraView('photo')">原图视角</button>
              <button type="button" @click="cameraView('back')">背面视角</button>
              <button type="button" @click="cameraView('top')">俯视</button>
              <button type="button" :class="{ active: autoOrbit }" @click="autoOrbit = !autoOrbit">
                {{ autoOrbit ? "暂停环绕" : "自动环绕" }}
              </button>
              <button type="button" :disabled="!selectedPlaced" @click="cameraView('focus')">聚焦装备</button>
            </div>
            <div v-if="dropActive" class="drop-prompt">松开，将装备放到场景表面</div>
            <div v-if="sceneMeshLoading || splatLoading || placementBusy" class="loading-indicator">
              <span />{{ placementBusy ? "正在加载装备模型…" : splatLoading ? "正在加载 3DGS 高斯点云…" : "正在加载三维场景…" }}
            </div>
            <div class="viewport-status" role="status">
              <span>{{ exportMessage || (clickPlaceMode ? "点击场景表面放置装备" : "左键环绕 · 右键平移 · 滚轮缩放") }}</span>
              <span class="mono">{{ placed.length }} 个装备 · 实时光照</span>
            </div>
          </div>

          <aside class="scene-side">
            <nav class="inspector-tabs" aria-label="编辑工具">
              <button
                v-for="tab in availableTabs"
                :key="tab.key"
                type="button"
                :class="{ active: inspectorTab === tab.key }"
                @click="inspectorTab = tab.key"
              >
                {{ tab.label }}
              </button>
            </nav>
            <div class="side-scroll">
              <template v-if="inspectorTab === 'scene'">
                <SceneSourcePanel
                  :route="currentRoute"
                  :active-mesh="activeMesh"
                  :route-scenes="routeScenes"
                  :loaded-meta="loadedMeta"
                  :scene-mesh-loading="sceneMeshLoading"
                  :splat-loading="splatLoading"
                  :has-splat="hasSplat"
                  :splat-error="splatError"
                  :lighting-mode="lightingMode"
                  :scene-texture-name="sceneTextureName"
                  :active-scene-code="activeSceneCode"
                  @select-scene="loadSceneMesh($event, false, currentRoute === '3dgs')"
                  @open-calibrated2d="emit('close')"
                  @set-lighting-mode="setLightingModeSafe"
                  @reset-flat-ground="resetFlatGround"
                  @upload-ground-texture="handleUploadGroundTexture"
                  @place-prop="placeSceneProp($event)"
                />
                <EquipmentPanel
                  :assets="assets"
                  :loading="assetsLoading"
                  :error="assetsError"
                  :click-place-mode="clickPlaceMode"
                  :pending-place-code="pendingPlaceCode"
                  @drag-asset="onAssetDrag"
                  @drag-end="dropActive = false"
                  @toggle-click-place="toggleClickPlace"
                  @cancel-click-place="clickPlaceMode = false"
                />
              </template>
              <TransformPanel
                v-else-if="inspectorTab === 'transform'"
                :placed="placedSummaries"
                :selected-id="selectedId"
                :transform-mode="transformMode"
                :inspector="inspector"
                :ground-align="groundAlign"
                @select="selectItem"
                @deselect="deselectItem"
                @set-mode="setTransformMode"
                @update-field="updateTransformField"
                @snap="snapSelected"
                @update:ground-align="groundAlign = $event"
                @remove="removeSelected"
              />
              <LightingPanel
                v-else-if="inspectorTab === 'light'"
                :route="currentRoute"
                :lighting="lighting"
                :lighting-mode="lightingMode"
                @preset="lightingPreset"
                @update-lighting="patchLighting"
                @set-lighting-mode="setLightingModeSafe"
              />
              <ExportPanel
                v-else-if="inspectorTab === 'export'"
                :standalone="props.standalone"
                :session-title="sessionTitle"
                :session-id="sessionId"
                :sessions="sessions"
                :session-busy="sessionBusy"
                :session-message="sessionMessage"
                :export-view-count="exportViewCount"
                :export-resolution="exportResolution"
                :exporting="exporting"
                :export-message="exportMessage"
                :placed-count="placed.length"
                :downloads="props.downloads"
                :batch-busy="batchBusy"
                :batch-message="batchMessage"
                :training-sets="trainingSets"
                @update:session-title="sessionTitle = $event"
                @save-session="saveCurrentSession"
                @new-session="startNewSession"
                @open-session="openSession"
                @snapshot="snapshot"
                @open-redraw="openRedrawStudio"
                @update:export-view-count="exportViewCount = $event"
                @update:export-resolution="exportResolution = $event"
                @run-export="runExport"
                @run-batch="runPhase3Batch"
              />
            </div>
          </aside>
        </div>
      </section>
    </div>
  </Teleport>

  <RedrawModal
    :visible="redrawModalOpen"
    :capture="redrawCapture"
    :scene-code="activeSceneCode"
    :scene-label="activeSceneLabel"
    :default-reference-url="activeMesh?.source_url || loadedMeta.referenceImageUrl || '/scene-demo/village-source.png'"
    @close="redrawModalOpen = false"
    @recapture="openRedrawStudio"
  />
</template>

<style scoped>
.scene-overlay {
  position: fixed;
  inset: 0;
  z-index: 4000;
  display: grid;
  place-items: center;
  padding: 12px;
}

.scene-scrim {
  position: absolute;
  inset: 0;
  border: 0;
  background: rgba(8, 10, 8, 0.72);
  cursor: pointer;
}

.scene-shell {
  width: 100vw;
  height: 100vh;
  display: grid;
  grid-template-rows: 54px minmax(0, 1fr);
  overflow: hidden;
  background: var(--ink-raised);
}

.scene-shell.modal-panel {
  position: relative;
  width: 100%;
  height: 100%;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-floating);
}

.scene-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 0 16px;
  border-bottom: 1px solid var(--line);
  background: var(--ink-raised);
}

.scene-title {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.scene-mark {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border-radius: 8px;
  color: var(--signal-ink);
  background: var(--signal);
  flex-shrink: 0;
}

.title-stack span.mono {
  display: block;
  font-size: 9px;
  letter-spacing: 0.14em;
  color: var(--text-muted);
}

.title-stack h2 {
  margin: 0;
  font-size: 15px;
  font-weight: 650;
  line-height: 1.2;
}

.route-pill {
  padding: 3px 9px;
  border-radius: 99px;
  font-size: 11px;
  font-weight: 600;
  background: rgba(201, 255, 87, 0.14);
  color: var(--signal);
  border: 1px solid rgba(201, 255, 87, 0.35);
  white-space: nowrap;
}

.route-pill.exp {
  background: rgba(243, 191, 79, 0.14);
  color: var(--amber);
  border-color: rgba(243, 191, 79, 0.38);
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.hdr-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
}

.hdr-btn:hover {
  border-color: var(--signal);
  color: var(--text-primary);
}

.hdr-btn.back {
  border-color: rgba(201, 255, 87, 0.4);
  color: var(--signal);
}

.hdr-btn.accent {
  border-color: rgba(201, 255, 87, 0.55);
  background: rgba(201, 255, 87, 0.14);
  color: var(--signal);
  font-weight: 600;
}

.hdr-btn.accent:hover {
  background: var(--signal);
  color: var(--signal-ink);
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

.icon-button {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface-muted);
  cursor: pointer;
}

.ready-badge {
  color: var(--text-muted);
  font-size: 11px;
  white-space: nowrap;
}

.ready-badge::before {
  content: "";
  display: inline-block;
  width: 6px;
  height: 6px;
  background: var(--signal);
  border-radius: 50%;
  margin-right: 6px;
}

.scene-body {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  min-height: 0;
}

.scene-viewport {
  position: relative;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
}

.scene-viewport.placing {
  cursor: crosshair;
}

.scene-viewport.drag-active {
  box-shadow: inset 0 0 0 3px var(--signal);
}

.scene-canvas {
  position: absolute;
  inset: 0;
  min-width: 0;
  min-height: 0;
  background: var(--ink);
}

.scene-canvas :deep(canvas) {
  display: block;
  width: 100%;
  height: 100%;
}

.stage-label {
  position: absolute;
  top: 18px;
  left: 20px;
  display: grid;
  gap: 4px;
  pointer-events: none;
  color: #fff;
  text-shadow: 0 1px 8px #1d2b27;
}

.stage-label span {
  font-size: 10px;
  letter-spacing: 0.2em;
  opacity: 0.85;
}

.stage-label strong {
  font-size: 16px;
  font-weight: 550;
}

.viewport-toolbar {
  position: absolute;
  bottom: 50px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  padding: 4px;
  gap: 3px;
  background: rgba(21, 29, 28, 0.91);
  border: 1px solid rgba(255, 255, 255, 0.15);
  border-radius: 10px;
  box-shadow: 0 8px 24px #0004;
  white-space: nowrap;
}

.viewport-toolbar button {
  border: 0;
  padding: 7px 11px;
  background: transparent;
  color: #e0e6df;
  cursor: pointer;
  font-size: 12px;
  border-radius: 6px;
}

.viewport-toolbar button:hover,
.viewport-toolbar button.active {
  background: var(--signal);
  color: var(--signal-ink);
}

.viewport-toolbar button:disabled {
  opacity: 0.35;
  cursor: default;
}

.viewport-status {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 9px 16px;
  color: #ccd6d0;
  background: rgba(20, 28, 25, 0.9);
  font-size: 11px;
  pointer-events: none;
}

.viewport-status .mono {
  white-space: nowrap;
  opacity: 0.65;
}

.drop-prompt,
.loading-indicator {
  position: absolute;
  left: 50%;
  top: 44%;
  transform: translate(-50%, -50%);
  padding: 14px 22px;
  background: rgba(20, 28, 25, 0.92);
  border: 1px solid var(--signal);
  border-radius: 12px;
  color: var(--text-primary);
  pointer-events: none;
  white-space: nowrap;
  font-size: 13px;
}

.loading-indicator span {
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 2px solid var(--text-muted);
  border-top-color: var(--signal);
  border-radius: 50%;
  animation: scene-spin 1s linear infinite;
  margin-right: 10px;
}

@keyframes scene-spin {
  to {
    transform: rotate(360deg);
  }
}

.scene-side {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  min-height: 0;
  border-left: 1px solid var(--line);
  background: var(--ink-soft);
}

.inspector-tabs {
  display: flex;
  padding: 4px 12px 0;
  background: var(--ink-raised);
  border-bottom: 1px solid var(--line);
  gap: 2px;
  overflow-x: auto;
}

.inspector-tabs button {
  flex: 1 1 auto;
  white-space: nowrap;
  padding: 10px 6px;
  border: 0;
  border-bottom: 2px solid transparent;
  background: transparent;
  color: var(--text-muted);
  font-size: 12px;
  cursor: pointer;
}

.inspector-tabs button.active {
  color: var(--signal);
  border-bottom-color: var(--signal);
  font-weight: 600;
}

.side-scroll {
  overflow-y: auto;
  padding: 14px 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

@media (max-width: 960px) {
  .scene-body {
    grid-template-columns: 1fr;
    grid-template-rows: minmax(260px, 1fr) auto;
  }

  .scene-side {
    border-left: 0;
    border-top: 1px solid var(--line);
    max-height: 44vh;
  }

  .ready-badge {
    display: none;
  }
}
</style>
