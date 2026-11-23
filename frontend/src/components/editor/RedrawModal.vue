<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from "vue";
import { Download, Sparkles, X } from "../icons";
import {
  fetchRedrawConfig,
  fetchSceneRedrawStatus,
  listSceneRedraws,
  pingRedrawComfyUI,
  setRedrawGatewayUpstream,
  submitSceneRedraw,
  type RedrawConfig,
  type RedrawHistoryItem,
  type RedrawPingResult,
  type RedrawProvider,
  type RedrawStylePreset,
  type RedrawTaskStatus,
} from "../../api/scene3d";

export type RedrawCaptureBundle = {
  screenshot: string;
  environment: string | null;
  protectionMask: string | null;
  width: number;
  height: number;
  equipmentCount: number;
};

const STORAGE_PREFS_KEY = "scene_minimax_redraw_prefs_v1";
const STORAGE_KEY_SESSION = "scene_minimax_redraw_apikey_v1";

const DEFAULT_PRESETS = [
  {
    key: "faithful" as const,
    label: "高保真还原原图",
    description: "严格锁定 3D 截图几何与机位，忠实复现原图航拍色彩、建筑材质与光照氛围",
  },
  {
    key: "detailed" as const,
    label: "强化写实细节",
    description: "保留几何布局的同时显著增强红砖、屋瓦、路面风化与植被真实细节",
  },
  {
    key: "match_3d_light" as const,
    label: "匹配当前3D光照",
    description: "沿用当前 3D 太阳方向与明暗情绪，仅迁移原图的真实材质肌理与摄影光学感",
  },
];

const props = defineProps<{
  visible: boolean;
  capture: RedrawCaptureBundle | null;
  sceneCode: string;
  sceneLabel: string;
  defaultReferenceUrl: string;
}>();

const emit = defineEmits<{
  close: [];
  recapture: [];
}>();

type ViewMode = "split" | "result" | "source" | "model" | "mask";

const config = ref<RedrawConfig | null>(null);
const provider = ref<RedrawProvider>("minimax_api");
const stylePreset = ref<RedrawStylePreset>("faithful");
const customPrompt = ref("");
const protectEquipment = ref(true);

const apiEndpointMode = ref<"cn" | "intl" | "custom">("cn");
const customBaseUrl = ref("https://api.minimaxi.com/v1");
const apiKey = ref("");
const showApiKey = ref(false);
const showConnPanel = ref(false);
const selectedModel = ref("MiniMax-H3-Image");
const selectedSize = ref("auto");

const comfyUrl = ref("");
const comfySteps = ref(6);
const comfyMegapixels = ref(4);
const comfySeed = ref("");
const pingBusy = ref(false);
const pingResult = ref<RedrawPingResult | null>(null);
const upstreamTunnelUrl = ref("");
const takeoverBusy = ref(false);
const takeoverMessage = ref("");
const takeoverError = ref(false);

const customRefDataUrl = ref("");
const customRefName = ref("");
const refFileInput = ref<HTMLInputElement | null>(null);

const submitting = ref(false);
const errorMessage = ref("");
const activeResult = ref<RedrawTaskStatus | null>(null);
const history = ref<RedrawHistoryItem[]>([]);
const elapsedSec = ref(0);

const viewMode = ref<ViewMode>("split");
const splitPercent = ref(52);
const showRevisedPrompt = ref(false);

let pollTimer: number | null = null;
let elapsedTimer: number | null = null;
let runStartedAt = 0;

function loadStoredPrefs(): void {
  try {
    const savedKey = sessionStorage.getItem(STORAGE_KEY_SESSION);
    if (savedKey) apiKey.value = savedKey;
    const raw = sessionStorage.getItem(STORAGE_PREFS_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    if (parsed.provider === "minimax_api" || parsed.provider === "comfyui") provider.value = parsed.provider;
    if (parsed.apiEndpointMode === "cn" || parsed.apiEndpointMode === "intl" || parsed.apiEndpointMode === "custom") {
      apiEndpointMode.value = parsed.apiEndpointMode;
    }
    if (typeof parsed.customBaseUrl === "string" && parsed.customBaseUrl) customBaseUrl.value = parsed.customBaseUrl;
    if (typeof parsed.comfyUrl === "string") comfyUrl.value = parsed.comfyUrl;
    if (typeof parsed.selectedModel === "string" && parsed.selectedModel) selectedModel.value = parsed.selectedModel;
    if (typeof parsed.selectedSize === "string" && parsed.selectedSize) selectedSize.value = parsed.selectedSize;
    if (
      parsed.stylePreset === "faithful" ||
      parsed.stylePreset === "detailed" ||
      parsed.stylePreset === "match_3d_light"
    ) {
      stylePreset.value = parsed.stylePreset;
    }
    if (typeof parsed.protectEquipment === "boolean") protectEquipment.value = parsed.protectEquipment;
    if (typeof parsed.comfySteps === "number") comfySteps.value = parsed.comfySteps;
    if (typeof parsed.comfyMegapixels === "number") comfyMegapixels.value = parsed.comfyMegapixels;
  } catch {
    // Ignore storage read failures in private mode.
  }
}

function savePrefs(): void {
  try {
    sessionStorage.setItem(
      STORAGE_PREFS_KEY,
      JSON.stringify({
        provider: provider.value,
        apiEndpointMode: apiEndpointMode.value,
        customBaseUrl: customBaseUrl.value,
        comfyUrl: comfyUrl.value,
        selectedModel: selectedModel.value,
        selectedSize: selectedSize.value,
        stylePreset: stylePreset.value,
        protectEquipment: protectEquipment.value,
        comfySteps: comfySteps.value,
        comfyMegapixels: comfyMegapixels.value,
      }),
    );
    if (apiKey.value.trim()) sessionStorage.setItem(STORAGE_KEY_SESSION, apiKey.value.trim());
    else sessionStorage.removeItem(STORAGE_KEY_SESSION);
  } catch {
    // Ignore storage write errors.
  }
}

watch(
  [
    provider,
    apiEndpointMode,
    customBaseUrl,
    comfyUrl,
    apiKey,
    selectedModel,
    selectedSize,
    stylePreset,
    protectEquipment,
    comfySteps,
    comfyMegapixels,
  ],
  savePrefs,
);

const presets = computed(() => config.value?.presets?.length ? config.value.presets : DEFAULT_PRESETS);
const activePresetMeta = computed(
  () => presets.value.find((p) => p.key === stylePreset.value) ?? presets.value[0],
);

const effectiveApiBaseUrl = computed(() => {
  if (apiEndpointMode.value === "cn") return "https://api.minimaxi.com/v1";
  if (apiEndpointMode.value === "intl") return "https://api.minimax.io/v1";
  return customBaseUrl.value.trim() || "https://api.minimaxi.com/v1";
});

const referencePreviewUrl = computed(() => customRefDataUrl.value || props.defaultReferenceUrl || "/scene-demo/village-source.png");

const modelOptions = computed(() => config.value?.providers.minimax_api.models ?? ["MiniMax-H3-Image", "MiniMax-H3-Image-Highspeed"]);

const hasServerMinimaxKey = computed(() => Boolean(config.value?.providers.minimax_api.configured));
const hasServerComfyUrl = computed(() => Boolean(config.value?.providers.comfyui.configured));

const beforeImageUrl = computed(() => {
  if (activeResult.value?.task_id && activeResult.value.source_url && !showCurrentLiveOnly.value) {
    return activeResult.value.source_url;
  }
  return props.capture?.screenshot || "";
});

const showCurrentLiveOnly = computed(
  () =>
    !!props.capture?.screenshot &&
    (!activeResult.value || activeResult.value.status !== "completed"),
);

const singleViewImage = computed(() => {
  if (viewMode.value === "source") return beforeImageUrl.value;
  if (viewMode.value === "model") return activeResult.value?.model_url || activeResult.value?.result_url || "";
  if (viewMode.value === "mask") return activeResult.value?.mask_url || props.capture?.protectionMask || "";
  return activeResult.value?.result_url || "";
});

function stopPolling(): void {
  if (pollTimer !== null) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
  if (elapsedTimer !== null) {
    window.clearInterval(elapsedTimer);
    elapsedTimer = null;
  }
}

async function refreshConfigAndHistory(): Promise<void> {
  try {
    const [cfg, items] = await Promise.all([fetchRedrawConfig(), listSceneRedraws()]);
    config.value = cfg;
    history.value = items;
    if (!sessionStorage.getItem(STORAGE_PREFS_KEY)) {
      provider.value = cfg.default_provider;
    }
    if (cfg.providers.minimax_api.default_model && !selectedModel.value) {
      selectedModel.value = cfg.providers.minimax_api.default_model;
    }
    if (cfg.providers.comfyui.url && (!comfyUrl.value || /\.trycloudflare\.com/i.test(comfyUrl.value))) {
      comfyUrl.value = cfg.providers.comfyui.url;
    }
    if (provider.value === "comfyui" && comfyUrl.value.trim() && !pingResult.value) {
      void handlePingComfy();
    }
    if (!cfg.providers.minimax_api.configured && !apiKey.value.trim()) {
      showConnPanel.value = true;
    }
  } catch {
    // Non-fatal: UI works with sensible defaults.
  }
}

watch(
  () => props.visible,
  (open) => {
    if (open) {
      loadStoredPrefs();
      errorMessage.value = "";
      void refreshConfigAndHistory();
    } else {
      stopPolling();
    }
  },
  { immediate: true },
);

function onKeyDown(event: KeyboardEvent): void {
  if (event.key === "Escape" && props.visible) {
    emit("close");
  }
}

window.addEventListener("keydown", onKeyDown);

onBeforeUnmount(() => {
  stopPolling();
  window.removeEventListener("keydown", onKeyDown);
});

async function handlePingComfy(): Promise<void> {
  const url = comfyUrl.value.trim();
  if (!url) {
    pingResult.value = { ok: false, url: "", comfyui_version: null, has_minimax_h3_image: false, message: "请先输入 ComfyUI 地址" };
    return;
  }
  pingBusy.value = true;
  pingResult.value = null;
  try {
    pingResult.value = await pingRedrawComfyUI(url);
  } catch (error) {
    pingResult.value = {
      ok: false,
      url,
      comfyui_version: null,
      has_minimax_h3_image: false,
      message: error instanceof Error ? error.message : String(error),
    };
  } finally {
    pingBusy.value = false;
  }
}

async function handleTakeoverUpstream(): Promise<void> {
  const upstream = upstreamTunnelUrl.value.trim();
  if (!upstream) {
    takeoverError.value = true;
    takeoverMessage.value = "请先填入正在运行的 trycloudflare 隧道地址";
    return;
  }
  const gw = comfyUrl.value.trim() || "http://127.0.0.1:8188";
  takeoverBusy.value = true;
  takeoverError.value = false;
  takeoverMessage.value = "";
  try {
    const res = await setRedrawGatewayUpstream(gw, upstream);
    comfyUrl.value = res.gateway_url;
    takeoverMessage.value = `已热接管上游隧道：${res.upstream_url}`;
    await handlePingComfy();
  } catch (error) {
    takeoverError.value = true;
    takeoverMessage.value = error instanceof Error ? error.message : String(error);
  } finally {
    takeoverBusy.value = false;
  }
}

function triggerRefUpload(): void {
  refFileInput.value?.click();
}

function onRefFileSelected(event: Event): void {
  const input = event.target as HTMLInputElement;
  const file = input.files?.[0];
  input.value = "";
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    if (typeof reader.result === "string") {
      customRefDataUrl.value = reader.result;
      customRefName.value = file.name;
    }
  };
  reader.readAsDataURL(file);
}

function resetRefToDefault(): void {
  customRefDataUrl.value = "";
  customRefName.value = "";
}

async function pollTask(taskId: string): Promise<void> {
  if (!props.visible) return;
  try {
    const state = await fetchSceneRedrawStatus(taskId);
    if (state.status === "completed") {
      stopPolling();
      submitting.value = false;
      activeResult.value = state;
      viewMode.value = "split";
      splitPercent.value = 55;
      try {
        history.value = await listSceneRedraws();
      } catch {
        // Ignore history refresh error.
      }
      return;
    }
    if (state.status === "failed") {
      stopPolling();
      submitting.value = false;
      errorMessage.value = state.error_message || "MiniMax-H3-Image 重绘失败";
      return;
    }
    pollTimer = window.setTimeout(() => {
      void pollTask(taskId);
    }, 1800);
  } catch (error) {
    stopPolling();
    submitting.value = false;
    errorMessage.value = `查询重绘状态失败：${error instanceof Error ? error.message : String(error)}`;
  }
}

async function startRedraw(): Promise<void> {
  if (submitting.value || !props.capture?.screenshot) return;
  errorMessage.value = "";
  if (provider.value === "minimax_api" && !hasServerMinimaxKey.value && !apiKey.value.trim()) {
    showConnPanel.value = true;
    errorMessage.value = "请先填写 MiniMax API Key（或于服务端 .env 中配置 MINIMAX_API_KEY）";
    return;
  }
  if (provider.value === "comfyui" && !hasServerComfyUrl.value && !comfyUrl.value.trim()) {
    errorMessage.value = "请先填写 ComfyUI 服务地址（如 https://xxx.trycloudflare.com）";
    return;
  }

  submitting.value = true;
  runStartedAt = performance.now();
  elapsedSec.value = 0;
  stopPolling();
  elapsedTimer = window.setInterval(() => {
    elapsedSec.value = Number(((performance.now() - runStartedAt) / 1000).toFixed(1));
  }, 200);

  const seedNum = comfySeed.value.trim() ? Number(comfySeed.value.trim()) : null;
  try {
    const res = await submitSceneRedraw({
      provider: provider.value,
      source_image: props.capture.screenshot,
      reference_image: customRefDataUrl.value || null,
      environment_image: props.capture.environment,
      protection_mask: props.capture.protectionMask,
      protect_equipment: protectEquipment.value,
      scene_code: props.sceneCode,
      scene_label: props.sceneLabel,
      style_preset: stylePreset.value,
      prompt: customPrompt.value.trim(),
      model: selectedModel.value,
      size: selectedSize.value,
      api_key: provider.value === "minimax_api" && apiKey.value.trim() ? apiKey.value.trim() : undefined,
      api_base_url: provider.value === "minimax_api" ? effectiveApiBaseUrl.value : undefined,
      comfyui_url: provider.value === "comfyui" && comfyUrl.value.trim() ? comfyUrl.value.trim() : undefined,
      steps: comfySteps.value,
      megapixels: comfyMegapixels.value,
      seed: Number.isFinite(seedNum) ? seedNum : null,
    });
    await pollTask(res.task_id);
  } catch (error) {
    stopPolling();
    submitting.value = false;
    errorMessage.value = error instanceof Error ? error.message : String(error);
  }
}

function selectHistoryItem(item: RedrawHistoryItem): void {
  activeResult.value = item;
  errorMessage.value = "";
  viewMode.value = "split";
  splitPercent.value = 55;
}
</script>

<template>
  <Teleport to="body">
    <div v-if="visible" class="redraw-overlay" role="dialog" aria-modal="true" aria-labelledby="redraw-modal-title">
      <button type="button" class="redraw-scrim" aria-label="关闭截图渲染窗口" @click="emit('close')" />
      <section class="redraw-modal">
        <header class="redraw-header">
          <div class="redraw-title">
            <span class="redraw-badge-icon"><Sparkles :size="16" /></span>
            <div>
              <span class="mono subcap">MINIMAX-H3-IMAGE · STYLE TRANSFER</span>
              <h2 id="redraw-modal-title">截图并渲染 · 原图风格重绘</h2>
            </div>
            <span class="scene-chip">{{ sceneLabel }}</span>
          </div>
          <div class="redraw-header-actions">
            <button type="button" class="hdr-ghost" :disabled="submitting" @click="emit('recapture')">
              重新截取当前视角
            </button>
            <button type="button" class="icon-close" aria-label="关闭" @click="emit('close')">
              <X :size="18" />
            </button>
          </div>
        </header>

        <div class="redraw-body">
          <aside class="redraw-controls">
            <div class="dual-cards">
              <div class="thumb-card">
                <div class="thumb-head">
                  <span class="mono">Picture 1 · 3D 截图</span>
                  <em v-if="capture" class="mono">{{ capture.width }}×{{ capture.height }}</em>
                </div>
                <div class="thumb-frame">
                  <img v-if="capture?.screenshot" :src="capture.screenshot" alt="3D 视口截图" />
                  <span v-if="capture && capture.equipmentCount > 0" class="thumb-pill signal">
                    含 {{ capture.equipmentCount }} 个装备
                  </span>
                </div>
              </div>

              <div class="thumb-card">
                <div class="thumb-head">
                  <span class="mono">Picture 2 · 原图参考</span>
                  <button type="button" class="mini-link" @click="triggerRefUpload">更换</button>
                </div>
                <div class="thumb-frame">
                  <img :src="referencePreviewUrl" alt="原图风格参考" />
                  <span class="thumb-pill">{{ customRefName || "默认场景原图" }}</span>
                </div>
                <button v-if="customRefDataUrl" type="button" class="mini-link reset-ref" @click="resetRefToDefault">
                  恢复默认原图
                </button>
                <input
                  ref="refFileInput"
                  type="file"
                  accept="image/png,image/jpeg,image/webp"
                  hidden
                  @change="onRefFileSelected"
                />
              </div>
            </div>

            <section class="ctrl-section">
              <h3 class="mono">重绘风格预设</h3>
              <div class="preset-grid" role="group" aria-label="风格预设">
                <button
                  v-for="item in presets"
                  :key="item.key"
                  type="button"
                  class="preset-btn"
                  :class="{ active: stylePreset === item.key }"
                  @click="stylePreset = item.key"
                >
                  {{ item.label }}
                </button>
              </div>
              <p class="hint">{{ activePresetMeta.description }}</p>
              <label class="field-stack">
                <span>补充风格描述（可选）</span>
                <textarea
                  v-model="customPrompt"
                  rows="2"
                  placeholder="例如：保持正午阳光方向与红砖风化细节，路面略带干燥尘土感…"
                />
              </label>
            </section>

            <section class="ctrl-section">
              <label class="toggle-row">
                <input v-model="protectEquipment" type="checkbox" />
                <div>
                  <strong>保留 3D 装备原像素（遮罩保护）</strong>
                  <p class="hint">
                    <template v-if="capture && capture.equipmentCount > 0">
                      已自动分离场景背景与深度遮罩：仅重绘环境，坦克等 3D 装备像素与阴影遮挡关系精确保留。
                    </template>
                    <template v-else>
                      当前视角未摆放 3D 装备，将对整帧建筑环境执行风格重绘；放入装备后自动启用遮罩合成。
                    </template>
                  </p>
                </div>
              </label>
            </section>

            <section class="ctrl-section">
              <div class="section-row">
                <h3 class="mono">推理引擎 · MINIMAX-H3-IMAGE</h3>
                <button
                  v-if="provider === 'minimax_api'"
                  type="button"
                  class="mini-link"
                  @click="showConnPanel = !showConnPanel"
                >
                  {{ showConnPanel ? "收起连接配置" : "连接与密钥" }}
                </button>
              </div>

              <div class="engine-switch" role="tablist" aria-label="推理引擎选择">
                <button
                  type="button"
                  role="tab"
                  :aria-selected="provider === 'minimax_api'"
                  :class="{ active: provider === 'minimax_api' }"
                  @click="provider = 'minimax_api'"
                >
                  MiniMax 云端 API
                  <em class="dot" :class="{ on: hasServerMinimaxKey || !!apiKey.trim() }" />
                </button>
                <button
                  type="button"
                  role="tab"
                  :aria-selected="provider === 'comfyui'"
                  :class="{ active: provider === 'comfyui' }"
                  @click="provider = 'comfyui'"
                >
                  ComfyUI 工作流
                  <em class="dot" :class="{ on: hasServerComfyUrl || !!comfyUrl.trim() }" />
                </button>
              </div>

              <template v-if="provider === 'minimax_api'">
                <div class="two-col">
                  <label class="field-stack">
                    <span>模型版本</span>
                    <select v-model="selectedModel">
                      <option v-for="m in modelOptions" :key="m" :value="m">{{ m }}</option>
                    </select>
                  </label>
                  <label class="field-stack">
                    <span>输出尺寸</span>
                    <select v-model="selectedSize">
                      <option value="auto">自动匹配 (auto)</option>
                      <option value="match">同截图尺寸 (match)</option>
                      <option value="2K">2K 高清 (同比例)</option>
                      <option value="4K">4K 超清 (同比例)</option>
                    </select>
                  </label>
                </div>

                <div v-if="showConnPanel" class="conn-box">
                  <label class="field-stack">
                    <span>API 接入点</span>
                    <select v-model="apiEndpointMode">
                      <option value="cn">国内站 (api.minimaxi.com/v1)</option>
                      <option value="intl">国际站 (api.minimax.io/v1)</option>
                      <option value="custom">自定义兼容网关</option>
                    </select>
                  </label>
                  <label v-if="apiEndpointMode === 'custom'" class="field-stack">
                    <span>自定义 Base URL</span>
                    <input v-model="customBaseUrl" type="url" placeholder="https://your-gateway.example/v1" />
                  </label>
                  <label class="field-stack">
                    <span class="label-inline">
                      <span>MiniMax API Key {{ hasServerMinimaxKey ? "（服务端已配置，留空即用 .env）" : "" }}</span>
                      <span class="key-actions">
                        <button type="button" class="mini-link" @click="showApiKey = !showApiKey">
                          {{ showApiKey ? "隐藏" : "显示" }}
                        </button>
                        <button v-if="apiKey" type="button" class="mini-link" @click="apiKey = ''">清除</button>
                      </span>
                    </span>
                    <input
                      v-model="apiKey"
                      :type="showApiKey ? 'text' : 'password'"
                      autocomplete="off"
                      placeholder="填写临时 API Key（仅存当前标签页内存，不落盘）"
                    />
                  </label>
                </div>
              </template>

              <template v-else>
                <div class="conn-box">
                  <label class="field-stack">
                    <span>ComfyUI 服务地址 {{ hasServerComfyUrl ? "（留空默认使用服务端环境变量）" : "" }}</span>
                    <div class="ping-row">
                      <input
                        v-model="comfyUrl"
                        type="url"
                        placeholder="http://127.0.0.1:8188 或 https://xxx.trycloudflare.com"
                      />
                      <button type="button" class="ping-btn" :disabled="pingBusy || !comfyUrl.trim()" @click="handlePingComfy">
                        {{ pingBusy ? "检测中…" : "测试连通性" }}
                      </button>
                    </div>
                  </label>
                  <p
                    v-if="pingResult"
                    class="ping-status"
                    :class="{
                      ok: pingResult.ok && pingResult.has_minimax_h3_image,
                      warn: (pingResult.ok && !pingResult.has_minimax_h3_image) || ['reconnecting', 'stopped', 'idle'].includes(pingResult.gateway_state || ''),
                    }"
                  >
                    {{ pingResult.message }}
                  </p>
                  <label class="field-stack">
                    <span>热接管已有隧道 (POST /_gateway/set-upstream)</span>
                    <div class="ping-row">
                      <input
                        v-model="upstreamTunnelUrl"
                        type="url"
                        placeholder="https://xxx.trycloudflare.com（可选：接管浏览器中已运行的隧道）"
                      />
                      <button
                        type="button"
                        class="ping-btn"
                        :disabled="takeoverBusy || !upstreamTunnelUrl.trim()"
                        @click="handleTakeoverUpstream"
                      >
                        {{ takeoverBusy ? "接管中…" : "热接管" }}
                      </button>
                    </div>
                  </label>
                  <p v-if="takeoverMessage" class="ping-status" :class="{ ok: !takeoverError }">
                    {{ takeoverMessage }}
                  </p>
                  <p class="gateway-tip">
                    固定地址 <code>http://127.0.0.1:8188</code> 已启用；不用时执行 <code>.\stop_colab.ps1</code> 停止计费。
                  </p>
                  <div class="three-col">
                    <label class="field-stack">
                      <span>步数 (steps)</span>
                      <input v-model.number="comfySteps" type="number" min="4" max="30" />
                    </label>
                    <label class="field-stack">
                      <span>目标像素</span>
                      <select v-model.number="comfyMegapixels">
                        <option :value="2">2 MP</option>
                        <option :value="4">4 MP（推荐）</option>
                        <option :value="8">8 MP</option>
                        <option :value="16">16 MP</option>
                      </select>
                    </label>
                    <label class="field-stack">
                      <span>随机种子</span>
                      <input v-model="comfySeed" type="text" placeholder="随机" />
                    </label>
                  </div>
                </div>
              </template>
            </section>

            <div class="submit-block">
              <button type="button" class="render-cta" :disabled="submitting || !capture?.screenshot" @click="startRedraw">
                <Sparkles :size="15" />
                {{ submitting ? `MiniMax-H3-Image 渲染中… (${elapsedSec.toFixed(1)}s)` : "开始截图重绘" }}
              </button>
              <p v-if="errorMessage" class="err-box">{{ errorMessage }}</p>
            </div>
          </aside>

          <main class="redraw-stage">
            <div class="stage-bar">
              <div class="view-pills" role="group" aria-label="对比视图模式">
                <button
                  type="button"
                  :class="{ active: viewMode === 'split' }"
                  :disabled="!activeResult?.result_url"
                  @click="viewMode = 'split'"
                >
                  左右对比
                </button>
                <button
                  type="button"
                  :class="{ active: viewMode === 'result' }"
                  :disabled="!activeResult?.result_url"
                  @click="viewMode = 'result'"
                >
                  仅看重绘
                </button>
                <button
                  type="button"
                  :class="{ active: viewMode === 'source' }"
                  :disabled="!beforeImageUrl"
                  @click="viewMode = 'source'"
                >
                  仅看 3D 截图
                </button>
                <button
                  v-if="activeResult?.mask_url"
                  type="button"
                  :class="{ active: viewMode === 'model' }"
                  @click="viewMode = 'model'"
                >
                  纯背景重绘
                </button>
                <button
                  v-if="activeResult?.mask_url || capture?.protectionMask"
                  type="button"
                  :class="{ active: viewMode === 'mask' }"
                  @click="viewMode = 'mask'"
                >
                  装备遮罩
                </button>
              </div>

              <div v-if="activeResult?.result_url" class="result-actions">
                <a class="dl-btn" :href="`${activeResult.result_url}?download=1`" download>
                  <Download :size="14" /> 下载重绘成图 PNG
                </a>
              </div>
            </div>

            <div class="compare-canvas">
              <template v-if="activeResult?.result_url && viewMode === 'split'">
                <div class="split-wrap">
                  <img class="layer-base" :src="activeResult.result_url" alt="MiniMax-H3-Image 重绘结果" />
                  <div
                    class="layer-before"
                    :style="{ clipPath: `inset(0 ${100 - splitPercent}% 0 0)` }"
                  >
                    <img :src="beforeImageUrl" alt="3D 原始截图" />
                  </div>
                  <div class="split-divider" :style="{ left: `${splitPercent}%` }">
                    <span class="split-handle">⇆</span>
                  </div>
                  <input
                    v-model.number="splitPercent"
                    class="split-range"
                    type="range"
                    min="0"
                    max="100"
                    aria-label="原图与重绘对比滑块"
                  />
                  <span class="corner-tag left">3D 原始截图</span>
                  <span class="corner-tag right">MiniMax-H3-Image 重绘</span>
                </div>
              </template>

              <template v-else-if="singleViewImage">
                <img class="single-preview" :src="singleViewImage" alt="画面预览" />
              </template>

              <div v-else class="empty-stage">
                <Sparkles :size="28" />
                <h3>准备就绪，等待发起风格重绘</h3>
                <p>
                  系统已自动捕获当前 3D 视角，并关联原图参考。点击左下角「开始截图重绘」，MiniMax-H3-Image
                  将保留您的三维建筑构图并还原航拍实景风格。
                </p>
              </div>

              <div v-if="submitting" class="stage-loading">
                <span class="spinner" />
                <div>
                  <strong>正在调用 MiniMax-H3-Image 渲染…（{{ elapsedSec.toFixed(1) }}s）</strong>
                  <p>高分辨率参考重绘通常需要 15–60 秒，请保持窗口开启</p>
                </div>
              </div>
            </div>

            <div v-if="activeResult?.result_url" class="result-meta-bar">
              <span class="meta-pill">{{ activeResult.provider === "comfyui" ? "ComfyUI" : "MiniMax API" }}</span>
              <span v-if="activeResult.model" class="meta-pill">{{ activeResult.model }}</span>
              <span v-if="activeResult.elapsed_sec" class="meta-pill">耗时 {{ activeResult.elapsed_sec }}s</span>
              <span v-if="activeResult.output_size" class="meta-pill">
                {{ activeResult.output_size[0] }}×{{ activeResult.output_size[1] }}
              </span>
              <span v-if="activeResult.equipment_protected" class="meta-pill protected">
                装备像素已锁定（{{ activeResult.protection_coverage_pct }}%）
              </span>
              <button
                v-if="activeResult.revised_prompt || activeResult.effective_prompt"
                type="button"
                class="mini-link meta-prompt-btn"
                @click="showRevisedPrompt = !showRevisedPrompt"
              >
                {{ showRevisedPrompt ? "收起提示词" : "查看提示词" }}
              </button>
            </div>
            <p v-if="showRevisedPrompt && (activeResult?.revised_prompt || activeResult?.effective_prompt)" class="revised-box mono">
              {{ activeResult?.revised_prompt || activeResult?.effective_prompt }}
            </p>

            <section v-if="history.length" class="history-strip">
              <h4 class="mono">历史重绘记录 ({{ history.length }})</h4>
              <div class="history-list">
                <button
                  v-for="item in history"
                  :key="item.task_id"
                  type="button"
                  class="history-card"
                  :class="{ active: activeResult?.task_id === item.task_id }"
                  @click="selectHistoryItem(item)"
                >
                  <img v-if="item.result_url" :src="item.result_url" :alt="item.task_id" />
                  <span class="mono">{{ item.elapsed_sec ? `${item.elapsed_sec}s` : "完成" }}</span>
                </button>
              </div>
            </section>
          </main>
        </div>
      </section>
    </div>
  </Teleport>
</template>

<style scoped>
.redraw-overlay {
  position: fixed;
  inset: 0;
  z-index: 5000;
  display: grid;
  place-items: center;
  padding: 18px;
}

.redraw-scrim {
  position: absolute;
  inset: 0;
  border: 0;
  background: rgba(8, 10, 8, 0.78);
  cursor: pointer;
}

.redraw-modal {
  position: relative;
  width: min(1360px, 100%);
  height: min(880px, calc(100vh - 36px));
  display: grid;
  grid-template-rows: 56px minmax(0, 1fr);
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-xl);
  background: var(--ink-raised);
  box-shadow: var(--shadow-floating);
  overflow: hidden;
}

.redraw-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 0 18px;
  border-bottom: 1px solid var(--line);
  background: var(--ink);
}

.redraw-title {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.redraw-badge-icon {
  display: grid;
  place-items: center;
  width: 30px;
  height: 30px;
  border-radius: 8px;
  background: var(--signal);
  color: var(--signal-ink);
  flex-shrink: 0;
}

.subcap {
  display: block;
  font-size: 9px;
  letter-spacing: 0.14em;
  color: var(--text-muted);
}

.redraw-title h2 {
  margin: 0;
  font-size: 15px;
  font-weight: 650;
}

.scene-chip {
  padding: 3px 10px;
  border: 1px solid rgba(201, 255, 87, 0.35);
  border-radius: 99px;
  background: rgba(201, 255, 87, 0.12);
  color: var(--signal);
  font-size: 11px;
  white-space: nowrap;
}

.redraw-header-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.hdr-ghost {
  padding: 6px 11px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  background: var(--surface-muted);
  font-size: 12px;
  cursor: pointer;
}

.hdr-ghost:hover:enabled {
  border-color: var(--signal);
  color: var(--signal);
}

.icon-close {
  display: grid;
  place-items: center;
  width: 32px;
  height: 32px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--surface-muted);
  cursor: pointer;
}

.redraw-body {
  display: grid;
  grid-template-columns: 380px minmax(0, 1fr);
  min-height: 0;
}

.redraw-controls {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 14px 16px 16px;
  border-right: 1px solid var(--line);
  background: var(--ink-soft);
  overflow-y: auto;
}

.dual-cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}

.thumb-card {
  display: grid;
  gap: 5px;
}

.thumb-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 10px;
  color: var(--text-muted);
}

.thumb-head em {
  font-style: normal;
}

.thumb-frame {
  position: relative;
  aspect-ratio: 16 / 10;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: var(--ink);
}

.thumb-frame img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.thumb-pill {
  position: absolute;
  left: 5px;
  bottom: 5px;
  padding: 2px 6px;
  border-radius: 4px;
  background: rgba(14, 18, 14, 0.85);
  color: var(--paper);
  font-size: 10px;
  max-width: calc(100% - 10px);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.thumb-pill.signal {
  color: var(--signal);
  border: 1px solid rgba(201, 255, 87, 0.45);
}

.mini-link {
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--signal);
  font-size: 11px;
  cursor: pointer;
}

.reset-ref {
  justify-self: end;
}

.ctrl-section {
  display: grid;
  gap: 8px;
  padding-top: 10px;
  border-top: 1px solid var(--line);
}

.ctrl-section h3 {
  margin: 0;
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--text-muted);
}

.section-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.preset-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 6px;
}

.preset-btn {
  padding: 7px 6px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  color: var(--text-secondary);
  font-size: 11px;
  cursor: pointer;
  text-align: center;
}

.preset-btn.active {
  border-color: var(--signal);
  background: rgba(201, 255, 87, 0.14);
  color: var(--signal);
  font-weight: 600;
}

.hint {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--text-muted);
}

.field-stack {
  display: grid;
  gap: 4px;
  font-size: 11px;
  color: var(--text-secondary);
}

.field-stack textarea,
.field-stack input,
.field-stack select {
  width: 100%;
  min-width: 0;
  padding: 7px 9px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--ink);
  color: var(--text-primary);
  font-size: 12px;
}

.toggle-row {
  display: flex;
  align-items: flex-start;
  gap: 9px;
  font-size: 12px;
  cursor: pointer;
}

.toggle-row input {
  margin-top: 3px;
  accent-color: var(--signal);
}

.engine-switch {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px;
}

.engine-switch button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 8px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--surface);
  font-size: 12px;
  cursor: pointer;
}

.engine-switch button.active {
  border-color: var(--signal);
  color: var(--signal);
  font-weight: 600;
}

.dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: rgba(242, 239, 230, 0.25);
}

.dot.on {
  background: var(--signal);
}

.two-col {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.three-col {
  display: grid;
  grid-template-columns: 1fr 1.1fr 1fr;
  gap: 8px;
}

.conn-box {
  display: grid;
  gap: 8px;
  padding: 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  background: rgba(18, 21, 18, 0.55);
}

.label-inline {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 6px;
}

.key-actions {
  display: inline-flex;
  gap: 8px;
}

.ping-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 6px;
}

.ping-btn {
  padding: 0 10px;
  border: 1px solid var(--line-strong);
  border-radius: var(--radius-sm);
  background: var(--surface);
  font-size: 11px;
  white-space: nowrap;
  cursor: pointer;
}

.ping-btn:hover:enabled {
  border-color: var(--signal);
  color: var(--signal);
}

.ping-status {
  margin: 0;
  font-size: 11px;
  color: var(--danger);
}

.ping-status.ok {
  color: var(--success);
}

.ping-status.warn {
  color: var(--amber);
}

.gateway-tip {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
  color: var(--text-muted);
}

.gateway-tip code {
  padding: 1px 4px;
  border-radius: 3px;
  background: rgba(255, 255, 255, 0.08);
  color: var(--text-secondary);
}

.submit-block {
  margin-top: auto;
  display: grid;
  gap: 8px;
  padding-top: 8px;
}

.render-cta {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  width: 100%;
  padding: 11px 14px;
  border: 0;
  border-radius: var(--radius-sm);
  background: var(--signal);
  color: var(--signal-ink);
  font-size: 13px;
  font-weight: 700;
  cursor: pointer;
}

.render-cta:disabled {
  opacity: 0.55;
}

.err-box {
  margin: 0;
  padding: 8px 10px;
  border: 1px solid rgba(255, 118, 87, 0.45);
  border-radius: var(--radius-sm);
  background: rgba(255, 118, 87, 0.12);
  color: var(--danger);
  font-size: 11px;
  line-height: 1.45;
}

.redraw-stage {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr) auto auto auto;
  gap: 10px;
  padding: 14px 16px;
  min-height: 0;
  background: var(--ink);
}

.stage-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
}

.view-pills {
  display: flex;
  gap: 4px;
  padding: 3px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--ink-raised);
}

.view-pills button {
  padding: 5px 10px;
  border: 0;
  border-radius: 4px;
  background: transparent;
  color: var(--text-muted);
  font-size: 12px;
  cursor: pointer;
}

.view-pills button.active {
  background: var(--signal);
  color: var(--signal-ink);
  font-weight: 600;
}

.view-pills button:disabled {
  opacity: 0.35;
}

.dl-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 12px;
  border: 1px solid rgba(201, 255, 87, 0.45);
  border-radius: var(--radius-sm);
  background: rgba(201, 255, 87, 0.14);
  color: var(--signal);
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
}

.compare-canvas {
  position: relative;
  min-height: 0;
  display: grid;
  place-items: center;
  border: 1px solid var(--line);
  border-radius: var(--radius-md);
  background: #0c0e0c;
  overflow: hidden;
}

.split-wrap {
  position: relative;
  width: 100%;
  height: 100%;
  user-select: none;
}

.layer-base,
.layer-before img,
.single-preview {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.layer-before {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.split-divider {
  position: absolute;
  top: 0;
  bottom: 0;
  width: 2px;
  background: var(--signal);
  pointer-events: none;
  transform: translateX(-1px);
}

.split-handle {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border-radius: 50%;
  background: var(--signal);
  color: var(--signal-ink);
  font-weight: 700;
  font-size: 13px;
  box-shadow: 0 2px 10px #0008;
}

.split-range {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  margin: 0;
  opacity: 0;
  cursor: ew-resize;
}

.corner-tag {
  position: absolute;
  top: 10px;
  padding: 4px 8px;
  border-radius: 4px;
  background: rgba(14, 18, 14, 0.8);
  color: var(--paper);
  font-size: 11px;
  pointer-events: none;
}

.corner-tag.left {
  left: 10px;
}

.corner-tag.right {
  right: 10px;
  color: var(--signal);
}

.empty-stage {
  display: grid;
  justify-items: center;
  gap: 8px;
  max-width: 440px;
  padding: 24px;
  text-align: center;
  color: var(--text-muted);
}

.empty-stage h3 {
  margin: 0;
  color: var(--text-primary);
  font-size: 15px;
}

.empty-stage p {
  margin: 0;
  font-size: 12px;
  line-height: 1.55;
}

.stage-loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: rgba(12, 15, 12, 0.78);
  backdrop-filter: blur(4px);
}

.stage-loading p {
  margin: 3px 0 0;
  font-size: 12px;
  color: var(--text-muted);
}

.spinner {
  width: 24px;
  height: 24px;
  border: 3px solid rgba(255, 255, 255, 0.2);
  border-top-color: var(--signal);
  border-radius: 50%;
  animation: redraw-spin 1s linear infinite;
}

@keyframes redraw-spin {
  to {
    transform: rotate(360deg);
  }
}

.result-meta-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.meta-pill {
  padding: 3px 8px;
  border: 1px solid var(--line);
  border-radius: 99px;
  font-size: 11px;
  color: var(--text-secondary);
}

.meta-pill.protected {
  border-color: rgba(201, 255, 87, 0.4);
  color: var(--signal);
}

.meta-prompt-btn {
  margin-left: auto;
}

.revised-box {
  margin: 0;
  padding: 8px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  background: var(--ink-raised);
  font-size: 11px;
  color: var(--text-secondary);
  max-height: 76px;
  overflow-y: auto;
}

.history-strip {
  display: grid;
  gap: 6px;
}

.history-strip h4 {
  margin: 0;
  font-size: 10px;
  letter-spacing: 0.12em;
  color: var(--text-muted);
}

.history-list {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 2px;
}

.history-card {
  position: relative;
  flex: 0 0 92px;
  height: 58px;
  padding: 0;
  border: 1px solid var(--line);
  border-radius: var(--radius-sm);
  overflow: hidden;
  background: var(--ink-raised);
  cursor: pointer;
}

.history-card.active {
  border-color: var(--signal);
  box-shadow: 0 0 0 1px var(--signal);
}

.history-card img {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.history-card span {
  position: absolute;
  right: 3px;
  bottom: 3px;
  padding: 1px 4px;
  border-radius: 3px;
  background: rgba(12, 14, 12, 0.84);
  font-size: 9px;
  color: var(--paper);
}

@media (max-width: 960px) {
  .redraw-body {
    grid-template-columns: 1fr;
    grid-template-rows: auto minmax(260px, 1fr);
  }

  .redraw-controls {
    border-right: 0;
    border-bottom: 1px solid var(--line);
    max-height: 45vh;
  }
}
</style>
