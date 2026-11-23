import { deleteJson, getJson, postJson, putJson } from "./http";

export type Asset3D = {
  code: string;
  label: string;
  category: string;
  has_model: boolean;
  url: string | null;
  dimensions: { length: number; width: number; height: number };
  origin: [number, number, number] | number[];
  up_axis: string;
  yaw_offset: number;
  default_scale: number;
  thumbnail: string | null;
};

export type SceneObjectBBox = {
  name: string;
  xmin: number;
  ymin: number;
  xmax: number;
  ymax: number;
  pose?: string;
  truncated?: number;
  difficult?: number;
};

export type SceneFrame = {
  image_base64: string;
  width: number;
  height: number;
  objects: SceneObjectBBox[];
  filename?: string;
  folder?: string;
};

export type SceneExportResult = {
  task_id: string;
  status: string;
};

export type SceneObjectState = {
  id?: string | null;
  asset_code: string;
  label?: string | null;
  kind?: string;
  position: [number, number, number] | number[];
  rotation: [number, number, number] | number[];
  scale: [number, number, number] | number[];
  sort_order?: number | null;
};

export type SceneLightingState = {
  azimuth: number;
  elevation: number;
  intensity: number;
  ambient: number;
  temperature: number;
};

export type SceneSessionState = {
  id: string;
  title: string;
  lighting: SceneLightingState;
  ground_texture: string | null;
  objects: SceneObjectState[];
  created_at?: string;
  updated_at?: string;
};

export type SceneSessionSummary = {
  id: string;
  title: string;
  ground_texture: string | null;
  created_at: string;
  updated_at: string;
  object_count: number;
};

type ApiEnvelope<T> = {
  data: T;
  message: string;
  status: number;
};

export async function submitSceneRefine(payload: {
  base64_image: string; protection_mask: string; environment_image: string; prompt: string;
  workflow_template?: string; steps: number; cfg_scale: number;
}): Promise<SceneExportResult> {
  const response = await postJson<ApiEnvelope<SceneExportResult>>("/api/generate", {
    task_id: `scene_refine_${Date.now()}`, type: "scene_refine", batch: 1,
    params: payload,
  });
  return response.data;
}

export async function pollSceneRefine(taskId: string): Promise<{
  status: string; error_message?: string; result?: Array<{ image: string }>;
}> {
  const response = await postJson<ApiEnvelope<{
    status: string; error_message?: string; result?: Array<{ image: string }>;
  }>>("/api/get_task_result", { task_id: taskId });
  return response.data;
}

export async function fetchAssets3D(reload = false): Promise<Asset3D[]> {
  const suffix = reload ? "?reload=true" : "";
  const response = await getJson<ApiEnvelope<{ assets: Asset3D[] }>>(
    `/api/assets/3d${suffix}`,
  );
  return response.data?.assets ?? [];
}

/** Submit rendered multi-angle frames for image + VOC XML persistence. */
export async function exportSceneFrames(
  frames: SceneFrame[],
  options: { map_id?: string; weather?: string; scene?: string } = {},
): Promise<SceneExportResult> {
  const response = await postJson<ApiEnvelope<SceneExportResult>>("/api/generate", {
    task_id: `scene_export_${Date.now()}`,
    type: "scene_export",
    batch: 1,
    params: {
      frames,
      map_id: options.map_id ?? null,
      weather: options.weather ?? null,
      scene: options.scene ?? null,
    },
  });
  return response.data;
}

export async function listSceneSessions(): Promise<SceneSessionSummary[]> {
  const response = await getJson<ApiEnvelope<{ sessions: SceneSessionSummary[] }>>(
    "/api/scene/sessions",
  );
  return response.data?.sessions ?? [];
}

export async function createSceneSession(payload: {
  title?: string;
  lighting?: Partial<SceneLightingState>;
  objects?: SceneObjectState[];
  ground_texture?: string | null;
}): Promise<SceneSessionState> {
  const response = await postJson<ApiEnvelope<SceneSessionState>>(
    "/api/scene/sessions",
    payload,
  );
  return response.data;
}

export async function getSceneSession(id: string): Promise<SceneSessionState> {
  const response = await getJson<ApiEnvelope<SceneSessionState>>(
    `/api/scene/sessions/${id}`,
  );
  return response.data;
}

export async function saveSceneSession(
  id: string,
  payload: {
    title?: string;
    lighting?: Partial<SceneLightingState>;
    objects?: SceneObjectState[];
    ground_texture?: string | null;
  },
): Promise<SceneSessionState> {
  const response = await putJson<ApiEnvelope<SceneSessionState>>(
    `/api/scene/sessions/${id}`,
    payload,
  );
  return response.data;
}

export async function deleteSceneSession(id: string): Promise<void> {
  await deleteJson(`/api/scene/sessions/${id}`);
}

export type SceneMeshInfo = {
  code: string;
  label: string;
  kind: string;
  map_id?: string | null;
  url: string;
  has_model: boolean;
  splat_url?: string | null;
  splat_transform?: {
    position?: number[];
    rotation?: number[];
    scale?: number[];
  } | null;
  source_url?: string;
  plan_url?: string;
  report_url?: string;
  ground_url?: string;
};

export async function fetchSceneMeshes(): Promise<SceneMeshInfo[]> {
  const response = await getJson<ApiEnvelope<{ scenes: SceneMeshInfo[] }>>(
    "/api/scenes/3d",
  );
  return response.data?.scenes ?? [];
}

export type SceneBatchSummary = {
  name: string;
  frames: number;
  classes: string[];
  has_masks: boolean;
  map_ids: string[];
  output_dir: string;
  frames_written?: number;
};

export async function runSceneBatch(payload: {
  name?: string;
  jobs: Record<string, unknown>[];
  defaults?: Record<string, unknown>;
  include_masks?: boolean;
  output_dir?: string | null;
}): Promise<SceneBatchSummary> {
  const response = await postJson<ApiEnvelope<SceneBatchSummary>>(
    "/api/scene/batch",
    payload,
  );
  return response.data;
}

export async function listTrainingSets(): Promise<
  Array<{
    name: string;
    path: string;
    frames?: number;
    classes?: string[];
    has_masks?: boolean;
    map_ids?: string[];
  }>
> {
  const response = await getJson<
    ApiEnvelope<{ training_sets: Array<Record<string, unknown>> }>
  >("/api/scene/training_sets");
  return (response.data?.training_sets ?? []) as Array<{
    name: string;
    path: string;
    frames?: number;
    classes?: string[];
    has_masks?: boolean;
    map_ids?: string[];
  }>;
}

export type RedrawProvider = "minimax_api" | "comfyui";
export type RedrawStylePreset = "faithful" | "detailed" | "match_3d_light";

export type RedrawPresetMeta = {
  key: RedrawStylePreset;
  label: string;
  description: string;
};

export type RedrawConfig = {
  default_provider: RedrawProvider;
  providers: {
    minimax_api: {
      configured: boolean;
      base_url: string;
      models: string[];
      default_model: string;
    };
    comfyui: {
      configured: boolean;
      url: string;
    };
  };
  presets: RedrawPresetMeta[];
};

export type RedrawPingResult = {
  ok: boolean;
  url: string;
  comfyui_version: string | null;
  has_minimax_h3_image: boolean;
  has_minimax_h3_video?: boolean;
  gateway_state?: string | null;
  recovery_stage?: string | null;
  upstream_url?: string | null;
  message: string;
};

export type RedrawSubmitPayload = {
  provider: RedrawProvider;
  source_image: string;
  reference_image?: string | null;
  environment_image?: string | null;
  protection_mask?: string | null;
  protect_equipment?: boolean;
  scene_code?: string;
  scene_label?: string;
  style_preset: RedrawStylePreset;
  prompt?: string;
  model?: string;
  size?: string;
  api_key?: string;
  api_base_url?: string;
  comfyui_url?: string;
  steps?: number;
  megapixels?: number;
  seed?: number | null;
};

export type RedrawTaskStatus = {
  task_id: string;
  status: "running" | "completed" | "failed";
  provider: RedrawProvider;
  model?: string;
  style_preset?: RedrawStylePreset;
  scene_code?: string;
  scene_label?: string;
  created_at?: string;
  finished_at?: string;
  elapsed_sec?: number;
  output_size?: [number, number];
  source_size?: [number, number];
  reference_size?: [number, number];
  requested_size?: string;
  equipment_protected?: boolean;
  protection_coverage_pct?: number;
  revised_prompt?: string;
  effective_prompt?: string;
  error_message?: string;
  result_url?: string;
  model_url?: string;
  source_url?: string;
  reference_url?: string;
  mask_url?: string;
};

export type RedrawHistoryItem = RedrawTaskStatus;

export async function fetchRedrawConfig(): Promise<RedrawConfig> {
  const response = await getJson<ApiEnvelope<RedrawConfig>>("/api/scene/redraw/config");
  return response.data;
}

export async function pingRedrawComfyUI(url: string): Promise<RedrawPingResult> {
  const response = await postJson<ApiEnvelope<RedrawPingResult>>("/api/scene/redraw/ping", { url });
  return response.data;
}

export async function setRedrawGatewayUpstream(gatewayUrl: string, upstreamUrl: string): Promise<{
  ok: boolean;
  gateway_url: string;
  upstream_url: string;
}> {
  const response = await postJson<ApiEnvelope<{ ok: boolean; gateway_url: string; upstream_url: string }>>(
    "/api/scene/redraw/gateway/set-upstream",
    { gateway_url: gatewayUrl, upstream_url: upstreamUrl },
  );
  return response.data;
}

export async function submitSceneRedraw(payload: RedrawSubmitPayload): Promise<SceneExportResult> {
  const response = await postJson<ApiEnvelope<SceneExportResult>>("/api/scene/redraw", payload);
  return response.data;
}

export async function fetchSceneRedrawStatus(taskId: string): Promise<RedrawTaskStatus> {
  const response = await getJson<ApiEnvelope<RedrawTaskStatus>>(`/api/scene/redraw/${encodeURIComponent(taskId)}`);
  return response.data;
}

export async function listSceneRedraws(): Promise<RedrawHistoryItem[]> {
  const response = await getJson<ApiEnvelope<{ redraws: RedrawHistoryItem[] }>>("/api/scene/redraws");
  return response.data?.redraws ?? [];
}
