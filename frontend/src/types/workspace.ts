import type { Asset3D, SceneMeshInfo } from "../api/scene3d";

export type WorkspaceRoute = "blender" | "calibrated" | "3dgs" | "sandbox";
export type CalibratedSubView = "2d" | "3d";
export type InspectorTab = "scene" | "ground" | "orbit" | "transform" | "light" | "export";
export type TransformMode = "translate" | "rotate" | "scale";
export type LightingMode = "splat_hybrid" | "match_object" | "relight_all";

export type LoadedSceneMeta = {
  buildingCount?: number;
  referenceImageUrl?: string;
  completionNote?: string;
};

export type ScenePropDef = {
  code: string;
  label: string;
  size: [number, number, number];
};

export const SCENE_PROPS: readonly ScenePropDef[] = [
  { code: "PROP_BLOCK", label: "建筑体块", size: [8, 6, 8] },
  { code: "PROP_WALL", label: "围墙", size: [10, 2.5, 0.6] },
  { code: "PROP_CRATE", label: "箱体", size: [2, 2, 2] },
] as const;

export const DEMO_ASSETS: Asset3D[] = [
  {
    code: "DEMO_VEHICLE",
    label: "演示载具",
    category: "演示",
    has_model: false,
    url: null,
    dimensions: { length: 9.0, width: 3.6, height: 2.3 },
    origin: [0, 0, 0],
    up_axis: "y",
    yaw_offset: 0,
    default_scale: 1,
    thumbnail: null,
  },
  {
    code: "DEMO_LIGHT",
    label: "轻型车辆",
    category: "演示",
    has_model: false,
    url: null,
    dimensions: { length: 5.0, width: 2.2, height: 1.8 },
    origin: [0, 0, 0],
    up_axis: "y",
    yaw_offset: 0,
    default_scale: 1,
    thumbnail: null,
  },
  {
    code: "DEMO_BUILDING",
    label: "建筑体块",
    category: "场景",
    has_model: false,
    url: null,
    dimensions: { length: 8.0, width: 8.0, height: 6.0 },
    origin: [0, 0, 0],
    up_axis: "y",
    yaw_offset: 0,
    default_scale: 1,
    thumbnail: null,
  },
];

export const ROUTE_META: Record<
  WorkspaceRoute,
  {
    title: string;
    shortLabel: string;
    badge: string;
    description: string;
    isExperimental: boolean;
  }
> = {
  blender: {
    title: "Blender 精模 PBR（主工作台）",
    shortLabel: "Blender 精模 · 2K PBR",
    badge: "BLENDER 2K PBR",
    description: "263 栋闭合水密建筑 + 2K Poly Haven CC0 PBR 贴图 + 实时全场景光影",
    isExperimental: false,
  },
  calibrated: {
    title: "照片标定重建",
    shortLabel: "照片标定重建",
    badge: "实验 · 照片标定",
    description: "单/多图 2D 描点标定、地面补全、补充视角与标定网格预览",
    isExperimental: true,
  },
  "3dgs": {
    title: "3DGS 高斯混合",
    shortLabel: "3DGS 高斯混合",
    badge: "实验 · 3DGS",
    description: "高斯泼溅点云背景与网格阴影代理实时合成",
    isExperimental: true,
  },
  sandbox: {
    title: "平地自由沙盘",
    shortLabel: "平地自由沙盘",
    badge: "实验 · 平地沙盘",
    description: "空白贴图平地 + 基础体块 / 围墙 / 箱体自由拼装",
    isExperimental: true,
  },
};

export function classifySceneMeshes(scenes: SceneMeshInfo[]): {
  blender: SceneMeshInfo[];
  calibrated: SceneMeshInfo[];
  splat: SceneMeshInfo[];
} {
  return {
    blender: scenes.filter(
      (s) =>
        s.code === "VILLAGE_BLENDER" ||
        (s.kind === "photo_completed_village" && s.code !== "VILLAGE_COMPLETE" && !s.plan_url),
    ),
    calibrated: scenes.filter(
      (s) =>
        Boolean(s.plan_url) ||
        s.kind === "calibrated_building_scene" ||
        s.code === "VILLAGE_COMPLETE",
    ),
    splat: scenes.filter((s) => Boolean(s.splat_url)),
  };
}

export function resolveInitialRoute(search: string): {
  route: WorkspaceRoute;
  calibratedSubView: CalibratedSubView;
} {
  const params = new URLSearchParams(search);
  const rawRoute = params.get("route");
  const panel = params.get("panel");
  const scene = params.get("scene") || "";
  const view3d = params.get("view3d") === "1";

  if (
    rawRoute === "blender" ||
    rawRoute === "calibrated" ||
    rawRoute === "3dgs" ||
    rawRoute === "sandbox"
  ) {
    if (rawRoute === "calibrated") {
      const open3d =
        view3d || Boolean(scene) || panel === "ground" || panel === "orbit";
      return { route: "calibrated", calibratedSubView: open3d ? "3d" : "2d" };
    }
    return { route: rawRoute, calibratedSubView: "2d" };
  }

  if (
    (scene && scene !== "VILLAGE_BLENDER") ||
    panel === "ground" ||
    panel === "orbit"
  ) {
    return { route: "calibrated", calibratedSubView: "3d" };
  }

  return { route: "blender", calibratedSubView: "2d" };
}
