import { getJson, postJson } from "./http";
import type { SceneMeshInfo } from "./scene3d";

export type Pair = [number, number];
export type Control = { uv: Pair; xz: Pair; check?: boolean };
export type Pose = { heightM: number; pitchDeg: number; rollDeg?: number };
export type View = { source: string; fovY: number; focalRatio?: number; principal?: Pair; controls: Control[]; approxPose?: Pose; groundMask?: string };
export type Observation = { source: string; roof: Pair[]; base?: Pair };
export type Building = { id: string; heightM?: number; heightSource?: "estimated" | "measured"; ridgeRiseM?: number; observations: Observation[] };
export type GroundSettings = { marginPercent?: number; surface?: "local" | "fields"; reliefM?: number; roads?: { points: Pair[]; widthM: number; extendEnds?: boolean }[] };
export type GroundReport = { method: "local" | "imported" | "qwen-image-2.1"; size: Pair; heightRangeM: Pair; observed: number; occluded: number; outside: number; unresolved: number; groundTriangles: number; parentScene?: string };
export type OrbitPoint = { guide: Pair; image: Pair; world?: [number, number, number]; check?:boolean };
export type Supplement = { source:string; reference:string; angle:number; provenance:"qwen-image-2.1"|"image-api"|"imported-synthetic"; provider?:string; model?:string; registration?:"measured"|"target"; points?:OrbitPoint[] };
export type OrbitCamera = { position:number[]; target:number[]; up:number[]; size:number[]; principal:number[]; focalRatio:number; fov:number };
export type Alignment = { camera?:OrbitCamera; detailPercent?:number; orbitDeltaDeg?:number; newlyVisibleFaces?:number; newlyVisiblePixels?:number; checkPoints?:number; checkMedianPx?:number; targetMismatch?:boolean; status:"aligned"|"needs-points"|"needs-detail"; method:string; actualAngle:number|null; rmsPx:number|null; inliers:number; coveragePercent:number; message:string };
export type OrbitDirection = { angle:number; label:string; azimuth:string; suggestedStrength:number };
export type OrbitProvider = { id:string; label:string; ready:boolean; models:string[]; angleMode:"continuous"|"directions"; defaultAngle:number;
  directions?:OrbitDirection[]; supportsStrength?:boolean; qualities?:string[]; defaultQuality?:string; guidanceModes?:{value:string;label:string}[];
  defaultGuidance?:string; acceptsInstruction?:boolean; message?:string };
export type OrbitCatalog = { defaultProvider:string|null; providers:OrbitProvider[]; minAngle:number; maxAngle:number; defaultAngle:number; presets:{angle:number;label:string}[] };
export type OrbitJob = { taskId:string; status:"running"|"failed"|"completed"; angle:number; reference:string; quality?:Alignment; generation?:{ model?:string; strength?:number; guidance?:string }; spec?:Supplement; imageUrl?:string; guideUrl?:string; overlayUrl?:string; message?:string };
export type Plan = { version: 1; name: string; scaleBasis: "estimated" | "measured"; views: View[]; buildings: Building[]; ground?: GroundSettings; supplements?:Supplement[]; videoEvidence?:string };
export type ViewReport = { source: string; cameraMethod?: string; fitPoints: number; fitRmsPx: number | null; checkPoints: number; checkRmsPx: number | null; checkRmsM: number | null };
export type RoofReport = { id: string; heightM: number; heightMethod: string; views: number; minRayAngleDeg: number | null; roofRmsPx: number; areaM2: number };
export type SurfaceDetail = {
  videoSurfaces?: { buildings:number; adjustedRoofHeights:number; correctedFootprints:number; joinedEaves:number; roofTextures:number; wallTextures:number; inspectionCamera?:OrbitCamera; reason?:string };
  photoProjectors?: {pairedAreaPercent?: Record<string,number>};
  localMaterialReferences?: {
    source:string; angle:number; model?:string; walls:number; roofs:number; yardTexels:number; checkMedianPx?:number|null;
  }[];
  reviewedFacesByView: Record<string,number>;
  context: Record<string,{infillBuildings:number; trees:number; boundaryRuns:number}>;
  globalSurface?: { triangles:number; byView:Record<string,number>; closedAnchorBuildings:number; retainedBuildings:number };
};
export type Report = { scaleBasis: string; views: ViewReport[]; buildings: RoofReport[]; notes: string[]; supplements?:(Supplement & Alignment)[]; orbitCoverage?:{largestGapDeg:number; suggestedAnglesDeg?:number[]}; mesh?: { photoFaces: number; partialFaces?: number; inferredFaces: number; ground?: GroundReport & {syntheticTexels?:number}; surfaceDetail?:SurfaceDetail; supplements?:{buildingFaces:number;openingFrames:number;byView:Record<string,number>} } };
export const maxSupplementViews = 16;
export type BuildResult = { scene: SceneMeshInfo; report: Report };

type Envelope<T> = { data: T };
export const examplePlan = () => getJson<Plan>("/api/builder/example");
export const imageUrl = (id: string) => "/api/builder/sources/" + encodeURIComponent(id) + "/image";
export const checkPlan = async (plan: Plan) => (await postJson<Envelope<Report>>("/api/builder/validate", plan)).data;
export const buildPlan = async (plan: Plan) => (await postJson<Envelope<{ scene: SceneMeshInfo; report: Report }>>("/api/builder/build", plan)).data;
export const getTools = async () => (await getJson<Envelope<{ planner:boolean; orbitViews:boolean; orbitProvider:string|null }>>("/api/builder/tools")).data;
export const getOrbitProviders = async () => (await getJson<Envelope<OrbitCatalog>>("/api/builder/orbit-providers")).data;
type OrbitRequest = {plan:Plan; reference:string; angle:number; image?:string; points?:OrbitPoint[]; strength?:number; provider?:string;
  model?:string; guidance?:string; quality?:string; instruction?:string; registration?:"measured"|"target" };
export const createOrbit = async (request:OrbitRequest) => (await postJson<Envelope<OrbitJob>>("/api/builder/orbit-views",request)).data;
export const buildOrbit = async (plan:Plan,taskIds:string[],keepSources:string[],acceptMeasuredPose:boolean) =>
  (await postJson<Envelope<BuildResult & {plan:Plan}>>("/api/builder/orbit-build",{plan,taskIds,keepSources,acceptMeasuredPose})).data;
export const getOrbit = async (id:string) => (await getJson<Envelope<OrbitJob>>("/api/builder/orbit-jobs/"+encodeURIComponent(id))).data;
export async function applyGround(base: string, file: File): Promise<BuildResult> {
  const response = await fetch(base + "/apply", { method: "POST", body: file });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "地面补图导入失败");
  return payload.data;
}
export const proposePlan = async (plan: Plan, instruction: string) =>
  (await postJson<Envelope<{ plan: Plan }>>("/api/builder/propose", { plan, instruction })).data.plan;

export async function uploadPhoto(file: File): Promise<string> {
  const res = await fetch("/api/builder/sources", { method: "POST", body: file });
  const result = await res.json();
  if (!res.ok) throw new Error(result.detail || "照片上传失败");
  return result.data.source;
}

// In-progress local drafts may have fewer controls/corners than a buildable plan.
// Check the shape before binding imported data to the editing controls.
export function editablePlan(raw: unknown): raw is Plan {
  const p = raw as Plan | null;
  const pair = (v: unknown): v is Pair => Array.isArray(v) && v.length === 2 && v.every(Number.isFinite);
  const uv = (v: unknown) => pair(v) && v.every(n => n >= 0 && n <= 1);
  const id = (v: unknown) => typeof v === "string" && /^[A-Za-z0-9_-]{1,80}$/.test(v);
  if (!p || p.version !== 1 || typeof p.name !== "string" || !["estimated", "measured"].includes(p.scaleBasis) ||
      !Array.isArray(p.views) || !p.views.length || p.views.length > 6 || !Array.isArray(p.buildings) || p.buildings.length > 150) return false;
  if (!p.views.every(v => v && id(v.source) && Number.isFinite(v.fovY) && Array.isArray(v.controls) &&
      v.controls.every(c => c && uv(c.uv) && pair(c.xz)))) return false;
  const sources = new Set(p.views.map(v => v.source));
  if (p.supplements !== undefined && (!Array.isArray(p.supplements) || p.supplements.length>maxSupplementViews || !p.supplements.every(s=>
    s && id(s.source) && sources.has(s.reference) && Number.isFinite(s.angle) && Math.abs(s.angle)>=10 && Math.abs(s.angle)<=180 &&
    ["qwen-image-2.1","image-api","imported-synthetic"].includes(s.provenance) && (!s.points || (Array.isArray(s.points) &&
      s.points.length>=6 && s.points.length<=64 && s.points.every(pt=>pt && uv(pt.guide) && uv(pt.image))))))) return false;
  return sources.size === p.views.length && new Set(p.buildings.map(b => b?.id)).size === p.buildings.length &&
    p.buildings.every(b => b && id(b.id) && Array.isArray(b.observations) && b.observations.every(o =>
      o && sources.has(o.source) && Array.isArray(o.roof) && o.roof.every(uv) && (o.base === undefined || uv(o.base))));
}
