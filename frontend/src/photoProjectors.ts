import * as THREE from "three";
import type { GLTFParser } from "three/examples/jsm/loaders/GLTFLoader.js";

type CameraRecord = {
  source: string; texture: number; depthTexture: number; depthScale: number; depthErrorRange?: number;
  size: number[]; eye: number[]; rotation: number[]; intrinsics: number[]; paired?: boolean;
};
type Projector = {
  color: THREE.Texture; depth: THREE.Texture; eye: THREE.Vector3; rotation: THREE.Matrix3;
  intrinsics: THREE.Vector4; size: THREE.Vector2; scale: number; depthErrorRange: number; score: number; paired: boolean;
};

const vectors = `
varying vec3 vPhotoWorld;
varying vec3 vPhotoNormal;
`;
const declarations = (i: number) => `
uniform sampler2D uImage${i};
uniform sampler2D uDepth${i};
uniform vec3 uEye${i};
uniform mat3 uRotation${i};
uniform vec4 uK${i};
uniform vec2 uSize${i};
uniform float uDepthScale${i};
uniform float uDepthErrorRange${i};
uniform float uRank${i};
`;
const project = `
struct PhotoSample { vec3 color; float weight; float opacity; float alignment; };
PhotoSample samplePhoto(sampler2D photo, sampler2D depth, vec3 eye, mat3 rotation,
  vec4 K, vec2 imageSize, float depthScale, float depthErrorRange, float rank, vec3 viewRay, vec3 normal) {
  PhotoSample s = PhotoSample(vec3(0.), 0., 0., 0.);
  vec3 local = rotation * (vPhotoWorld - eye);
  if (rank <= 0. || local.z <= 0.001) return s;
  vec2 px = K.xy * local.xy / local.z + K.zw;
  vec2 edge = min(px, imageSize - px);
  if (min(edge.x, edge.y) <= 0.) return s;
  vec2 uv = px / imageSize;
  vec4 packedDepth = texture2D(depth, uv);
  vec2 rg = floor(packedDepth.rg * 255. + 0.5);
  float quantized = dot(rg, vec2(256., 1.));
  float tol = max(.24, .006 * local.z);
  float depthVisibility = 1. - smoothstep(tol, 2. * tol, abs(local.z - quantized * depthScale));
  if (quantized < .5 || depthVisibility <= 0.) return s;
  vec3 sourceRay = normalize(eye - vPhotoWorld);
  // A depth discrepancy only creates image slip when the observer moves off the source ray.
  float baseline = length(cross(cameraPosition - eye, sourceRay));
  float slipPx = max(K.x, K.y) * (baseline / local.z) * packedDepth.b * depthErrorRange;
  float correspondence = 1. - smoothstep(3., 18., slipPx);
  float alignment = dot(viewRay, sourceRay);
  float angular = smoothstep(.32, .91, alignment);
  float facing = smoothstep(-.04, .14, dot(normal, sourceRay));
  facing = mix(facing, 1., smoothstep(.94, .995, alignment));
  s.opacity = rank * angular * facing * depthVisibility * packedDepth.a * correspondence * smoothstep(0., 1.5, min(edge.x, edge.y));
  s.weight = s.opacity * pow(max(0., alignment), 12.);
  s.alignment = alignment;
  s.color = texture2D(photo, uv).rgb;
  return s;
}`;
const applyColor = `
float photoCoverage = 0.;
if (uPhotoEnabled > 0.) {
  vec3 viewRay = normalize(cameraPosition - vPhotoWorld);
  vec3 normal = normalize(vPhotoNormal);
  if (dot(normal, viewRay) < 0.) normal = -normal;
  PhotoSample a = samplePhoto(uImage0, uDepth0, uEye0, uRotation0, uK0,
    uSize0, uDepthScale0, uDepthErrorRange0, uRank0, viewRay, normal);
  PhotoSample b = samplePhoto(uImage1, uDepth1, uEye1, uRotation1, uK1,
    uSize1, uDepthScale1, uDepthErrorRange1, uRank1, viewRay, normal);
  // At a registered eye prefer its own colour even if another eye also sees the surface.
  a.weight *= 1. - b.opacity * smoothstep(.985, .9999, b.alignment) * smoothstep(.002, .04, b.alignment-a.alignment);
  b.weight *= 1. - a.opacity * smoothstep(.985, .9999, a.alignment) * smoothstep(.002, .04, a.alignment-b.alignment);
  float total = a.weight + b.weight;
  if (total > .00001) {
    vec3 color = (a.color*a.weight + b.color*b.weight) / total;
    photoCoverage = max(a.opacity, b.opacity) * uPhotoEnabled;
    diffuseColor.rgb = mix(diffuseColor.rgb, color, photoCoverage);
  }
  PhotoSample c = samplePhoto(uImage2, uDepth2, uEye2, uRotation2, uK2,
    uSize2, uDepthScale2, uDepthErrorRange2, uRank2, viewRay, normal);
  PhotoSample d = samplePhoto(uImage3, uDepth3, uEye3, uRotation3, uK3,
    uSize3, uDepthScale3, uDepthErrorRange3, uRank3, viewRay, normal);
  c.weight *= 1. - d.opacity * smoothstep(.985,.9999,d.alignment) * smoothstep(.002,.04,d.alignment-c.alignment);
  d.weight *= 1. - c.opacity * smoothstep(.985,.9999,c.alignment) * smoothstep(.002,.04,c.alignment-d.alignment);
  float verified = c.weight+d.weight;
  if (verified > .00001) {
    float pairedOpacity = max(c.opacity,d.opacity)*uPhotoEnabled;
    diffuseColor.rgb = mix(diffuseColor.rgb,(c.color*c.weight+d.color*d.weight)/verified,pairedOpacity);
    photoCoverage = 1. - (1.-photoCoverage)*(1.-pairedOpacity);
  }
}`;

function validCamera(record: unknown): record is CameraRecord {
  if (!record || typeof record !== "object") return false;
  const r = record as CameraRecord;
  const finite = (a: number[], n: number) => Array.isArray(a) && a.length === n && a.every(Number.isFinite);
  return Number.isInteger(r.texture) && r.texture >= 0 && Number.isInteger(r.depthTexture) && r.depthTexture >= 0
    && Number.isFinite(r.depthScale) && r.depthScale > 0 && finite(r.eye, 3) && finite(r.rotation, 9)
    && finite(r.size, 2) && r.size.every(n => n > 0) && finite(r.intrinsics, 4) && r.intrinsics[0] > 0 && r.intrinsics[1] > 0;
}

async function readCamera(parser: GLTFParser, record: CameraRecord): Promise<Projector> {
  const [photo, visibility] = await Promise.all([record.texture, record.depthTexture].map(i => parser.getDependency("texture", i)));
  if (!photo?.isTexture || !visibility?.isTexture) throw new Error("缺少投影照片或深度纹理");
  const color: THREE.Texture = photo.clone(), depth: THREE.Texture = visibility.clone();
  color.colorSpace = THREE.SRGBColorSpace;
  depth.colorSpace = THREE.NoColorSpace;
  depth.minFilter = depth.magFilter = THREE.NearestFilter;
  depth.generateMipmaps = false;
  for (const texture of [color, depth]) {
    texture.flipY = false;
    texture.wrapS = texture.wrapT = THREE.ClampToEdgeWrapping;
    texture.needsUpdate = true;
  }
  const rotation = new THREE.Matrix3().fromArray(record.rotation).transpose();
  return { color, depth, eye: new THREE.Vector3().fromArray(record.eye), rotation,
    intrinsics: new THREE.Vector4().fromArray(record.intrinsics), size: new THREE.Vector2().fromArray(record.size),
    scale: record.depthScale, depthErrorRange: Number.isFinite(record.depthErrorRange) ? Math.max(0, record.depthErrorRange!) : 0,
    score: 0, paired: record.paired === true };
}

export async function readPhotoProjectors(parser: GLTFParser, extras: unknown) {
  const metadata = extras as {version?: number; cameras?: unknown[]};
  if (metadata?.version !== 1 || !Array.isArray(metadata.cameras)) return null;
  const loaded = await Promise.allSettled(metadata.cameras.filter(validCamera).map(r => readCamera(parser, r)));
  const cameras = loaded.flatMap(result => result.status === "fulfilled" ? [result.value] : []);
  if (!cameras.length) return null;
  const enabled = { value: 1 };
  const references = cameras.filter(p => !p.paired), paired = cameras.filter(p => p.paired);
  const uniforms: Record<string, THREE.IUniform> = { uPhotoEnabled: enabled };
  const center = new THREE.Vector3(), view = new THREE.Vector3(), direction = new THREE.Vector3();
  const previousEye = new THREE.Vector3(Infinity, Infinity, Infinity);
  const setSlot = (i: number, p: Projector, rank: number) => {
    const values = { uImage: p.color, uDepth: p.depth, uEye: p.eye, uRotation: p.rotation,
      uK: p.intrinsics, uSize: p.size, uDepthScale: p.scale, uDepthErrorRange: p.depthErrorRange, uRank: rank };
    for (const [name, value] of Object.entries(values)) {
      const key = name+i;
      if (uniforms[key]) uniforms[key].value = value;
      else uniforms[key] = { value };
    }
  };
  setSlot(0, cameras[0], 1);
  setSlot(1, cameras[0], 0);
  setSlot(2, cameras[0], 0);
  setSlot(3, cameras[0], 0);
  return {
    center,
    update(eye: THREE.Vector3) {
      if (eye.equals(previousEye)) return;
      previousEye.copy(eye);
      view.subVectors(eye, center).normalize();
      for (const p of cameras) p.score = direction.subVectors(p.eye, center).normalize().dot(view);
      // Sparse paired observations supplement the established view only at
      // verified pixels. They never evict a useful reference by camera rank.
      references.sort((a,b) => b.score-a.score);
      paired.sort((a,b) => b.score-a.score);
      for (const [bank,offset] of [[references,0],[paired,2]] as const) {
        for (let i=0; i<2; i++) {
          const p=bank[i];
          let rank=p && bank.length>2 ? THREE.MathUtils.smoothstep(p.score-bank[2].score,0,offset ? .012 : .10) : p ? 1 : 0;
          if (p && offset) {
            rank *= THREE.MathUtils.smoothstep(p.score,.86,.985);
            const closeReference = references.length ? THREE.MathUtils.smoothstep(references[0].score,.996,1.) : 0;
            rank *= 1-closeReference*(1-THREE.MathUtils.smoothstep(p.score,.995,1.));
          }
          setSlot(offset+i,p || cameras[0],rank);
        }
      }
    },
    bind(material: THREE.MeshBasicMaterial) {
      material.userData.photoProjectors = { enabled };
      material.customProgramCacheKey = () => "registered-photo-on-world-v2";
      material.onBeforeCompile = shader => {
        Object.assign(shader.uniforms, uniforms);
        shader.vertexShader = shader.vertexShader.replace("#include <common>", "#include <common>\n"+vectors)
          .replace("#include <project_vertex>", "#include <project_vertex>\nvPhotoWorld = (modelMatrix * vec4(transformed, 1.)).xyz;\nvPhotoNormal = mat3(modelMatrix) * normal;");
        shader.fragmentShader = shader.fragmentShader.replace("#include <common>",
          "#include <common>\n"+vectors+"\nuniform float uPhotoEnabled;\n"+[0,1,2,3].map(declarations).join("")+project)
          .replace("#include <color_fragment>", "#include <color_fragment>\n"+applyColor)
          // The source photograph already contains atmospheric haze at its own distance.
          .replace("#include <fog_fragment>", "vec3 cameraColor = gl_FragColor.rgb;\n#include <fog_fragment>\ngl_FragColor.rgb = mix(gl_FragColor.rgb, cameraColor, photoCoverage);");
      };
    },
    dispose() { for (const p of cameras) { p.color.dispose(); p.depth.dispose(); } },
  };
}
