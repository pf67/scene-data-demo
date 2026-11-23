import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

/** Render visible equipment coverage with the same camera and occluding scene geometry. */
export function captureProtectedScene(
  renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera,
  equipment: THREE.Object3D[], helpers: Array<THREE.Object3D | null>,
): { source: string; mask: string; environment: string } {
  const visibility = helpers.map(object => ({ object, visible: object?.visible }));
  const materials: Array<{ mesh: THREE.Mesh; material: THREE.Material | THREE.Material[] }> = [];
  const splatVisibility: Array<{ object: THREE.Object3D; visible: boolean }> = [];
  const temporary: THREE.Material[] = [];
  const background = scene.background;
  const fog = scene.fog;
  const toneMapping = renderer.toneMapping;
  const shadows = renderer.shadowMap.enabled;
  const shadowAutoUpdate = renderer.shadowMap.autoUpdate;
  const shadowNeedsUpdate = renderer.shadowMap.needsUpdate;
  const equipmentVisibility = equipment.map(object => ({ object, visible: object.visible }));
  const clearColor = renderer.getClearColor(new THREE.Color());
  const clearAlpha = renderer.getClearAlpha();
  const roots = new Set(equipment);
  try {
    for (const { object } of visibility) if (object) object.visible = false;
    renderer.render(scene, camera);
    const source = renderer.domElement.toDataURL("image/png");
    // Render clean environment while retaining the existing equipment shadow map.
    renderer.shadowMap.autoUpdate = false;
    renderer.shadowMap.needsUpdate = false;
    for (const { object } of equipmentVisibility) object.visible = false;
    renderer.render(scene, camera);
    const environment = renderer.domElement.toDataURL("image/png");
    for (const { object, visible } of equipmentVisibility) object.visible = visible;
    scene.traverse(object => {
      if (object instanceof SparkRenderer || object instanceof SplatMesh || object.name.startsWith("splat_")) {
        splatVisibility.push({ object, visible: object.visible });
        object.visible = false;
        return;
      }
      if (!(object instanceof THREE.Mesh)) return;
      let node: THREE.Object3D | null = object;
      let protectedPixel = false;
      while (node) {
        if (roots.has(node)) protectedPixel = true;
        node = node.parent;
      }
      materials.push({ mesh: object, material: object.material });
      const convert = (original: THREE.Material) => {
        const textured = original as THREE.MeshStandardMaterial;
        const shadowCatcher = original instanceof THREE.ShadowMaterial;
        const material = new THREE.MeshBasicMaterial({
          color: protectedPixel ? 0xffffff : 0x000000,
          map: textured.map ?? null, alphaMap: textured.alphaMap ?? null,
          alphaTest: shadowCatcher ? 0 : original.alphaTest,
          opacity: shadowCatcher ? 1 : original.opacity,
          transparent: shadowCatcher ? false : original.transparent,
          side: original.side,
          depthTest: original.depthTest,
          depthWrite: shadowCatcher ? true : original.depthWrite,
          toneMapped: false,
        });
        // Keep texture alpha cutouts while replacing only its color.
        const value = protectedPixel ? "1.0" : "0.0";
        material.onBeforeCompile = shader => {
          shader.fragmentShader = shader.fragmentShader.replace(
            "#include <map_fragment>", `#include <map_fragment>\ndiffuseColor.rgb = vec3(${value});`,
          );
        };
        material.customProgramCacheKey = () => `protection-${value}`;
        temporary.push(material);
        return material;
      };
      object.material = Array.isArray(object.material) ? object.material.map(convert) : convert(object.material);
    });
    scene.background = new THREE.Color(0x000000);
    scene.fog = null;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.shadowMap.enabled = false;
    renderer.setClearColor(0x000000, 1);
    renderer.render(scene, camera);
    const mask = renderer.domElement.toDataURL("image/png");
    return { source, mask, environment };
  } finally {
    for (const item of materials) item.mesh.material = item.material;
    for (const { object, visible } of splatVisibility) object.visible = visible;
    for (const material of temporary) material.dispose();
    scene.background = background;
    scene.fog = fog;
    renderer.toneMapping = toneMapping;
    renderer.shadowMap.enabled = shadows;
    renderer.shadowMap.autoUpdate = shadowAutoUpdate;
    renderer.shadowMap.needsUpdate = shadowNeedsUpdate;
    for (const { object, visible } of equipmentVisibility) object.visible = visible;
    renderer.setClearColor(clearColor, clearAlpha);
    for (const { object, visible } of visibility) if (object) object.visible = visible ?? true;
    renderer.render(scene, camera);
  }
}
