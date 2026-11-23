"""Export reviewed Blender village scene to VILLAGE_BLENDER.glb, metadata JSON, topology report, and packed .blend."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools/blender_bridge")]

from blender_bridge_addon import blender_to_three, run_audit  # noqa: E402


def export_village(
    *,
    spec_path: Path,
    blend_path: Path,
    out_glb: Path,
    label: str = "村庄 · Blender 精模与多视角视频烘焙",
) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    scene = bpy.context.scene

    # Exclude guide-only collections/objects from export
    guide_col = bpy.data.collections.get("COL_Ref_Guide")
    guide_names = {ob.name for ob in guide_col.objects} if guide_col else set()

    audit = run_audit(
        {
            "expected_closed_buildings": len(spec["buildings"]),
            "road_polygon_xz": spec["road_polygon_xz"],
            "road_apron_polygon_xz": spec.get("road_apron_polygon_xz"),
            "strict_layout_audit": True,
        }
    )
    if spec.get("layout_audit") is not None:
        audit["layout_audit"] = spec["layout_audit"]
    if audit["verdict"] != "pass":
        raise SystemExit(f"Export blocked by failed audit gate ({audit['verdict']}): {audit['failures'][:5]}")

    mesh_objects = [
        ob
        for ob in scene.objects
        if ob.type == "MESH" and not ob.hide_render and ob.name not in guide_names and not ob.get("guide_only", False)
    ]
    if not mesh_objects:
        raise SystemExit("No exportable mesh geometry found")

    # Set exact Three.js camera extras from CAM_ref or spec
    src_name = Path(spec.get("source_image", "village_road_near.png")).name
    scene["preserve_scale"] = True
    scene["scene_kind"] = "photo_completed_village"
    scene["reference_image_url"] = (
        "/api/builder/sources/village/image" if out_glb.stem == "VILLAGE_BLENDER" else f"/api/scenes/3d/{out_glb.stem}/source"
    )
    cam = bpy.data.objects.get("CAM_ref") or scene.camera
    if cam is not None:
        eye = cam.matrix_world.translation
        forward = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
        scene["camera_position"] = spec["camera"]["position"]
        scene["camera_target"] = spec["camera"]["target"]
        scene["camera_fov"] = float(spec["camera"]["fov"])
    else:
        scene["camera_position"] = spec["camera"]["position"]
        scene["camera_target"] = spec["camera"]["target"]
        scene["camera_fov"] = float(spec["camera"]["fov"])

    scene["building_count"] = int(audit["closed_buildings"])
    scene["source_image"] = src_name
    scene["completion_note"] = (
        f"{int(audit['closed_buildings'])} closed watertight rural buildings with real photo front facades, "
        "CC0 Poly Haven PBR atlases, dedicated tiled asphalt road, alpha-cutout 3D foliage, compound walls/gates, and vehicles."
    )

    bpy.ops.object.select_all(action="DESELECT")
    for ob in mesh_objects:
        ob.select_set(True)

    out_glb.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(out_glb),
        export_format="GLB",
        use_selection=True,
        export_extras=True,
        export_animations=False,
        export_image_format="AUTO",
        export_jpeg_quality=88,
        export_materials="EXPORT",
    )

    default_notes = (
        f"Blender 4.5 写实村落（{int(audit['closed_buildings'])} 栋闭合建筑 + 高清摄影立面与 CC0 PBR 贴图 + "
        "独立沥青道路/围墙门楼/透贴 3D 树木/车辆）。"
    )
    meta = {
        "code": out_glb.stem,
        "label": label,
        "kind": "photo_completed_village",
        "map_id": str(spec.get("map_id") or ("VILLAGE_CN" if out_glb.stem == "VILLAGE_BLENDER" else out_glb.stem)),
        "building_count": int(audit["closed_buildings"]),
        "preserve_scale": True,
        "source_image": src_name,
        "camera": spec["camera"],
        "notes": str(spec.get("notes") or default_notes),
    }
    out_glb.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    out_glb.with_suffix(".topology.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    size_mb = round(out_glb.stat().st_size / (1024 * 1024), 2)
    res = {"glb": str(out_glb), "size_mb": size_mb, "meshes": len(mesh_objects), "closed_buildings": audit["closed_buildings"]}
    print("EXPORT_VILLAGE_OK", json.dumps(res))
    return res


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, default=ROOT / "artifacts/blender_village/village_spec.json")
    p.add_argument("--blend", type=Path, default=ROOT / "artifacts/blender_village/village_master.blend")
    p.add_argument("--output", type=Path, default=ROOT / "assets/scene_meshes/VILLAGE_BLENDER.glb")
    p.add_argument("--label", default="村庄 · Blender 精模与多视角视频烘焙")
    return p.parse_args(argv)


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    args = parse_args(argv)
    export_village(
        spec_path=args.spec.resolve(),
        blend_path=args.blend.resolve(),
        out_glb=args.output.resolve(),
        label=args.label,
    )
