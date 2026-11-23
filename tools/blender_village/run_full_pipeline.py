"""End-to-end 4-layer Blender village pipeline orchestrator (PBR -> Dense Fabric -> Orbit -> H3 -> Re-Bake -> GLB)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "tools/blender_bridge"),
    str(ROOT / "tools/blender_village"),
    str(ROOT / "tools/scene_pipeline"),
]

from bridge_client import run_blender_script  # noqa: E402
from fetch_pbr_assets import fetch_and_build_pbr_assets  # noqa: E402
from prepare_evidence import prepare_evidence  # noqa: E402
from reconstruct_bev_dsm import run_bev_dsm_pipeline  # noqa: E402
from render_guide_orbit import render_guide_orbit  # noqa: E402
from bake_multiview import bake_atlases  # noqa: E402
from visual_qc import run_visual_qc  # noqa: E402
from colab_h3_setup import write_colab_notebook  # noqa: E402
import minimax_h3_orbit as h3  # noqa: E402
import train_3dgs  # noqa: E402


def run_full_pipeline(
    *,
    out_dir: Path,
    colmap_dir: Path,
    images_dir: Path,
    out_glb: Path,
    comfy_url: str | None = None,
    h3_videos: list[Path] | None = None,
    bev_dsm_dir: Path | None = None,
    run_bev_dsm: bool = False,
    offline: bool = False,
    skip_orbit_render: bool = False,
) -> dict[str, Any]:
    """Run all 4 photorealistic village pipeline layers end-to-end and emit pipeline_report.json."""
    t_start = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    blend_path = out_dir / "village_master.blend"
    spec_path = out_dir / "village_spec.json"
    uv_map_path = out_dir / "uv_atlas_map.json"
    probes_path = out_dir / "bake_probes.npz"
    qc_dir = out_dir / "qc"
    orbit_dir = out_dir / "orbit_guide"
    report: dict[str, Any] = {"stages": {}, "h3_status": "pending"}

    def record_stage(name: str, t0: float, detail: Any) -> None:
        report["stages"][name] = {"elapsed_sec": round(time.monotonic() - t0, 2), "detail": detail}

    # Stage 1: Fetch/build Poly Haven CC0 1K PBR & foliage assets
    t0 = time.monotonic()
    pbr = fetch_and_build_pbr_assets(out_dir, offline=offline)
    record_stage("1_fetch_pbr_assets", t0, {"swatches": len(pbr["swatches"]), "foliage": len(pbr["foliage"])})

    # Optional Route B: BEV DSM dense multi-view reconstruction on local GPU
    h3_sparse_existing = out_dir / "h3_colmap/colmap/sparse/0"
    h3_imgs_existing = out_dir / "h3_colmap/colmap/images"
    has_existing_h3 = (h3_sparse_existing / "images.txt").is_file()
    if run_bev_dsm:
        t0 = time.monotonic()
        bev_res = run_bev_dsm_pipeline(
            colmap_dir=colmap_dir,
            images_dir=images_dir,
            extra_colmap_dir=h3_sparse_existing if has_existing_h3 else None,
            extra_images_dir=h3_imgs_existing if has_existing_h3 else None,
            out_dir=out_dir,
        )
        record_stage("1b_reconstruct_bev_dsm", t0, {k: v for k, v in bev_res.items() if k != "frame_stats"})

    # Stage 2: Prepare regularized dense village_spec.json (reusing existing H3 & BEV DSM when available)
    t0 = time.monotonic()
    spec = prepare_evidence(
        colmap_dir=colmap_dir,
        images_dir=images_dir,
        out_dir=out_dir,
        extra_colmap_dir=h3_sparse_existing if has_existing_h3 else None,
        extra_images_dir=h3_imgs_existing if has_existing_h3 else None,
        bev_dsm_dir=bev_dsm_dir,
    )
    record_stage(
        "2_prepare_evidence",
        t0,
        {
            "buildings": len(spec["buildings"]),
            "courtyard_walls": len(spec["courtyard_walls"]),
            "trees": len(spec["trees"]),
            "layout_audit": spec.get("layout_audit"),
        },
    )

    # Stage 3: Build dense watertight village in Blender 4.5 + BVH photo-facade gating
    t0 = time.monotonic()
    build_proc = run_blender_script(
        ROOT / "tools/blender_village/build_village.py",
        ["--spec", str(spec_path), "--blend", str(blend_path)],
    )
    record_stage("3_build_village", t0, build_proc.stdout.strip().splitlines()[-1] if build_proc.stdout.strip() else "ok")

    # Stage 4: BVH visibility & 16-ray hemisphere AO probes
    t0 = time.monotonic()
    probe_proc = run_blender_script(
        ROOT / "tools/blender_village/export_bake_buffers.py",
        ["--spec", str(spec_path), "--uv-map", str(uv_map_path), "--out", str(probes_path)],
        blend_path=blend_path,
    )
    record_stage("4_export_bake_buffers", t0, probe_proc.stdout.strip().splitlines()[-1] if probe_proc.stdout.strip() else "ok")

    # Stage 5: Initial multi-view dominant projection + CC0 PBR weathering bake + Blender material link
    t0 = time.monotonic()
    bake_res = bake_atlases(
        spec_path=spec_path,
        uv_map_path=uv_map_path,
        probes_path=probes_path,
        out_dir=out_dir,
        blend_path=blend_path,
    )
    record_stage("5_bake_multiview", t0, bake_res)

    # Stage 6: Render TEXTURED 360-degree Blender guide turntable + 4 quadrant references for MiniMax-H3
    t0 = time.monotonic()
    if skip_orbit_render and (orbit_dir / "orbit_guide.mp4").is_file():
        orbit_res = {"skipped": True, "mp4": str(orbit_dir / "orbit_guide.mp4")}
    else:
        orbit_res = render_guide_orbit(spec_path=spec_path, blend_path=blend_path, out_dir=orbit_dir)
    record_stage("6_render_guide_orbit", t0, orbit_res)

    # Stage 7: MiniMax-H3 360° Bridge (live ComfyUI or local H3 video SfM, or dry-run validation + Colab .ipynb)
    t0 = time.monotonic()
    ipynb_path = write_colab_notebook(out_dir / "MiniMax_H3_Village_360_Colab.ipynb")
    url = (comfy_url or os.environ.get("MINIMAX_H3_COMFY_URL", "")).strip()
    video_list: list[Path] = list(h3_videos or [])

    if url:
        health = h3.check_comfy_health(url)
        if health.get("ok"):
            h3_args = h3.parse_args(
                [
                    "--comfy-url", url,
                    "--first-frame", spec["ref_clean_image"],
                    "--last-frame", spec["ref_clean_image"],
                    "--mode", "loop360",
                    "--width", "1152",
                    "--height", "768",
                    "--length", "175",
                    "--no-clean-watermark",
                    "--out-dir", str(out_dir / "h3_videos"),
                ]
            )
            h3_res = h3.run_orbit_generation(h3_args)
            video_list.append(Path(h3_res["videoPath"]))
            report["h3_status"] = "generated_via_comfyui"
        else:
            report["h3_status"] = f"comfy_unreachable: {health.get('error', 'unknown')}"
            print(f"[WARN] Live ComfyUI H3 unreachable ({url}): {health.get('error')}. Continuing locally.")

    if video_list:
        h3_work = out_dir / "h3_colmap"
        sfm_report = train_3dgs.run_joint_sfm(
            ROOT / "builder/example.json",
            Path(spec["ref_clean_image"]),
            ROOT / "assets/scene_meshes/VILLAGE_CALIBRATED.glb",
            video_list,
            h3_work / "colmap",
        )
        spec = prepare_evidence(
            colmap_dir=colmap_dir,
            images_dir=images_dir,
            out_dir=out_dir,
            extra_colmap_dir=h3_work / "colmap/sparse/0",
            extra_images_dir=h3_work / "colmap/images",
            bev_dsm_dir=bev_dsm_dir,
        )
        run_blender_script(
            ROOT / "tools/blender_village/export_bake_buffers.py",
            ["--spec", str(spec_path), "--uv-map", str(uv_map_path), "--out", str(probes_path)],
            blend_path=blend_path,
        )
        bake_res = bake_atlases(
            spec_path=spec_path,
            uv_map_path=uv_map_path,
            probes_path=probes_path,
            out_dir=out_dir,
            blend_path=blend_path,
        )
        report["h3_status"] = "aligned_and_merged"
        record_stage("7_minimax_h3_bridge", t0, {"status": report["h3_status"], "sfm": sfm_report, "rebake": bake_res})
    else:
        dry_args = h3.parse_args(["--first-frame", spec["ref_clean_image"], "--mode", "loop360", "--dry-run"])
        dry_res = h3.run_orbit_generation(dry_args)
        if report["h3_status"] == "pending":
            report["h3_status"] = "ready_for_colab_a100"
        record_stage(
            "7_minimax_h3_bridge",
            t0,
            {"status": report["h3_status"], "dryRunNodes": len(dry_res["workflow"]), "colabNotebook": str(ipynb_path)},
        )

    # Stage 8: Audit gate & 8-angle EEVEE Next visual QC
    t0 = time.monotonic()
    qc_res = run_visual_qc(spec_path, blend_path, qc_dir)
    record_stage("8_visual_qc", t0, qc_res)

    # Stage 9: Final audited GLB export & packed .blend
    t0 = time.monotonic()
    exp_proc = run_blender_script(
        ROOT / "tools/blender_village/export_village.py",
        ["--spec", str(spec_path), "--blend", str(blend_path), "--output", str(out_glb)],
        blend_path=blend_path,
    )
    record_stage("9_export_village", t0, exp_proc.stdout.strip().splitlines()[-1] if exp_proc.stdout.strip() else "ok")

    report["total_elapsed_sec"] = round(time.monotonic() - t_start, 2)
    report["glb"] = str(out_glb)
    report["glb_size_mb"] = round(out_glb.stat().st_size / (1024 * 1024), 2) if out_glb.is_file() else None
    report_path = out_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"FULL_PIPELINE_OK ({report['total_elapsed_sec']}s, GLB={report['glb_size_mb']} MB) -> {report_path}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/blender_village")
    p.add_argument("--colmap-dir", type=Path, default=ROOT / "artifacts/3dgs_build_clean/colmap/sparse/0")
    p.add_argument("--images-dir", type=Path, default=ROOT / "artifacts/3dgs_build_clean/colmap/images")
    p.add_argument("--output", type=Path, default=ROOT / "assets/scene_meshes/VILLAGE_BLENDER.glb")
    p.add_argument("--comfy-url", default=None, help="Live ComfyUI URL running MiniMax-H3 (or MINIMAX_H3_COMFY_URL)")
    p.add_argument("--h3-videos", type=Path, nargs="*", default=None, help="Optional pre-generated H3 orbit MP4 file(s)")
    p.add_argument("--bev-dsm-dir", type=Path, default=None, help="Optional directory containing bev_buildings.json")
    p.add_argument("--run-bev-dsm", action="store_true", help="Run DA3 dense multi-view BEV DSM reconstruction before prepare_evidence")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--skip-orbit-render", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_full_pipeline(
        out_dir=args.out_dir.resolve(),
        colmap_dir=args.colmap_dir.resolve(),
        images_dir=args.images_dir.resolve(),
        out_glb=args.output.resolve(),
        comfy_url=args.comfy_url,
        h3_videos=[p.resolve() for p in args.h3_videos] if args.h3_videos else None,
        bev_dsm_dir=args.bev_dsm_dir.resolve() if args.bev_dsm_dir else None,
        run_bev_dsm=args.run_bev_dsm,
        offline=args.offline,
        skip_orbit_render=args.skip_orbit_render,
    )
