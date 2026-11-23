"""Unit and contract tests for general single-drone-photo 3D Blender reconstruction pipeline."""

from __future__ import annotations

import json
import math
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "tools/blender_village"),
    str(ROOT / "tools/blender_bridge"),
    str(ROOT / "tools/scene_pipeline"),
]

import layout_solver as ls  # noqa: E402
import reconstruct_from_photo as rfp  # noqa: E402
import visual_qc  # noqa: E402


class GeneralPhotoPipelineTests(unittest.TestCase):
    def _make_synthetic_oblique_scene(
        self,
        *,
        width: int = 240,
        height: int = 160,
        cam_height_m: float = 36.0,
        cam_dist_z: float = 48.0,
        scale_div: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate exact synthetic oblique drone depth map with flat ground (Y=0) + 3 extruded boxes."""
        fx = fy = float(width) * 1.08
        cx, cy = width * 0.5, height * 0.5
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        eye_w = np.array([0.0, cam_height_m, cam_dist_z], dtype=np.float64)
        target_w = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        fwd_w = target_w - eye_w
        fwd_w /= np.linalg.norm(fwd_w)
        # Camera +Z is forward, +Y is down in OpenCV
        up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right_w = np.cross(fwd_w, up_world)
        right_w /= np.linalg.norm(right_w)
        cam_down_w = np.cross(fwd_w, right_w)
        cam_down_w /= np.linalg.norm(cam_down_w)
        # R_cw rows = cam basis vectors expressed in world coords
        r_cw = np.stack([right_w, cam_down_w, fwd_w], axis=0)
        t_cw = -r_cw @ eye_w

        us, vs = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
        rays_c = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us)], axis=-1)
        rays_w = rays_c @ r_cw  # (H, W, 3)

        # Ray-plane intersection with Y = 0 ground
        t_hit = -eye_w[1] / np.minimum(rays_w[:, :, 1], -1e-4)
        pts_w = eye_w[None, None, :] + rays_w * t_hit[:, :, None]

        # Place 2 synthetic boxes (y=6.0m rooftops) inside center frame without green hue
        rgb = np.full((height, width, 3), 120, dtype=np.uint8)
        rgb[:, :, 0] = 85
        rgb[:, :, 1] = 130  # grass ground
        rgb[:, :, 2] = 65

        for bx0, bx1, bz0, bz1, bh in [(-18.0, -6.0, -4.0, 6.0, 6.0), (8.0, 20.0, -8.0, 4.0, 6.0)]:
            # Intersect Y = bh plane where rooftop lies inside [bx0..bx1, bz0..bz1]
            t_roof = (bh - eye_w[1]) / np.minimum(rays_w[:, :, 1], -1e-4)
            pts_roof = eye_w[None, None, :] + rays_w * t_roof[:, :, None]
            on_roof = (
                (pts_roof[:, :, 0] >= bx0)
                & (pts_roof[:, :, 0] <= bx1)
                & (pts_roof[:, :, 2] >= bz0)
                & (pts_roof[:, :, 2] <= bz1)
            )
            t_hit = np.where(on_roof, t_roof, t_hit)
            pts_w = np.where(on_roof[:, :, None], pts_roof, pts_w)
            rgb[on_roof] = (145, 78, 62)

        pts_c = pts_w @ r_cw.T + t_cw
        depth = pts_c[:, :, 2] / scale_div
        return rgb, depth, k, eye_w, r_cw

    def test_ground_plane_and_camera_reprojection_roundtrip(self):
        rgb, depth, k, eye_gt, _ = self._make_synthetic_oblique_scene(scale_div=1.0)
        pose = rfp.fit_ground_plane_and_camera(rgb, depth, k, sample_step=1, target_relief_m=6.0)
        self.assertAlmostEqual(float(np.linalg.det(pose["r_cw"])), 1.0, places=4)
        self.assertLess(abs(float(pose["eye_w"][1]) - float(eye_gt[1])), 0.8)
        self.assertLess(abs(float(pose["eye_w"][2]) - float(eye_gt[2])), 0.8)

        # Reproject reconstructed world points back into camera pixels and check < 0.5px error
        h, w = depth.shape
        us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        gt_uv = np.stack([us, vs], axis=-1).reshape(-1, 2)
        uv_proj, zc = rfp.project_world_to_photo(pose["pts_world"], pose["r_cw"], pose["t_cw"], k)
        ok = pose["valid_mask_flat"] & (zc > 1.0)
        err = np.linalg.norm(uv_proj[ok] - gt_uv[ok], axis=1)
        self.assertLess(float(np.median(err)), 0.05)
        self.assertLess(float(np.max(err)), 0.5)

    def test_synthetic_scaled_depth_and_bev_extraction(self):
        # Simulate canonical compressed depth (scale_div=25) -> self-calibrating scale recovers metric buildings
        rgb, depth, k, _, _ = self._make_synthetic_oblique_scene(width=600, height=400, scale_div=25.0)
        pose = rfp.fit_ground_plane_and_camera(rgb, depth, k, sample_step=1, target_relief_m=6.0)
        self.assertGreater(pose["scale_factor"], 15.0)
        bev = rfp.rasterize_single_view_bev(pose["pts_world"], pose["pts_rgb"], pose["pts_conf"], pose["valid_mask_flat"])
        road = rfp.extract_road_centerline(bev, None, pose, k)
        out = rfp.extract_buildings_from_bev(bev, road, rgb, pose, k)
        self.assertEqual(out["layout_audit"]["verdict"], "pass")
        self.assertGreaterEqual(len(out["buildings"]), 2)

    def test_roof_and_wall_swatch_classifiers(self):
        self.assertEqual(rfp.classify_roof_from_rgb(np.array([150.0, 82.0, 62.0]), 9.0, 7.0)[0], "terracotta_tile")
        self.assertEqual(rfp.classify_roof_from_rgb(np.array([62.0, 95.0, 148.0]), 9.0, 7.0)[0], "corrugated_blue")
        self.assertEqual(rfp.classify_roof_from_rgb(np.array([68.0, 72.0, 76.0]), 9.0, 7.0)[0], "slate_roof")
        self.assertEqual(rfp.classify_roof_from_rgb(np.array([190.0, 188.0, 184.0]), 9.0, 7.0)[0], "concrete_roof")

        self.assertEqual(rfp.classify_wall_from_rgb(np.array([148.0, 88.0, 72.0])), "red_brick")
        self.assertEqual(rfp.classify_wall_from_rgb(np.array([228.0, 224.0, 218.0])), "plaster_white")
        self.assertEqual(rfp.classify_wall_from_rgb(np.array([185.0, 165.0, 138.0])), "plaster_warm")
        self.assertEqual(rfp.classify_wall_from_rgb(np.array([128.0, 130.0, 132.0])), "plaster_grey")

    def test_adaptive_qc_views_centering(self):
        fake_spec = {
            "imageSize": [1536, 1024],
            "camera": {"position": [10.0, 120.0, 140.0], "target": [10.0, 0.0, 20.0], "fov": 30.0},
            "road_centerline_xz": [[-20.0, 25.0], [40.0, 15.0]],
            "buildings": [
                {"footprint_xz": [[5.0, 15.0], [15.0, 15.0], [15.0, 25.0], [5.0, 25.0]]},
                {"footprint_xz": [[25.0, 35.0], [35.0, 35.0], [35.0, 45.0], [25.0, 45.0]]},
            ],
        }
        views = visual_qc.compute_adaptive_qc_views(fake_spec)
        names = [v[0] for v in views]
        self.assertEqual(
            names,
            [
                "01_ref_raw.png",
                "04_orbit_right30.png",
                "05_orbit_left35.png",
                "06_rear_180.png",
                "07_street_level.png",
                "08_top_overview.png",
            ],
        )
        top_cfg = dict(views)["08_top_overview.png"]
        self.assertAlmostEqual(top_cfg["target"][0], 20.0, places=2)
        self.assertAlmostEqual(top_cfg["target"][2], 30.0, places=2)

    def test_exported_uk_village_glb_and_metadata_contract(self):
        glb_path = ROOT / "assets/scene_meshes/UK_VILLAGE_BLENDER.glb"
        meta_path = ROOT / "assets/scene_meshes/UK_VILLAGE_BLENDER.json"
        topo_path = ROOT / "assets/scene_meshes/UK_VILLAGE_BLENDER.topology.json"
        src_path = ROOT / "assets/scene_sources/uk_houses_drone.png"
        spec_path = ROOT / "artifacts/generalization_test/uk_houses/village_spec.json"

        for p in (glb_path, meta_path, topo_path, src_path, spec_path):
            self.assertTrue(p.is_file(), f"Missing {p}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["code"], "UK_VILLAGE_BLENDER")
        self.assertEqual(meta["kind"], "photo_completed_village")
        self.assertEqual(meta["source_image"], "uk_houses_drone.png")
        self.assertGreaterEqual(int(meta["building_count"]), 60)

        topo = json.loads(topo_path.read_text(encoding="utf-8"))
        self.assertEqual(topo["verdict"], "pass")
        self.assertEqual(topo["failure_count"], 0)
        self.assertEqual(int(topo["closed_buildings"]), int(meta["building_count"]))

        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        layout_audit = ls.audit_layout_geometry(
            spec["buildings"],
            spec["road_centerline_xz"],
            spec["road_polygon_xz"],
            spec["road_apron_polygon_xz"],
        )
        self.assertEqual(layout_audit["verdict"], "pass", layout_audit)

        data = glb_path.read_bytes()
        magic, version, length = struct.unpack_from("<III", data)
        self.assertEqual((magic, version, length), (0x46546C67, 2, len(data)))
        json_len = struct.unpack_from("<I", data, 12)[0]
        doc = json.loads(data[20 : 20 + json_len])
        houses = [n for n in doc["nodes"] if n.get("extras", {}).get("closed_building")]
        self.assertEqual(len(houses), len(spec["buildings"]))
        self.assertTrue(all("bufferView" in img for img in doc.get("images", [])))


if __name__ == "__main__":
    unittest.main()
