"""Regression tests for multi-view orbit synthesis, ECC alignment, blind-spot BEV fusion, and facade compositing."""

from __future__ import annotations

import io
import json
import os
import struct
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/blender_village"), str(ROOT / "tools/scene_pipeline")]

import bake_multiview as bm  # noqa: E402
import colab_h3_setup as ch3  # noqa: E402
import reconstruct_from_photo as rfp  # noqa: E402
import synthesize_multiview as sm  # noqa: E402


def _make_synthetic_village_guide(width: int = 640, height: int = 424) -> np.ndarray:
    rng = np.random.default_rng(19)
    img = np.full((height, width, 3), 92, dtype=np.uint8)
    img[:, :, 1] = 122
    cv2.rectangle(img, (0, int(height * 0.65)), (width, height), (140, 140, 145), -1)
    for x0, y0, bw, bh, c in (
        (90, 140, 120, 75, (195, 185, 165)),
        (270, 120, 140, 90, (175, 182, 196)),
        (460, 155, 110, 70, (205, 192, 174)),
        (170, 240, 130, 80, (186, 178, 162)),
        (365, 235, 125, 85, (190, 185, 170)),
    ):
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), c, -1)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + 22), (72, 76, 88), -1)
        cv2.rectangle(img, (x0 + 18, y0 + 34), (x0 + 44, y0 + 58), (52, 58, 66), -1)
    noise = rng.integers(-6, 7, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


class MultiViewSynthesisTests(unittest.TestCase):
    def test_compute_multiview_orbit_cameras_geometry_and_keys(self):
        spec = {
            "camera": {"position": [12.0, 48.0, 62.0], "target": [0.0, 0.0, 0.0], "fov": 42.0},
            "buildings": [
                {"footprint_xz": [[0.0, -4.0], [8.0, -4.0], [8.0, 4.0], [0.0, 4.0]]},
                {"footprint_xz": [[-8.0, -4.0], [0.0, -4.0], [0.0, 4.0], [-8.0, 4.0]]},
            ],
        }
        cams = sm.compute_multiview_orbit_cameras(spec, [-60.0, 60.0, 180.0], width=1248, height=832)
        self.assertEqual([c["cam_key"] for c in cams], ["novel_left60", "novel_right60", "novel_rear180"])
        r0 = np.linalg.norm(np.array([12.0, 62.0]))
        for cam in cams:
            eye = np.asarray(cam["eye_three"], dtype=np.float64)
            tgt = np.asarray(cam["target_three"], dtype=np.float64)
            r_cw = np.asarray(cam["r_cw"], dtype=np.float64)
            self.assertTrue(cam["allow_facade_bake"])
            self.assertAlmostEqual(eye[1], 48.0, places=4)
            self.assertAlmostEqual(float(np.linalg.norm(eye[[0, 2]] - tgt[[0, 2]])), r0, places=2)
            self.assertTrue(np.allclose(r_cw @ r_cw.T, np.eye(3), atol=1e-5))
        self.assertLess(cams[0]["eye_three"][0], cams[1]["eye_three"][0])
        self.assertAlmostEqual(cams[2]["eye_three"][2], -62.0, places=1)

    def test_align_and_verify_novel_view_recovers_shift_and_rejects_noise(self):
        guide = _make_synthetic_village_guide()
        h, w = guide.shape[:2]
        m_shift = np.array([[1.0, 0.0, 42.0], [0.0, 1.0, -28.0]], dtype=np.float32)
        shifted = cv2.warpAffine(guide, m_shift, (w, h), borderMode=cv2.BORDER_REPLICATE)

        res = sm.align_and_verify_novel_view(guide, shifted, guide)
        self.assertTrue(res["warp_applied"], res)
        self.assertGreater(res["after_corr"], res["before_corr"])
        self.assertGreaterEqual(res["after_corr"], 0.70)
        self.assertTrue(res["accepted"])

        rng = np.random.default_rng(42)
        junk = rng.integers(0, 255, size=guide.shape, dtype=np.uint8)
        bad = sm.align_and_verify_novel_view(guide, junk, guide)
        self.assertFalse(bad["accepted"])

    def test_resolve_provider_routing_and_synthesize_novel_views_mocked(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "MINIMAX_API_KEY": ""}, clear=True),
            patch("synthesize_multiview.mm_img.probe_comfy_gateway", return_value=None),
        ):
            with patch("synthesize_multiview.mm_img.ping_comfyui", return_value={"ok": True, "has_minimax_h3_image": True}):
                self.assertEqual(sm.resolve_multiview_provider("auto", comfy_url="http://mock:8188"), "minimax_h3_image")
            with (
                patch("synthesize_multiview.mm_img.ping_comfyui", return_value={"ok": False}),
                patch("synthesize_multiview.mm_orb.check_comfy_health", return_value={"ok": True}),
            ):
                self.assertEqual(sm.resolve_multiview_provider("auto", comfy_url="http://mock:8188"), "minimax_h3_video")
            with patch("synthesize_multiview.mm_img.ping_comfyui", return_value={"ok": False, "gateway_state": "reconnecting", "message": "booting"}):
                self.assertEqual(sm.resolve_multiview_provider("auto", comfy_url="http://127.0.0.1:8188"), "gpt_image_2")
            self.assertEqual(sm.resolve_multiview_provider("auto", comfy_url=None), "gpt_image_2")
        # Auto-discovery of local fixed gateway http://127.0.0.1:8188 when env has no COMFYUI_URL
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("synthesize_multiview.mm_img.probe_comfy_gateway", return_value={"state": "ready"}),
        ):
            self.assertEqual(sm._resolve_comfy_url(None), sm.mm_img.DEFAULT_LOCAL_COMFY_GATEWAY)

        guide = _make_synthetic_village_guide(480, 320)
        h, w = guide.shape[:2]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref_path = root / "ref.png"
            blend_path = root / "scene.blend"
            spec_path = root / "village_spec.json"
            blend_path.write_text("stub", encoding="utf-8")
            Image.fromarray(guide).save(ref_path)
            cams0 = sm.compute_multiview_orbit_cameras(
                {"camera": {"position": [0.0, 42.0, 58.0], "target": [0.0, 0.0, 0.0], "fov": 42.0}, "buildings": []},
                [-60.0, 180.0],
                width=w,
                height=h,
            )
            spec_path.write_text(
                json.dumps(
                    {
                        "ref_clean_image": str(ref_path),
                        "camera": {"position": [0.0, 42.0, 58.0], "target": [0.0, 0.0, 0.0], "fov": 42.0},
                        "buildings": [],
                        "cameras": [{"cam_key": "ref", "source_clip": "ref", "image_path": str(ref_path)}],
                    }
                ),
                encoding="utf-8",
            )
            g_dir = root / "mv_out" / "multiview" / "guides"
            g_dir.mkdir(parents=True, exist_ok=True)
            for c in cams0:
                gp = g_dir / f"guide_{c['cam_key']}.png"
                Image.fromarray(guide).save(gp)
                c["guide_path"] = str(gp)

            def _fake_generate_view(*_a, **_kw):
                return _png_bytes(guide), {"mocked": True}

            with (
                patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock", "MINIMAX_H3_COMFY_URL": "http://mock:8188"}, clear=False),
                patch("synthesize_multiview.render_multiview_guides", return_value=cams0),
                patch("synthesize_multiview.image_api.generate_view", side_effect=_fake_generate_view) as gpt_mock,
            ):
                res1 = sm.synthesize_novel_views(
                    spec_path=spec_path,
                    blend_path=blend_path,
                    out_dir=root / "mv_out",
                    angles_deg=[-60.0, 180.0],
                    provider="gpt_image_2",
                )
                self.assertEqual(res1["provider_used"], "gpt_image_2")
                self.assertEqual(res1["report"]["accepted_count"], 2)
                self.assertEqual(gpt_mock.call_count, 2)
                self.assertTrue(Path(res1["report_path"]).is_file())

                # Verify disk cache hit avoids duplicate remote API calls
                res2 = sm.synthesize_novel_views(
                    spec_path=spec_path,
                    blend_path=blend_path,
                    out_dir=root / "mv_out",
                    angles_deg=[-60.0, 180.0],
                    provider="gpt_image_2",
                    force=False,
                )
                self.assertEqual(gpt_mock.call_count, 2)
                self.assertTrue(all(v["meta"].get("cached") for v in res2["report"]["views"]))

            with (
                patch("synthesize_multiview.render_multiview_guides", return_value=cams0),
                patch("synthesize_multiview.mm_img.call_comfyui_minimax_h3", return_value=(_png_bytes(guide), {"node": "h3_img"})) as mm_i,
            ):
                res_mmi = sm.synthesize_novel_views(
                    spec_path=spec_path,
                    blend_path=blend_path,
                    out_dir=root / "mv_mm_img",
                    angles_deg=[-60.0, 180.0],
                    provider="minimax_h3_image",
                    comfy_url="http://mock:8188",
                )
                self.assertEqual(mm_i.call_count, 2)
                self.assertEqual(res_mmi["report"]["accepted_count"], 2)

            v_matches = {c["cam_key"]: (Image.fromarray(guide), {"frame_idx": i}) for i, c in enumerate(cams0)}
            fake_vid = root / "mv_mm_vid" / "multiview" / "h3_video" / "orbit.mp4"
            fake_vid.parent.mkdir(parents=True, exist_ok=True)
            fake_vid.write_bytes(b"mp4")
            with (
                patch("synthesize_multiview.render_multiview_guides", return_value=cams0),
                patch("synthesize_multiview.mm_orb.check_comfy_health", return_value={"ok": True}),
                patch("synthesize_multiview.mm_orb.run_orbit_generation", return_value={"videoPath": str(fake_vid)}),
                patch("synthesize_multiview._extract_calibrated_video_views", return_value=v_matches),
            ):
                res_mmv = sm.synthesize_novel_views(
                    spec_path=spec_path,
                    blend_path=blend_path,
                    out_dir=root / "mv_mm_vid",
                    angles_deg=[-60.0, 180.0],
                    provider="minimax_h3_video",
                    comfy_url="http://mock:8188",
                )
                self.assertEqual(res_mmv["report"]["accepted_count"], 2)

    def test_composite_novel_facade_rejects_slate_shadow_and_foliage(self):
        h, w = 64, 64
        fb = np.full((h, w, 3), [188.0, 176.0, 158.0], dtype=np.float32)
        fb[::2, ::2] += 8.0
        obs_alpha = np.full((h, w), 0.9, dtype=np.float32)
        face_mask = np.full((h, w), 255, dtype=np.uint8)
        interior = np.ones((h, w), dtype=bool)
        tg = np.broadcast_to(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], (h, w)).copy()
        ao_up = np.ones((h, w), dtype=np.float32)
        u_m = np.broadcast_to(np.linspace(0.0, 6.0, w, dtype=np.float32)[None, :], (h, w)).copy()
        v_m = np.broadcast_to(np.linspace(0.0, 4.0, h, dtype=np.float32)[:, None], (h, w)).copy()

        # Valid warm stone masonry observation -> accepted & tinted
        obs_warm = np.full((h, w, 3), [208.0, 192.0, 168.0], dtype=np.float32)
        comp_ok, accepted = bm.composite_novel_facade(fb, obs_warm, obs_alpha, face_mask, interior, tg, ao_up, u_m, v_m, 4.0, seed=1)
        self.assertTrue(accepted)
        self.assertGreater(float(comp_ok[32, 32, 0]), float(comp_ok[32, 32, 2]))

        # Slate-blue roof smear, deep cast shadow, and bright green vegetation must ALL be rejected
        for bad_rgb in (
            [72.0, 82.0, 118.0],
            [28.0, 28.0, 32.0],
            [40.0, 165.0, 35.0],
        ):
            obs_bad = np.full((h, w, 3), bad_rgb, dtype=np.float32)
            comp_bad, rej = bm.composite_novel_facade(fb, obs_bad, obs_alpha, face_mask, interior, tg, ao_up, u_m, v_m, 4.0, seed=1)
            self.assertFalse(rej, f"Expected rejection for {bad_rgb}")
            self.assertGreater(float(comp_bad.mean()), 80.0)

    def test_merge_blindspot_bev_preserves_stage1_cells(self):
        gw = gh = 16
        stage1_obs = np.zeros((gh, gw), dtype=np.int32)
        stage1_obs[2:8, 2:8] = 5
        stage1_dsm = np.where(stage1_obs > 0, 4.2, np.nan).astype(np.float32)
        stage1_dtm = np.zeros((gh, gw), dtype=np.float32)
        stage1_ortho = np.zeros((gh, gw, 3), dtype=np.uint8)
        stage1_ortho[2:8, 2:8] = [190, 175, 160]
        stage1_bev = {
            "bounds_xz": (0.0, 16.0, 0.0, 16.0),
            "res_m": 1.0,
            "gw": gw,
            "gh": gh,
            "dsm": stage1_dsm,
            "dtm": stage1_dtm,
            "ndsm": np.where(stage1_obs > 0, 4.2, 0.0).astype(np.float32),
            "ortho": stage1_ortho,
            "obs_count": stage1_obs,
            "veg_mask": np.zeros((gh, gw), dtype=bool),
            "grass_mask": np.zeros((gh, gw), dtype=bool),
            "road_prior": np.zeros((gh, gw), dtype=bool),
            "xx": np.zeros((gh, gw), dtype=np.float64),
            "zz": np.zeros((gh, gw), dtype=np.float64),
        }
        mv_pts = np.array([[4.5, 9.9, 4.5], [11.5, 5.5, 11.5], [12.5, 5.5, 12.5]], dtype=np.float32)
        mv_rgb = np.array([[10, 10, 10], [180, 160, 140], [180, 160, 140]], dtype=np.uint8)
        mv_conf = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        merged = rfp._merge_blindspot_bev(stage1_bev, mv_pts, mv_rgb, mv_conf)
        self.assertTrue(np.array_equal(merged["obs_count"][2:8, 2:8], stage1_obs[2:8, 2:8]))
        self.assertTrue(np.allclose(merged["dsm"][2:8, 2:8], 4.2))
        self.assertGreater(merged["obs_count"][11, 11], 0)
        self.assertAlmostEqual(float(merged["ndsm"][11, 11]), 5.5, places=3)

    def test_colab_h3_setup_writes_dual_mode_notebook(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb_path = Path(tmp) / "MiniMax_H3_MultiView_Colab.ipynb"
            out = ch3.write_colab_notebook(nb_path)
            self.assertTrue(out.is_file())
            raw = out.read_text(encoding="utf-8")
            doc = json.loads(raw)
            self.assertEqual(doc["nbformat"], 4)
            self.assertIn("MiniMaxH3ImageConditioning", raw)
            self.assertIn("MiniMaxH3ImageToVideo", raw)
            self.assertIn("qwen3vl_32b_minimax_h3_int8_convrot.safetensors", raw)
            self.assertIn("minimax_h3_image_pruned_int8_convrot.safetensors", raw)
            self.assertIn("MINIMAX_H3_MODE", raw)

    def test_uk_village_blender_multiview_glb_and_audit_contract(self):
        out_dir = ROOT / "artifacts/generalization_test/uk_houses"
        glb_path = ROOT / "assets/scene_meshes/UK_VILLAGE_BLENDER.glb"
        meta_path = ROOT / "assets/scene_meshes/UK_VILLAGE_BLENDER.json"
        topo_path = ROOT / "assets/scene_meshes/UK_VILLAGE_BLENDER.topology.json"
        mv_sum_path = out_dir / "multiview_summary.json"
        qc_path = out_dir / "qc/qc_metrics.json"

        for p in (glb_path, meta_path, topo_path, mv_sum_path, qc_path):
            self.assertTrue(p.is_file(), f"Missing {p}")

        mv_sum = json.loads(mv_sum_path.read_text(encoding="utf-8"))
        p1, p2 = mv_sum["pass1_single_view"], mv_sum["pass2_multiview"]
        self.assertGreaterEqual(mv_sum["synthesis_report"]["accepted_count"], 2)
        self.assertGreater(p2["photo_facade_faces"], p1["photo_facade_faces"])
        self.assertGreaterEqual(p2["bev_obs_coverage_pct"], p1["bev_obs_coverage_pct"])

        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        self.assertIn("01_ref_compare", qc)
        self.assertGreaterEqual(qc["01_ref_compare"]["luma_corr"], 0.25)

        topo = json.loads(topo_path.read_text(encoding="utf-8"))
        self.assertEqual(topo["verdict"], "pass", topo)
        self.assertEqual(topo["failure_count"], 0)

        data = glb_path.read_bytes()
        magic, version, length = struct.unpack_from("<III", data)
        self.assertEqual((magic, version, length), (0x46546C67, 2, len(data)))
        json_len = struct.unpack_from("<I", data, 12)[0]
        doc = json.loads(data[20 : 20 + json_len])
        binary = memoryview(data)[28 + json_len :]
        dtypes = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}

        def accessor(index: int) -> np.ndarray:
            info = doc["accessors"][index]
            view = doc["bufferViews"][info["bufferView"]]
            count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[info["type"]]
            dtype = np.dtype(dtypes[info["componentType"]])
            offset = view.get("byteOffset", 0) + info.get("byteOffset", 0)
            stride = view.get("byteStride", dtype.itemsize * count)
            return np.ndarray((info["count"], count), dtype, binary, offset, strides=(stride, dtype.itemsize)).copy()

        houses = [n for n in doc["nodes"] if n.get("extras", {}).get("closed_building")]
        self.assertEqual(len(houses), p2["buildings"])
        for node in houses:
            triangles = []
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                verts = accessor(prim["attributes"]["POSITION"])
                idx = accessor(prim["indices"]).ravel().reshape(-1, 3)
                triangles.append(verts[idx])
            tris = np.concatenate(triangles)
            _, inv = np.unique(np.round(tris.reshape(-1, 3), 5), axis=0, return_inverse=True)
            faces = inv.reshape(-1, 3)
            edges = Counter(tuple(sorted((int(a), int(b)))) for face in faces for a, b in zip(face, np.roll(face, -1)))
            self.assertTrue(all(cnt == 2 for cnt in edges.values()), f"Non-watertight {node.get('name')}")


if __name__ == "__main__":
    unittest.main()
