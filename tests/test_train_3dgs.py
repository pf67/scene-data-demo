import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/scene_pipeline"), str(ROOT / "builder")]

from geometry import Camera  # noqa: E402
import train_3dgs as t3d  # noqa: E402


class Train3DGSTests(unittest.TestCase):
    def test_rotation_matrix_to_colmap_quat_roundtrip(self):
        rot = Rotation.from_euler("xyz", [22.5, -47.0, 11.2], degrees=True).as_matrix()
        q = t3d.rotation_matrix_to_colmap_quat(rot)
        self.assertEqual(q.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=6)
        recovered = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        np.testing.assert_allclose(recovered, rot, atol=1e-6)

    def test_sim3_alignment_and_colmap_text_export(self):
        # Construct synthetic OpenCV-convention cameras (r[0]=right, r[1]=down, r[2]=forward) in Y-up world.
        def look_at_opencv(eye, target):
            forward = target - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            return np.stack([right, down, forward], axis=0)

        size = np.array([1280.0, 720.0])
        k = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]])
        eye_world_ref = np.array([6.0, 38.0, 88.0])
        r_world_ref = look_at_opencv(eye_world_ref, np.array([0.0, 2.0, 0.0]))
        base_cam = Camera("ref", size, k, r_world_ref, -r_world_ref @ eye_world_ref)

        eye_world_2 = np.array([-40.0, 36.0, 60.0])
        r_world_2 = look_at_opencv(eye_world_2, np.array([0.0, 2.0, 0.0]))

        # Known 3D world points in front of both cameras.
        rng = np.random.default_rng(7)
        pts_world = np.column_stack(
            [
                rng.uniform(-25.0, 25.0, 40),
                rng.uniform(0.0, 8.0, 40),
                rng.uniform(-25.0, 25.0, 40),
            ]
        )

        # Apply an arbitrary Sim(3) from world -> synthetic COLMAP space.
        sim_rot = Rotation.from_euler("xyz", [40.0, 70.0, -30.0], degrees=True).as_matrix()
        sim_scale = 0.25
        sim_shift = np.array([3.5, -2.0, 7.0])

        def to_colmap(p_world):
            return (p_world - eye_world_ref) @ sim_rot.T * sim_scale + sim_shift

        pts_col = to_colmap(pts_world)

        def make_fake_image(img_id, name, cam_id, r_w, eye_w, track_len=4):
            r_col = r_w @ sim_rot.T
            eye_col = to_colmap(eye_w)
            t_col = -r_col @ eye_col
            pose = SimpleNamespace(rotation=SimpleNamespace(matrix=lambda r=r_col: r), translation=t_col)
            uv, _ = Camera(name, size, k, r_w, -r_w @ eye_w).pixels(pts_world)
            pts2d = [
                SimpleNamespace(
                    xy=uv[idx] + 0.5,
                    point3D_id=idx + 1,
                    has_point3D=lambda: True,
                )
                for idx in range(len(pts_world))
            ]
            return SimpleNamespace(
                image_id=img_id,
                name=name,
                camera_id=cam_id,
                cam_from_world=lambda p=pose: p,
                projection_center=lambda c=eye_col: c,
                points2D=pts2d,
            )

        ref_img = make_fake_image(1, "ref/0000_reference.jpg", 1, r_world_ref, eye_world_ref)
        other_img = make_fake_image(2, "clip0/00010.jpg", 1, r_world_2, eye_world_2)

        points3d = {
            idx + 1: SimpleNamespace(
                xyz=pts_col[idx],
                color=np.array([150, 140, 130]),
                error=0.4,
                track=SimpleNamespace(length=lambda: 4),
            )
            for idx in range(len(pts_world))
        }
        fake_cam = SimpleNamespace(
            model=SimpleNamespace(name="PINHOLE"),
            width=1280,
            height=720,
            params=np.array([1000.0, 1000.0, 640.0, 360.0]),
        )
        model = SimpleNamespace(
            images={1: ref_img, 2: other_img},
            reg_image_ids=lambda: [1, 2],
            points3D=points3d,
            cameras={1: fake_cam},
        )

        # Build exact reference depth buffer matching pts_world at ref_img pixels.
        ref_depth = np.full((720, 1280), 80.0, dtype=np.float32)
        uv_ref, z_ref = base_cam.pixels(pts_world)
        for (u, v), z in zip(uv_ref, z_ref):
            ix, iy = int(round(u)), int(round(v))
            if 2 <= ix < 1278 and 2 <= iy < 718:
                ref_depth[iy - 2 : iy + 3, ix - 2 : ix + 3] = z

        rotation, origin, factor, scale_err = t3d.compute_sim3_alignment(
            model, "ref/0000_reference.jpg", base_cam, ref_depth
        )
        self.assertLess(scale_err, 0.05)

        with tempfile.TemporaryDirectory() as tmp:
            sparse_dir = Path(tmp) / "sparse/0"
            report = t3d.write_aligned_colmap_sparse(
                model,
                sparse_dir,
                "ref/0000_reference.jpg",
                base_cam,
                rotation,
                origin,
                factor,
                np.array([0.0, 2.0, 0.0]),
                min_image_points=10,
            )
            self.assertEqual(report["registeredImages"], 2)
            self.assertEqual(report["points3D"], 40)

            img_lines = [
                ln
                for ln in (sparse_dir / "images.txt").read_text("utf8").splitlines()
                if not ln.startswith("#")
            ]
            self.assertEqual(len(img_lines), 4)
            parts2 = img_lines[2].split()
            qw, qx, qy, qz = (float(x) for x in parts2[1:5])
            tx, ty, tz = (float(x) for x in parts2[5:8])
            r_out = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t_out = np.array([tx, ty, tz])
            eye_out = -r_out.T @ t_out
            np.testing.assert_allclose(r_out, r_world_2, atol=1e-4)
            np.testing.assert_allclose(eye_out, eye_world_2, atol=1e-3)

            pt_lines = [
                ln
                for ln in (sparse_dir / "points3D.txt").read_text("utf8").splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            pts_out = np.array([[float(v) for v in ln.split()[1:4]] for ln in pt_lines])
            np.testing.assert_allclose(pts_out, pts_world, atol=1e-3)

    def test_prune_splat_ply(self):
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("opacity", "<f4"),
                ("scale_0", "<f4"),
                ("scale_1", "<f4"),
                ("scale_2", "<f4"),
            ]
        )
        rows = np.array(
            [
                (1.0, 2.0, 3.0, 1.0, -2.0, -2.0, -2.0),  # valid
                (1.0, 2.0, 3.0, 1.0, 0.5, 0.5, -5.0),  # flat 2D disk -> keep
                (np.nan, 0.0, 0.0, 1.0, -2.0, -2.0, -2.0),  # NaN
                (1.0, 2.0, 3.0, -6.0, -2.0, -2.0, -2.0),  # invisible opacity
                (1.0, 2.0, 3.0, 1.0, 5.0, -2.0, -2.0),  # giant floater
                (900.0, 2.0, 0.0, 1.0, -2.0, -2.0, -2.0),  # valid within 1200m
                (1500.0, 2.0, 0.0, 1.0, -2.0, -2.0, -2.0),  # out of radius
                (1.0, 25.0, 3.0, 1.0, -2.0, -2.0, -2.0),  # midair floater (y > 20.0)
                (1.0, 2.0, 3.0, 1.0, 0.5, -5.0, -5.0),  # 1D needle (s_max/s_mid > 14)
            ],
            dtype=dtype,
        )
        header = (
            "ply\nformat binary_little_endian 1.0\nelement vertex 9\n"
            + "\n".join(f"property float {name}" for name in dtype.names)
            + "\nend_header\n"
        ).encode("ascii")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / "in.ply"
            dst = tmp_path / "out.ply"
            src.write_bytes(header + rows.tobytes())
            stats = t3d.prune_splat_ply(src, dst)
            self.assertEqual(stats["inputSplats"], 9)
            self.assertEqual(stats["keptSplats"], 3)
            self.assertEqual(stats["removedSplats"], 6)
            self.assertIn("radiusP99", stats)
            self.assertIn("maxLogScaleP99", stats)
            raw = dst.read_bytes()
            self.assertIn(b"element vertex 3\n", raw)

            stats_bounded = t3d.prune_splat_ply(src, dst, bounds_xz=(-10.0, 10.0, -10.0, 10.0))
            self.assertEqual(stats_bounded["keptSplats"], 2)

    def test_build_match_pairs(self):
        k = np.eye(3)
        clip0 = [f"clip0/{i:05d}.jpg" for i in range(18)]
        clip1 = [f"clip1/{i:05d}.jpg" for i in range(12)]
        groups = [
            ("ref", k, ["ref/0000_reference.jpg"]),
            ("clip0", k, clip0),
            ("clip1", k, clip1),
        ]
        pairs = t3d.build_match_pairs("ref/0000_reference.jpg", groups)
        pair_set = set(pairs)
        # No self-pairs and canonical sorted pairs with no duplicates
        self.assertEqual(len(pairs), len(pair_set))
        for a, b in pairs:
            self.assertLess(a, b)
        # Sequential offset checks
        self.assertIn(("clip0/00000.jpg", "clip0/00001.jpg"), pair_set)
        self.assertIn(("clip0/00000.jpg", "clip0/00014.jpg"), pair_set)
        # Loop closure
        self.assertIn(("clip0/00000.jpg", "clip0/00017.jpg"), pair_set)
        # Reference anchor
        self.assertIn(("clip0/00000.jpg", "ref/0000_reference.jpg"), pair_set)
        self.assertIn(("clip1/00011.jpg", "ref/0000_reference.jpg"), pair_set)
        # Cross-clip overlap
        self.assertIn(("clip0/00000.jpg", "clip1/00000.jpg"), pair_set)
        self.assertIn(("clip0/00005.jpg", "clip1/00005.jpg"), pair_set)

    def test_train_with_brush_rejects_stray_bin_or_ply(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            fake_bin = tmp_path / "brush_app.exe"
            fake_bin.write_bytes(b"")
            colmap_dir = tmp_path / "colmap"
            (colmap_dir / "sparse/0").mkdir(parents=True)
            (colmap_dir / "sparse/0/cameras.txt").write_text("#\n", encoding="utf8")
            (colmap_dir / "raw_sparse/0").mkdir(parents=True)
            (colmap_dir / "raw_sparse/0/cameras.bin").write_bytes(b"0")
            with self.assertRaises(ValueError):
                t3d.train_with_brush(fake_bin, colmap_dir, tmp_path / "export")

            (colmap_dir / "raw_sparse/0/cameras.bin").unlink()
            (colmap_dir / "stray.ply").write_bytes(b"ply\n")
            with self.assertRaises(ValueError):
                t3d.train_with_brush(fake_bin, colmap_dir, tmp_path / "export")


if __name__ == "__main__":
    unittest.main()
