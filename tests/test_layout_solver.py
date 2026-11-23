"""Unit and regression tests for street-grid road regularization, layout solver, and collision/skew gating."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/blender_village")]

import layout_solver as ls  # noqa: E402


class LayoutSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.centerline = [
            [-36.91, 28.13],
            [-19.43, 12.4],
            [6.66, -10.05],
            [47.29, -41.14],
            [92.12, -63.43],
        ]
        self.road = ls.build_regularized_road(self.centerline, carriageway_half_width=4.4, apron_width=3.2)

    def test_constant_road_width_and_aprons(self):
        left = np.asarray(self.road["road_left_edge_xz"])
        right = np.asarray(self.road["road_right_edge_xz"])
        widths = np.linalg.norm(left - right, axis=1)
        self.assertEqual(len(widths), len(self.centerline))
        for w in widths:
            self.assertAlmostEqual(float(w), 8.8, delta=0.15)
        self.assertGreaterEqual(len(self.road["road_polygon_xz"]), 10)
        self.assertGreaterEqual(len(self.road["road_apron_polygon_xz"]), 10)

    def test_regularize_skewed_parallelogram_preserves_area_and_kills_skew(self):
        _, _, u, n_s, _ = ls.local_road_frame([50.3, -44.7], self.centerline)
        # Construct a 27-degree skewed parallelogram mirroring house_33
        skew_axis = u * np.cos(np.radians(27.0)) + n_s * np.sin(np.radians(27.0))
        c = np.array([50.3, -44.7])
        raw_poly = np.array(
            [
                c - u * 4.8 - skew_axis * 4.0,
                c + u * 4.8 - skew_axis * 4.0,
                c + u * 4.8 + skew_axis * 4.0,
                c - u * 4.8 + skew_axis * 4.0,
            ]
        )
        self.assertGreater(ls.polygon_corner_skew_deg(raw_poly), 20.0)
        reg = ls.regularize_footprint_to_grid(raw_poly, self.centerline)
        self.assertLess(ls.polygon_corner_skew_deg(reg), 0.1)
        from shapely.geometry import Polygon

        area_raw = Polygon(raw_poly).area
        area_reg = Polygon(reg).area
        self.assertLess(abs(area_reg - area_raw) / area_raw, 0.05)

    def test_solve_clears_road_apron_and_separates_overlaps(self):
        _, _, u_dir, n_s, near = ls.local_road_frame([50.3, -44.7], self.centerline)
        # Building 1: centered only 1.8m North of road centerline (mirrors house_33 intruding carriageway)
        q_road = ls.make_oriented_quad_xz(
            float((near - n_s * 1.8)[0]),
            float((near - n_s * 1.8)[1]),
            9.5,
            8.0,
            u_dir,
        )
        # Building 2 & 3: overlapping South-side pair (mirrors house_25/24 or 47/46)
        _, _, u_s, n_ss, near_s = ls.local_road_frame([-15.0, -8.2], self.centerline)
        c_pair = near_s + n_ss * 13.0
        q_ov_a = ls.make_oriented_quad_xz(float(c_pair[0]), float(c_pair[1]), 9.0, 7.0, u_s)
        c_pair_b = c_pair + u_s * 4.5 + n_ss * 1.0
        q_ov_b = ls.make_oriented_quad_xz(float(c_pair_b[0]), float(c_pair_b[1]), 9.0, 7.0, u_s)

        buildings = [
            {"id": "intruder", "index": 0, "is_core": True, "height_m": 4.6, "footprint_xz": q_road.tolist()},
            {"id": "ov_a", "index": 1, "is_core": True, "height_m": 6.2, "footprint_xz": q_ov_a.tolist()},
            {"id": "ov_b", "index": 2, "is_core": True, "height_m": 5.0, "footprint_xz": q_ov_b.tolist()},
        ]
        # Before solving, audit must FAIL
        pre = ls.audit_layout_geometry(
            buildings, self.centerline, self.road["road_polygon_xz"], self.road["road_apron_polygon_xz"]
        )
        self.assertEqual(pre["verdict"], "fail")
        self.assertGreater(pre["road_intrusion_count"], 0)
        self.assertGreater(pre["building_overlap_count"], 0)

        solved = ls.solve_building_layout(
            buildings, self.centerline, self.road["road_polygon_xz"], self.road["road_apron_polygon_xz"]
        )
        post = ls.audit_layout_geometry(
            solved, self.centerline, self.road["road_polygon_xz"], self.road["road_apron_polygon_xz"]
        )
        self.assertEqual(post["verdict"], "pass", post)
        self.assertEqual(post["road_intrusion_count"], 0)
        self.assertEqual(post["apron_intrusion_count"], 0)
        self.assertEqual(post["building_overlap_count"], 0)

        # Intruder originally had negative signed_dist -> must remain on North side (sd < 0)
        c_solved = np.asarray(solved[0]["footprint_xz"]).mean(0)
        _, sd_after, _, _, _ = ls.local_road_frame(c_solved, self.centerline)
        self.assertLess(sd_after, -7.9)


if __name__ == "__main__":
    unittest.main()
