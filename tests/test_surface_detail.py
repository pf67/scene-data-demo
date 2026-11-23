import json
import struct
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "builder"))
from geometry import Camera
from mesh import Glb
from photo_projectors import visibility_image
from surface_depth import SurfaceDepth
from surface_marks import SurfaceMarks
from view_surface import add_view_surfaces, triangulate
from worker import orbit_coverage


class SurfaceDetailTests(unittest.TestCase):
    def setUp(self):
        size = np.array([64, 48])
        k = np.array([[55.0, 0, 32], [0, 55.0, 24], [0, 0, 1]])
        r = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
        self.cam = Camera("photo", size, k, r, -r @ [0, 10, 0])
        self.image = np.zeros((48, 64, 3), np.uint8)
        z = np.full((48, 64), 10.0, np.float32)
        z[:, :32] = 6.0
        normal = np.zeros((48, 64, 3), np.float32)
        normal[:, :, 1] = 1
        self.depth = SurfaceDepth(self.cam, z, normal, 10 - z, 0.8)

    def test_rooftop_rejected_as_ground_and_depth_samples_reproject(self):
        accepted = self.depth.check(np.array([[10.0, 12], [48.0, 12]]), [10, 10], None, self.cam.size)
        np.testing.assert_array_equal(accepted, [False, True])
        (vertices, normals, uv), indices, coverage, radius, margins = triangulate(self.depth)
        projected, z = self.cam.pixels(vertices)
        np.testing.assert_allclose((projected + 0.5) / self.cam.size, uv, atol=1e-6)
        self.assertGreater(coverage, 0.95)
        self.assertGreater(radius, 0)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-6)
        main_depths = z[indices.reshape(-1, 3)]
        self.assertEqual(float(np.ptp(main_depths, axis=1).max()), 0)
        margin_depths = self.cam.pixels(margins[0])[1].reshape(-1, 3)
        np.testing.assert_allclose(margin_depths, 10.05, atol=0.001)

    def test_alternate_scenes_keep_default_nodes_and_omit_unreliable_depth(self):
        glb = Glb({})
        base_node = glb.add("nodes", {"name": "Closed scene"})
        glb.doc["scenes"][0]["nodes"] = [base_node]
        result = add_view_surfaces(
            glb, {"photo": self.depth, "bad": SimpleNamespace(anchor_error=12.0)}, {"photo": self.image}, {}
        )
        self.assertEqual(glb.doc["scene"], 0)
        self.assertEqual(glb.doc["scenes"][0]["nodes"], [base_node])
        self.assertEqual(set(result), {"photo"})
        alternative = glb.doc["scenes"][result["photo"]["sceneIndex"]]
        self.assertEqual(alternative["extras"]["surface_source"], "photo")
        self.assertGreater(result["photo"]["marginTriangles"], 0)

    def test_cyclic_annotation_rectifies_local_roof_without_moving_it(self):
        corners = np.array([[12.0, 10], [40, 10], [40, 30], [12, 30]])
        roof = self.cam.plane(corners / self.cam.size, 3)
        observed = corners + np.array([3.0, 2])
        data = {
            "size": self.cam.size.tolist(),
            "buildings": [
                {"id": "home", "roof": np.roll(observed[::-1], 1, axis=0).tolist(), "feet": [None] * 4}
            ],
        }
        marks = SurfaceMarks("photo", self.cam, self.image, data)
        mapping = marks.match(roof, "home")
        uv = mapping.texture(roof, np.array([0.0, 1, 0]), roof)
        np.testing.assert_allclose(uv * self.cam.size, observed, atol=0.001)
        np.testing.assert_allclose(self.cam.pixels(roof)[0], corners, atol=0.001)
        self.assertIsNone(marks.match(roof, "other"))

    def test_shipped_glb_and_report_describe_the_same_distinct_meshes(self):
        report = json.loads((ROOT / "builder/example-report.json").read_text("utf8"))
        with (ROOT / "assets/scene_meshes/VILLAGE_CALIBRATED.glb").open("rb") as stream:
            _, _, _, size, _ = struct.unpack("<4sIII4s", stream.read(20))
            doc = json.loads(stream.read(size))
        detail = report["mesh"]["surfaceDetail"]
        main = doc["scenes"][doc["scene"]]
        self.assertTrue(main["extras"]["global_surface"])
        construction = doc["scenes"][main["extras"]["construction_scene"]]
        names = {doc["nodes"][n]["name"] for n in construction["nodes"]}
        self.assertTrue(all(b["id"] in names for b in report["buildings"]))
        self.assertEqual(
            sum(n.startswith("Inferred ") for n in names),
            sum(row["infillBuildings"] for row in detail["context"].values()),
        )
        self.assertFalse(any("working_radius" in s.get("extras",{}) for s in doc["scenes"]))
        visible = [doc["nodes"][i]["name"] for i in main["nodes"]]
        self.assertIn("Ground",visible)
        self.assertIn("Surrounding fields",visible)
        self.assertEqual(sum(n.startswith("Photo building ") for n in visible),detail["globalSurface"]["closedAnchorBuildings"])
        self.assertGreater(detail["globalSurface"]["triangles"],0)
        projections = main["extras"]["photo_projectors"]
        self.assertEqual(projections["version"], 1)
        self.assertEqual([p["source"] for p in projections["cameras"]], detail["photoProjectors"]["sources"])
        for p in projections["cameras"]:
            self.assertGreater(p["depthScale"], 0)
            image = doc["images"][doc["textures"][p["depthTexture"]]["source"]]
            self.assertEqual(image["mimeType"], "image/png")
            self.assertEqual(len(p["rotation"]), 9)
            self.assertNotEqual(p["texture"], p["depthTexture"])
        self.assertTrue(all("bufferView" in image for image in doc["images"]))

    def test_coverage_accounts_for_the_unsampled_wraparound(self):
        roofs=[np.array([[-1,2,-1],[1,2,1]])]
        def camera(yaw):
            angle=np.deg2rad(yaw)
            return SimpleNamespace(eye=np.array([10*np.sin(angle),8,10*np.cos(angle)]))
        coverage=orbit_coverage({"original":camera(0),"right":camera(90),"left":camera(-36)},roofs)
        self.assertEqual(coverage["largestGapDeg"],234)
        self.assertIn("left",coverage["anglesDeg"])
        self.assertTrue(all(abs(a)<=180 for a in coverage["suggestedAnglesDeg"]))

    def test_reference_depth_encodes_nearest_fixed_surface_and_empty_background(self):
        def triangle(corners, height):
            return self.cam.plane(np.array(corners) / self.cam.size, height)
        far = triangle([[4, 4], [60, 4], [32, 44]], 0.0)
        near = triangle([[21, 15], [43, 15], [32, 33]], 4.0)
        for triangles in ([far, near], [near, far]):
            image, units = visibility_image(self.cam, np.asarray(triangles))
            rgb = np.asarray(image, dtype=np.uint16)
            z = (rgb[:, :, 0] * 256 + rgb[:, :, 1]) * units
            self.assertAlmostEqual(z[20, 32], 6.0, places=2)
            self.assertAlmostEqual(z[10, 32], 10.0, places=2)
            self.assertEqual(z[41, 5], 0.0)
        observed = np.full((48, 64), 6.0, np.float32)
        mapped, _ = visibility_image(self.cam, np.asarray([far, near]), reference_depth=observed)
        mismatch = np.asarray(mapped)[:, :, 2]
        self.assertEqual(mismatch[20, 32], 0)
        self.assertEqual(mismatch[10, 32], 255)


if __name__ == "__main__":
    unittest.main()
