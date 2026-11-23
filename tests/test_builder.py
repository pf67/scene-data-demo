import copy
import json
import struct
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "builder"))
sys.path.insert(0, str(ROOT / "tools/scene_pipeline"))
from calibration_example import HEIGHT, SIZE, create
from worker import run


class BuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.plan, cls.files = create(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_surveyed_two_view_height_and_withheld_checks(self):
        plan = copy.deepcopy(self.plan)
        # Simulate a hand click. The same noisy annotation never acts as a check.
        rng = np.random.default_rng(83)
        for v, o in zip(plan["views"], plan["buildings"][0]["observations"]):
            for point in v["controls"]:
                if not point["check"]:
                    point["uv"] = (point["uv"] + rng.normal(0, .55, 2) / SIZE).tolist()
            o["roof"] = (o["roof"] + rng.normal(0, .55, (4, 2)) / SIZE).tolist()
        result = run(plan, self.files)
        house = result["buildings"][0]
        self.assertEqual(house["heightMethod"], "multiview")
        self.assertAlmostEqual(house["heightM"], HEIGHT, delta=.09)
        self.assertAlmostEqual(house["areaM2"], 48, delta=.7)
        self.assertGreater(house["minRayAngleDeg"], 20)
        self.assertTrue(all(v["checkPoints"] == 1 and v["checkRmsM"] < .12 for v in result["views"]))

    def test_one_view_vertical_and_explicit_fallback(self):
        p = copy.deepcopy(self.plan)
        p["views"].pop()
        house = p["buildings"][0]
        house["observations"].pop()
        row = run(p, self.files)["buildings"][0]
        self.assertEqual(row["heightMethod"], "vertical")
        self.assertAlmostEqual(row["heightM"], HEIGHT, places=3)
        del house["observations"][0]["base"]
        with self.assertRaisesRegex(ValueError, "檐高"):
            run(p, self.files)
        house.update(heightM=4, heightSource="estimated")
        self.assertEqual(run(p, self.files)["buildings"][0]["heightM"], 4)

    def test_near_duplicate_camera_wrong_matches_and_degenerate_controls(self):
        p = copy.deepcopy(self.plan)
        p["views"][1]["controls"] = p["views"][0]["controls"]
        obs = p["buildings"][0]["observations"]
        obs[1]["roof"] = obs[0]["roof"]
        with self.assertRaisesRegex(ValueError, "交角"):
            run(p, self.files)
        p = copy.deepcopy(self.plan)
        p["buildings"][0]["observations"][1]["roof"].reverse()
        with self.assertRaisesRegex(ValueError, "重投影|核对|檐高"):
            run(p, self.files)
        for c in p["views"][0]["controls"]:
            c["xz"][1] = 0
        with self.assertRaisesRegex(ValueError, "共线"):
            run(p, self.files)

    def test_holdout_is_never_used_for_camera_fit(self):
        p = copy.deepcopy(self.plan)
        p["views"][0]["controls"][-1]["xz"][0] += 2
        result = run(p, self.files)
        self.assertEqual(result["views"][0]["fitRmsPx"], 0)
        self.assertAlmostEqual(result["views"][0]["checkRmsM"], 2, places=3)
        self.assertAlmostEqual(result["buildings"][0]["heightM"], HEIGHT, places=3)

    def test_quick_single_photo_has_no_fabricated_quality_or_scale(self):
        p = copy.deepcopy(self.plan)
        p["views"] = [p["views"][0]]
        p["views"][0].update(controls=[], approxPose={"heightM": 14, "pitchDeg": 25})
        p["buildings"][0].update(heightM=5, observations=p["buildings"][0]["observations"][:1])
        del p["buildings"][0]["observations"][0]["base"]
        with self.assertRaisesRegex(ValueError, "估算尺度"):
            run(p, self.files)
        p["scaleBasis"] = "estimated"
        result = run(p, self.files, Path(self.tmp.name) / "rough.glb")
        self.assertIsNone(result["views"][0]["fitRmsPx"])
        self.assertEqual(result["buildings"][0]["heightMethod"], "estimated")

    def test_export_embeds_images_and_closed_outward_building(self):
        path = Path(self.tmp.name) / "fixture.glb"
        p = copy.deepcopy(self.plan)
        p["buildings"][0]["ridgeRiseM"] = 1.25
        run(p, self.files, path)
        data = path.read_bytes()
        magic, version, size, length, kind = struct.unpack_from("<4sIII4s", data)
        self.assertEqual((magic, version, size, kind), (b"glTF", 2, len(data), b"JSON"))
        doc = json.loads(data[20:20 + length])
        start = 28 + length
        self.assertEqual(len(doc["nodes"]), 2)
        self.assertGreaterEqual(len(doc["images"]), 3)
        self.assertTrue(all("bufferView" in im for im in doc["images"]))
        edges, volume = Counter(), 0.
        for prim in doc["meshes"][0]["primitives"]:
            a = doc["accessors"][prim["attributes"]["POSITION"]]
            view = doc["bufferViews"][a["bufferView"]]
            triangles = np.frombuffer(data, "<f4", a["count"] * 3, start + view["byteOffset"]).reshape(-1, 3, 3)
            for tri in triangles:
                volume += np.dot(tri[0], np.cross(tri[1], tri[2])) / 6
                corners = [tuple(pt.round(4)) for pt in tri]
                for j in range(3):
                    edges[tuple(sorted((corners[j], corners[(j + 1) % 3])))] += 1
        self.assertEqual(set(edges.values()), {2})
        self.assertAlmostEqual(volume, 48 * (HEIGHT + 1.25 / 2), delta=.01)


if __name__ == "__main__":
    unittest.main()
