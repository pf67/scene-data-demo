import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/scene_pipeline"), str(ROOT / "builder")]
from video_materials import load_mesh
from video_surfaces import add_evidence, primitives
from worker import prepare


class VillageVideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads((ROOT / "builder/example.json").read_bytes())
        cls.key = cls.plan["videoEvidence"]
        folder = ROOT / "builder/video-surfaces" / cls.key
        cls.spec = json.loads((folder / "manifest.json").read_bytes())
        cls.folder = folder

    def test_reimported_or_different_photo_does_not_receive_sample_geometry(self):
        other = Image.new("RGB", (8, 8), "white")
        result = add_evidence(None, self.plan, other)
        self.assertEqual(result["buildings"], 0)
        self.assertEqual(result["skippedBuildings"], len(self.spec["nodes"]))
        self.assertTrue(result["reason"])
        self.assertIsNone(add_evidence(None, {}, other))

    def test_shared_eave_is_closed_in_the_exported_mesh(self):
        manifest = (self.folder / "manifest.json").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(hashlib.sha256(manifest).hexdigest(), self.key)
        self.assertEqual(
            hashlib.sha256((self.folder / "atlas.jpg").read_bytes()).hexdigest(),
            self.spec["atlasSha256"],
        )
        glb = load_mesh(ROOT / "assets/scene_meshes/VILLAGE_CALIBRATED.glb")
        scene = glb.doc["scenes"][glb.doc["scene"]]
        nodes = {
            glb.doc["nodes"][i]["name"]: glb.doc["nodes"][i] for i in scene["nodes"]
        }
        source = self.plan["views"][0]["source"]
        camera = prepare(
            self.plan, {source: ROOT / "assets/scene_sources/village_road_near.png"}
        )[1][source]
        for joint in self.spec["sharedEaves"]:
            endpoints = []
            for name, indices in zip(joint["buildings"], joint["indices"]):
                check = self.spec["nodes"][name]["eaveChecks"]
                self.assertGreaterEqual(len(check["heldOutClips"]), 2)
                self.assertLess(check["correctedPx"], check["previousPx"])
                observed = np.asarray(check["referenceCorners"])[indices]
                expected = camera.plane(
                    observed / camera.size, joint["estimatedHeightM"]
                )
                vertices = np.concatenate(
                    [v.reshape(-1, 3) for _, v, _, _ in primitives(glb, nodes[name])]
                )
                nearest = vertices[
                    np.linalg.norm(vertices[:, None] - expected, axis=2).argmin(axis=0)
                ]
                np.testing.assert_allclose(nearest, expected, atol=0.002)
                endpoints.append(nearest)
            np.testing.assert_allclose(endpoints[0], endpoints[1], atol=0.001)
        status = scene["extras"]["video_surfaces"]
        self.assertEqual(status["joinedEaves"], len(self.spec["sharedEaves"]))
        self.assertEqual(status["skippedBuildings"], 0)
        self.assertIn("inspectionCamera", status)
