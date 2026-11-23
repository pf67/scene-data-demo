"""Regression cases for rotated image registration and observation priority."""
import json
import base64
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "builder"))
from geometry import Camera
from mesh import bake_face, subdivide, window_surrounds
from orbit import make_layout, register
from worker import prepare, registration_options, scene_faces
sys.path.insert(0, str(ROOT))
from refinement.qma import camera_direction, camera_options, generate_view
from refinement import image_api


class OrbitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(2)
        plan = json.loads((ROOT / "builder/example.json").read_text("utf-8"))
        _, cameras, photos, roofs, _ = prepare(plan, {"village": ROOT / "assets/scene_sources/village_road_near.png"})
        cls.photo = photos["village"]
        cls.faces = scene_faces(plan, roofs)
        cls.layout = make_layout(plan, cameras["village"], cls.photo, roofs, 38)
        cls.plan, cls.camera, cls.roofs = plan, cameras["village"], roofs

    def test_guided_large_angle_checks_are_held_out_and_limit_inferred_surfaces(self):
        spec = next(s for s in self.plan["supplements"] if s["source"] == "village-orbit-image-right90")
        layout = make_layout(self.plan, self.camera, self.photo, self.roofs, spec["angle"])
        options = registration_options(spec)
        with Image.open(ROOT / "builder/sample-views" / (spec["source"] + ".jpg")) as candidate:
            view, q, _ = register(layout, self.photo, candidate, spec["angle"], self.faces, spec["points"], **options)
            self.assertIsNotNone(view)
            self.assertAlmostEqual(q["actualAngle"], 90, delta=2)
            self.assertEqual((q["matches"], q["checkPoints"]), (19, 7))
            self.assertGreaterEqual(q["checkInliers"], 4)
            self.assertLess(q["localSupportPercent"], 35)
            self.assertGreater(q["newlyVisibleFaces"], 0)
            bad = [dict(p, image=[.02, .03]) if p.get("check") else p for p in spec["points"]]
            view, rejected, _ = register(layout, self.photo, candidate, spec["angle"], self.faces, bad, **options)
            self.assertIsNone(view)
            self.assertEqual(rejected["status"], "needs-points")
            self.assertEqual(rejected["actualAngle"], q["actualAngle"])
            self.assertGreater(rejected["checkMedianPx"], q["checkMedianPx"])

    def test_world_landmarks_must_project_to_their_recorded_guide_position(self):
        spec = next(s for s in self.plan["supplements"] if s["source"]=="village-orbit-step-right145")
        layout = make_layout(self.plan,self.camera,self.photo,self.roofs,spec["angle"])
        with Image.open(ROOT/"builder/sample-views"/(spec["source"]+".jpg")) as photo:
            view, quality, _ = register(layout,self.photo,photo,spec["angle"],self.faces,spec["points"],**registration_options(spec))
            self.assertIsNotNone(view)
            self.assertEqual(quality["checkPoints"],7)
            bad = [dict(spec["points"][0],guide=[0.,0.]), *spec["points"][1:]]
            with self.assertRaisesRegex(ValueError,"三维点"):
                register(layout,self.photo,photo,spec["angle"],self.faces,bad,**registration_options(spec))

    def test_image_edits_uses_completed_event_and_scopes_gateway_auth(self):
        photo = Image.new("RGB", (90,60))
        response = Mock(status_code=200, headers={"Content-Type": "text/event-stream"})
        partial = b'data: {"type":"image_edit.partial_image","b64_json":"cGFydA=="}'
        complete = b'data: {"type":"image_edit.completed","b64_json":"ZmluYWw="}'
        only_official_keys = {"OPENAI_API_KEY": "official-fixture", "key": "legacy-fixture",
                              "ORBIT_IMAGE_BASE_URL": "http://localhost:8800/v1", "ORBIT_IMAGE_MODELS": "fixture-v2"}
        with patch.dict(os.environ, only_official_keys, clear=True), patch("refinement.image_api.requests.post") as post:
            post.return_value.__enter__.return_value = response
            response.iter_lines.return_value = [partial, b"", complete, b""]
            output, generation = image_api.generate_view(photo, None, -55, guidance="photo")
            self.assertEqual(output, b"final")
            self.assertEqual(generation["inputRoles"], ["reference"])
            sent = post.call_args.kwargs
            self.assertEqual(sent["headers"], {})
            self.assertFalse(sent["allow_redirects"])
            self.assertEqual([key for key,_ in sent["files"]], ["image"])
            self.assertEqual(sent["data"]["model"], "fixture-v2")
            self.assertIn("55 degrees", sent["data"]["prompt"])
            self.assertIn("left", sent["data"]["prompt"])
            response.iter_lines.return_value = [partial, b"", b"data: [DONE]", b""]
            with self.assertRaises(ValueError):
                image_api.generate_view(photo, None, -55, guidance="photo")

    def test_real_qwen_result_and_unmoved_photo(self):
        candidate = ROOT / "builder/sample-views/village-orbit-right.jpg"
        with Image.open(candidate) as image:
            view, quality, _ = register(self.layout, self.photo, image, 38, self.faces)
        self.assertIsNotNone(view)
        self.assertGreater(quality["inliers"], 40)
        self.assertAlmostEqual(quality["actualAngle"], 38, delta=5)
        self.assertTrue(np.isfinite(quality["camera"]["position"]).all())
        view, quality, _ = register(self.layout, self.photo, self.photo, 38, self.faces)
        self.assertIsNone(view)
        self.assertEqual(quality["status"], "needs-points")

    def test_qma_calls_original_photo_and_uses_measured_pose(self):
        session = Mock()
        session.post.return_value.json.return_value = {"job_id":"fixture-12"}
        state = Mock()
        state.json.return_value = {"status":"succeeded","images":[{"url":"/v1/jobs/fixture-12/images/0"}]}
        output = Mock(content=(ROOT/"builder/sample-views/village-orbit-qma.jpg").read_bytes())
        options = Mock()
        options.json.return_value = {"default_strength":.32, "azimuths":["front-right quarter view"]}
        session.get.side_effect = [options,state,output]
        with patch.dict(os.environ, {"QMA_API_URL":"http://localhost:8190"}), patch("refinement.qma.requests.Session") as factory:
            factory.return_value.__enter__.return_value = session
            data, generation = generate_view(self.photo,45,20261118)
        payload = session.post.call_args.kwargs["json"]
        self.assertNotIn("image_path",payload)
        self.assertEqual(payload["strength"],.32)
        self.assertTrue(session.get.call_args_list[0].args[0].endswith("/v1/options"))
        self.assertEqual(generation["azimuth"],"front-right quarter view")
        png = Image.open(io.BytesIO(base64.b64decode(payload["image_base64"])))
        np.testing.assert_array_equal(np.asarray(png),np.asarray(self.photo))
        self.assertEqual(data,output.content)
        self.assertTrue(session.get.call_args_list[-1].args[0].endswith("/v1/jobs/fixture-12/images/0"))
        report = json.loads((ROOT/"builder/example-report.json").read_text("utf8"))
        item = next(s for s in report["supplements"] if s.get("provider")=="qma")
        self.assertTrue(item["targetMismatch"])
        self.assertGreater(item["newlyVisiblePixels"],256)
        self.assertGreater(report["mesh"]["supplements"]["byView"][item["source"]],0)

    def test_strength_follows_service_default_angle_and_supported_directions(self):
        session = Mock()
        azimuths = ["front-left quarter view", "right side view", "back-left quarter view", "back view"]
        with patch.dict(os.environ, {"QMA_API_URL":"http://localhost:8190"}):
            for base, expected in [(.4,[.4,.65,.80,.90]), (.55,[.55,.80,.95,1])]:
                session.get.return_value.json.return_value = {"default_strength":base,"azimuths":azimuths}
                options = camera_options(session)
                self.assertEqual(options["defaultStrength"],base)
                by_angle = {d["angle"]:d["suggestedStrength"] for d in options["directions"]}
                self.assertEqual(by_angle,dict(zip([-45,90,-135,180],expected)))
            for invalid in [False, None, 1.5, "0.4"]:
                session.get.return_value.json.return_value = {"default_strength":invalid,"azimuths":azimuths}
                with self.assertRaises(ValueError):
                    camera_options(session)
        self.assertEqual(camera_direction(-180)[2],"back view")
        self.assertEqual(camera_direction(-112.5)[2],"back-left quarter view")

    def test_wide_angle_generation_and_manual_strength_reach_qma(self):
        photo = Image.new("RGB",(8,8))
        for manual, expected in [(None,.80),(0,0),(.873,.873),(1,1)]:
            session = Mock()
            session.post.return_value.json.return_value = {"job_id":"strength-test"}
            options = Mock()
            options.json.return_value = {"default_strength":.4,"azimuths":["back-right quarter view"]}
            done = Mock()
            done.json.return_value = {"status":"succeeded","images":[{}]}
            session.get.side_effect = ([options] if manual is None else []) + [done,Mock(content=b"image")]
            with patch.dict(os.environ, {"QMA_API_URL":"http://localhost:8190"}), patch("refinement.qma.requests.Session") as factory:
                factory.return_value.__enter__.return_value = session
                _, generation = generate_view(photo,135,7,manual)
            self.assertEqual(session.post.call_args.kwargs["json"]["strength"],expected)
            self.assertEqual(generation["strength"],expected)
            self.assertEqual(generation["azimuth"],"back-right quarter view")

    def test_manual_correspondences_and_concentrated_input(self):
        layout = self.layout
        points = []
        for row in np.array_split(np.arange(layout.ids.shape[0]), 4):
            for col in np.array_split(np.arange(layout.ids.shape[1]), 4):
                yy, xx = np.nonzero(layout.ids[np.ix_(row,col)] >= 0)
                if len(xx):
                    i = len(xx)//2
                    uv = ((np.array([col[xx[i]],row[yy[i]]]))/(layout.camera.size-1)).tolist()
                    points.append({"guide":uv, "image":uv})
        view, quality, _ = register(layout, self.photo, layout.image, 38, self.faces, points)
        self.assertIsNone(view)
        self.assertEqual(quality["status"], "needs-detail")
        self.assertEqual(quality["method"], "control-points")
        self.assertLess(quality["rmsPx"], .1)
        view, quality, _ = register(layout, self.photo, layout.image, 38, self.faces, [points[0]]*8)
        self.assertIsNone(view)

    def test_generated_colors_only_fill_missing_wall_pixels(self):
        K = np.array([[38.,0,32],[0,38.,32],[0,0,1]])
        cam = Camera("source",np.array([64,64]),K,np.eye(3),np.array([0,-1,8.]))
        face = np.array([[-4.,0,0],[4.,0,0],[4.,3,0],[-4.,3,0]])
        normal = np.array([0.,0,-1.])
        buf = np.full((64,64),np.inf)
        buf[:,:32] = 1.
        image = np.full((64,64,3), [232,27,19],np.uint8)
        args = (face,normal,subdivide(face,0),{"source":cam},{"source":(buf,1)},{"source":image},0)

        class Candidate:
            camera = cam
            def sample(self, world):
                return np.ones(world.shape[:2],bool), np.full(world.shape,[18,95,114],np.uint8)

        baseline, _, _, _, _ = bake_face(*args, [])
        merged, _, _, counts, _ = bake_face(*args, [Candidate()])
        observed = np.all(baseline==[232,27,19],axis=2)
        self.assertGreater(observed.sum(), 500)
        np.testing.assert_array_equal(merged[observed],baseline[observed])
        self.assertGreater(counts[0], 500)
        self.assertEqual(counts[0], np.all(merged==[18,95,114],axis=2).sum())

    def test_trim_requires_entire_opening_in_inferred_region(self):
        color = np.full((128,128,3),180,np.uint8)
        color[40:69,45:72] = 55
        x,y = np.meshgrid(np.linspace(0,5,128),np.linspace(0,4,128))
        world = np.stack([x,y,np.zeros_like(x)],axis=2)
        labels = np.zeros((128,128),np.int16)
        args = (world, np.array([0,0,-1.]), np.array([5.,4.]))
        self.assertEqual(len(window_surrounds(color,labels,*args)),12)
        labels[51,:] = -1
        self.assertEqual(window_surrounds(color,labels,*args),[])
