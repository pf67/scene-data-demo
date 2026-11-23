import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "builder"))
from geometry import Camera
from ground import OBSERVED, OCCLUDED, OUTSIDE, UNRESOLVED, collect_observations, ground_surface, terrain_mesh
from ground_edit import edit_ground
from mesh import Glb
from worker import run


def unpack_model(path):
    raw = path.read_bytes()
    magic, version, total, size, kind = struct.unpack_from("<4sIII4s", raw)
    assert (magic, version, total, kind) == (b"glTF", 2, len(raw), b"JSON")
    doc = json.loads(raw[20:20+size])
    binary = raw[28+size:]
    return doc, [binary[b["byteOffset"]:b["byteOffset"]+b["byteLength"]] for b in doc["bufferViews"]]


class GroundTests(unittest.TestCase):
    def test_second_view_recovers_occlusion_and_protects_reliable_pixels(self):
        r = np.array([[1.,0,0], [0,0,1], [0,-1,0]])
        k = np.array([[40.,0,32], [0,40.,32], [0,0,1]])
        cameras = {i: Camera(i, np.array([64,64]), k, r, -r @ [x,10,0]) for i,x in [("a",0),("b",4)]}
        photos = {i: np.full((64,64,3), color, np.uint8) for i,color in [("a",[180,90,30]),("b",[40,170,80])]}
        za = np.full((64,64), np.inf, np.float32)
        za[23:42,23:42] = 4
        buffers = {"a": (za,1), "b": (np.full_like(za,np.inf),1)}
        lo, hi = np.array([-16.,-16.]), np.array([16.,16.])
        masks = {"a": np.ones((64,64),bool)}
        masks["a"][48:,8:24] = False
        args = (lo,hi,129,129,masks)
        _, first, _, _ = collect_observations({"a":cameras["a"]},photos,buffers,[],*args)
        rgb, both, footprints, counts = collect_observations(cameras,photos,buffers,[],*args)
        self.assertEqual(first[64,64], OCCLUDED)
        self.assertEqual(both[64,64], OBSERVED)
        np.testing.assert_array_equal(rgb[64,64], [40,170,80])
        self.assertEqual(first[85,47], UNRESOLVED)
        self.assertEqual(both[0,0], OUTSIDE)
        self.assertGreater((both == OBSERVED).sum(), (first == OBSERVED).sum())
        self.assertGreater(counts["b"], 0)
        settings = {"surface":"fields","roads":[{"points":[[-2,0],[4,0]],"widthM":4}]}
        texture, clearance, _ = ground_surface(rgb,both,footprints,lo,hi,settings)
        repeat, _, _ = ground_surface(rgb,both,footprints,lo,hi,settings)
        np.testing.assert_array_equal(texture, repeat)
        np.testing.assert_array_equal(texture[both == OBSERVED],rgb[both == OBSERVED])
        self.assertGreater(texture[both == OUTSIDE].std(), 2)
        vertices, _, _, indices, span = terrain_mesh(lo,hi,clearance,1)
        self.assertGreater(len(indices), 6)
        self.assertLess(span[0], 0)
        self.assertGreater(span[1], 0)
        along_road = np.abs(vertices[:,2]) < 2
        np.testing.assert_allclose(vertices[along_road,1], -.015)

    def test_ground_only_photo_supports_unobserved_mask_and_no_buildings(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            Image.new("RGB",(128,96),(123,130,90)).save(p/"view.png")
            Image.new("L",(128,96),0).save(p/"mask.png")
            plan = {"version":1,"name":"field","scaleBasis":"estimated","buildings":[],
                    "views":[{"source":"a","fovY":50,"controls":[],"approxPose":{"heightM":10,"pitchDeg":55},
                              "groundMask":"mask"}]}
            report = run(plan,{"a":p/"view.png","mask":p/"mask.png"},p/"scene.glb",p/"ground")
            self.assertEqual(report["mesh"]["ground"]["observed"],0)
            self.assertEqual(report["mesh"]["ground"]["donorPatches"],0)
            self.assertEqual([m["name"] for m in unpack_model(p/"scene.glb")[0]["meshes"]],["Ground"])
            with Image.open(p/"ground/mask.png") as mask:
                self.assertEqual(mask.getextrema(),(255,255))

    def test_edit_changes_only_missing_texels_and_embedded_ground_image(self):
        original = ROOT / "assets/scene_meshes/VILLAGE_CALIBRATED.glb"
        maps = ROOT / "builder/example-ground"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            with Image.open(maps/"texture.png") as source, Image.open(maps/"mask.png") as mask:
                base = np.array(source)
                editable = np.array(mask) > 0
                Image.new("RGB",source.size,(17,209,119)).save(p/"input.png")
            new = p/"new.glb"
            self.assertEqual(edit_ground(original,maps,new,p/"ground",image=p/"input.png")["method"], "imported")
            before, blobs = unpack_model(original)
            after, changed = unpack_model(new)
            for key in ("meshes","materials","accessors","scenes","images","textures"):
                self.assertEqual(before[key],after[key])
            if before["scenes"][0].get("extras", {}).get("photo_projectors", {}).get("cameras"):
                for node in before["nodes"]:
                    if node.get("mesh") == len(before["meshes"]) - 1:
                        node.setdefault("extras", {})["photo_projectors"] = False
            self.assertEqual(before["nodes"], after["nodes"])
            differing = [i for i,(a,b) in enumerate(zip(blobs,changed)) if a != b]
            ground_material = after["meshes"][-1]["primitives"][0]["material"]
            texture = after["materials"][ground_material]["pbrMetallicRoughness"]["baseColorTexture"]["index"]
            ground_view = after["images"][after["textures"][texture]["source"]]["bufferView"]
            self.assertEqual(differing,[ground_view])
            self.assertEqual(changed[differing[0]],(p/"ground/texture.png").read_bytes())
            with Image.open(io.BytesIO(changed[differing[0]])) as img:
                result = np.array(img)
            np.testing.assert_array_equal(result[~editable],base[~editable])
            self.assertTrue(np.any(np.all(result[editable] == [17,209,119],axis=1)))
            for name in ("mask.png","coverage.png"):
                self.assertEqual((maps/name).read_bytes(), (p/"ground"/name).read_bytes())
            Image.new("RGB",(192,64)).save(p/"wrong.png")
            with self.assertRaisesRegex(ValueError,"宽高比"):
                edit_ground(original,maps,p/"invalid.glb",p/"invalid",image=p/"wrong.png")
            self.assertFalse((p/"invalid.glb").exists())
            with self.assertRaisesRegex(ValueError,"独立"):
                edit_ground(original,maps,new,maps,image=p/"input.png")

    def test_projected_ground_edit_preserves_embedded_cameras_and_other_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            maps = p / "maps"
            maps.mkdir()
            ground = Image.new("RGB", (48, 32), (73, 82, 91))
            ground.save(maps / "texture.png")
            for name in ("mask.png", "coverage.png"):
                Image.new("L", ground.size, 255).save(maps / name)
            edit = Image.new("RGB", ground.size, (94, 33, 86))
            edit.save(p / "edit.png")

            glb = Glb({})
            wall = glb.material("Building", [1, 1, 1])
            soil = glb.material("Ground", [1, 1, 1], ground, lossless=True)
            triangle = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1.]])
            parts = [(triangle, np.tile([0, 1, 0], (3, 1)), triangle[:, [0, 2]])]
            glb.mesh("Ground", {wall: parts})  # A user's building may share the terrain node name.
            glb.mesh("Ground", {soil: parts})
            glb.doc["nodes"][-1]["extras"] = {"units": "m"}
            photo = glb.texture(Image.new("RGB", ground.size, (177, 83, 102)))
            depth = glb.texture(Image.new("RGB", ground.size, (1, 6, 0)), lossless=True)
            rig = {"version": 1, "cameras": [{"source":"v", "texture":photo, "depthTexture":depth}]}
            glb.doc["scenes"][0]["extras"]["photo_projectors"] = rig
            glb.write(p / "original.glb")

            self.assertEqual(edit_ground(p / "original.glb", maps, p / "updated.glb",
                                        p / "updated", image=p / "edit.png"), {"method": "imported"})
            before, buffers = unpack_model(p / "original.glb")
            after, edited = unpack_model(p / "updated.glb")
            self.assertEqual(after["scenes"], before["scenes"])
            self.assertEqual(after["nodes"][0], before["nodes"][0])
            self.assertEqual(after["nodes"][1]["extras"], {"units": "m", "photo_projectors": False})
            indices = [after["images"][after["textures"][i]["source"]]["bufferView"] for i in (0, photo, depth)]
            self.assertEqual([i for i,(a,b) in enumerate(zip(buffers,edited)) if a != b], indices[:1])
            self.assertEqual(edited[indices[0]], (p / "updated" / "texture.png").read_bytes())


if __name__ == "__main__":
    unittest.main()
