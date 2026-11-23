import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "builder"))
from facade_material import FacadeMaterials
from local_ground_patch import apply_local_yards


class LocalPatchTests(unittest.TestCase):
    def test_paired_soil_texture_changes_only_verified_unlocked_pixels(self):
        size=np.array([32,32])
        cam=SimpleNamespace(size=size,
            plane=lambda uv,height:np.column_stack((uv[:,0]*32,np.full(len(uv),height),uv[:,1]*32)),
            pixels=lambda xyz:(xyz[:,[0,2]],np.ones(len(xyz))))
        proof=np.zeros((32,32),bool);proof[7:20,8:18]=True
        photo=np.full((32,32,3),200,np.uint8)
        bank=SimpleNamespace(reviewed_yards=[],paired_yards=[SimpleNamespace(paired_ground=proof,cam=cam,image=photo,source="paired")])
        locked=np.zeros_like(proof);locked[12:15,11:14]=True
        old=Image.new("RGB",(32,32))
        out,counts=apply_local_yards(old,{"boundsXZ":[[0,0],[31,31]]},locked,bank,[])
        changed=np.asarray(out)[:,:,0]>0
        np.testing.assert_array_equal(changed,proof&~locked)
        self.assertEqual(counts["paired"],int(changed.sum()))

    def test_local_yard_respects_observed_texels_buildings_and_edges(self):
        photo = np.full((64, 64, 3), [178, 107, 54], np.uint8)
        dest = Image.new("RGB", (64, 64), (10, 12, 16))
        source = np.float32([[3, 3], [60, 3], [60, 60], [3, 60]])
        world = np.float32([[8, 8], [55, 8], [55, 55], [8, 55]])
        roof = np.array([[19, 3, 19], [31, 3, 19], [31, 3, 31], [19, 3, 31]])
        locked = np.zeros((64, 64), bool)
        locked[37:42, 37:42] = True
        bank = SimpleNamespace(reviewed_yards=[("rear", photo, source, world)])
        result, counts = apply_local_yards(dest, {"boundsXZ": [[0, 0], [63, 63]]}, locked, bank, [roof])
        out = np.asarray(result)
        for x, y in [(5, 5), (8, 42), (25, 25), (39, 39)]:
            np.testing.assert_array_equal(out[y, x], [10, 12, 16])
        np.testing.assert_array_equal(out[47, 40], photo[0, 0])
        self.assertEqual(counts["rear"], np.count_nonzero(np.any(out != np.asarray(dest), axis=2)))
        self.assertEqual(locked.sum(), 25)

    def test_reviewed_roof_matches_canonical_corner_order(self):
        bank = FacadeMaterials.__new__(FacadeMaterials)
        image = np.zeros((100, 120, 3), np.uint8)
        quad = np.float32([[10, 20], [90, 15], [85, 80], [15, 85]])
        roof = np.array([[2, 4, 5], [13, 4, 4], [14, 4, 11], [3, 4, 11]])
        bank.reviewed_roofs = {4: (2000, "rear-detail", image, quad)}
        source, rgb, uv = bank.reviewed_roof_uv(4, roof, roof)
        self.assertEqual(source, "rear-detail")
        self.assertIs(rgb, image)
        np.testing.assert_allclose(uv * [120, 100] - .5, quad, atol=1e-4)
        self.assertIsNone(bank.reviewed_roof_uv(4, roof, roof + [50, 0, 0]))

    def test_reviewed_wall_chooses_largest_donor_for_the_same_elevation(self):
        bank = FacadeMaterials.__new__(FacadeMaterials)
        n = np.array([0., 0., 1.])
        small = np.full((16, 12, 3), 40, np.uint8)
        big = np.full((19, 21, 3), 150, np.uint8)
        bank.reviewed_materials = {0: [(n, small, 1, "a", 140), (n, big, 1, "b", 220),
                                       (-n, small, 3, "c", 350)]}
        bank.reviewed_wall_hits, bank.wall_sources = set(), {}
        u = np.full((2, 2), .5)
        np.testing.assert_array_equal(bank.reviewed_wall(0, n, u, u), np.full((2, 2, 3), 150))
        self.assertEqual(bank.wall_sources, {(0, 1): "b"})
