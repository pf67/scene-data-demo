"""Verify exported VILLAGE_BLENDER.glb topology, embedded textures, metadata, and audit gate."""

import json
import struct
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/blender_village")]

import layout_solver as ls  # noqa: E402


class BlenderVillageTests(unittest.TestCase):
    def test_village_blender_glb_and_audit_contract(self):
        glb_path = ROOT / "assets/scene_meshes/VILLAGE_BLENDER.glb"
        meta_path = ROOT / "assets/scene_meshes/VILLAGE_BLENDER.json"
        topo_path = ROOT / "assets/scene_meshes/VILLAGE_BLENDER.topology.json"
        spec_path = ROOT / "artifacts/blender_village/village_spec.json"

        self.assertTrue(glb_path.is_file(), f"Missing {glb_path}")
        self.assertTrue(meta_path.is_file(), f"Missing {meta_path}")
        self.assertTrue(topo_path.is_file(), f"Missing {topo_path}")
        self.assertTrue(spec_path.is_file(), f"Missing {spec_path}")

        data = glb_path.read_bytes()
        size_mb = len(data) / (1024 * 1024)
        self.assertLess(size_mb, 45.0, f"VILLAGE_BLENDER.glb too large: {size_mb:.2f} MB")

        magic, version, length = struct.unpack_from("<III", data)
        self.assertEqual(magic, 0x46546C67)
        self.assertEqual(version, 2)
        self.assertEqual(length, len(data))

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
            return np.ndarray(
                (info["count"], count),
                dtype,
                binary,
                offset,
                strides=(stride, dtype.itemsize),
            ).copy()

        # Guide point cloud must NOT leak into exported GLB
        node_names = [n.get("name", "") for n in doc.get("nodes", [])]
        self.assertNotIn("COLMAP_Sparse_Points", node_names)

        houses = [n for n in doc["nodes"] if n.get("extras", {}).get("closed_building")]
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        expected_count = len(spec["buildings"])
        self.assertGreaterEqual(len(houses), 170)
        self.assertEqual(len(houses), expected_count)
        self.assertIn("Road_Asphalt_Carriageway", node_names)

        # Independent Shapely layout audit (zero road/apron intrusion, zero overlap, zero skew, high village density)
        layout_audit = ls.audit_layout_geometry(
            spec["buildings"],
            spec["road_centerline_xz"],
            spec["road_polygon_xz"],
            spec["road_apron_polygon_xz"],
        )
        self.assertEqual(layout_audit["verdict"], "pass", layout_audit)
        self.assertEqual(layout_audit["road_intrusion_count"], 0)
        self.assertEqual(layout_audit["road_intrusion_area_m2"], 0.0)
        self.assertEqual(layout_audit["apron_intrusion_count"], 0)
        self.assertEqual(layout_audit["apron_intrusion_area_m2"], 0.0)
        self.assertEqual(layout_audit["building_overlap_count"], 0)
        self.assertEqual(layout_audit["max_building_overlap_m2"], 0.0)
        self.assertLessEqual(layout_audit["max_corner_skew_deg"], 1.5)
        self.assertGreaterEqual(layout_audit["footprint_density_pct"], 40.0)

        for node in houses:
            triangles = []
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                verts = accessor(prim["attributes"]["POSITION"])
                idx = accessor(prim["indices"]).ravel().reshape(-1, 3)
                triangles.append(verts[idx])
            tris = np.concatenate(triangles)
            _, inv = np.unique(np.round(tris.reshape(-1, 3), 5), axis=0, return_inverse=True)
            faces = inv.reshape(-1, 3)
            edges = Counter(
                tuple(sorted((int(a), int(b))))
                for face in faces
                for a, b in zip(face, np.roll(face, -1))
            )
            self.assertTrue(
                all(cnt == 2 for cnt in edges.values()),
                f"Non-watertight building in GLB: {node.get('name')}",
            )
            vol = float(np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6.0)
            self.assertGreater(vol, 0.0, f"Inverted building normals: {node.get('name')}")

        for sheet_name in ("village_ground", "Road_Asphalt_Carriageway", "Road_Concrete_Aprons"):
            sheet_node = next((n for n in doc["nodes"] if n.get("name") == sheet_name), None)
            self.assertIsNotNone(sheet_node, f"Missing terrain/road sheet node: {sheet_name}")
            for prim in doc["meshes"][sheet_node["mesh"]]["primitives"]:
                normals = accessor(prim["attributes"]["NORMAL"])
                self.assertGreater(
                    float(normals[:, 1].min()),
                    0.99,
                    f"Downward/non-upward normals on {sheet_name}: min ny={normals[:, 1].min():.3f}",
                )

        self.assertTrue(any(m.get("alphaMode") == "MASK" for m in doc.get("materials", [])), "Missing alpha-MASK foliage material")
        self.assertTrue(all("bufferView" in img for img in doc.get("images", [])), "External image URI leaked into GLB")
        self.assertGreaterEqual(len(doc.get("images", [])), 8)

        extras = doc["scenes"][0]["extras"]
        self.assertTrue(extras.get("preserve_scale"))
        self.assertEqual(extras.get("building_count"), expected_count)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["code"], "VILLAGE_BLENDER")
        self.assertEqual(meta["kind"], "photo_completed_village")
        self.assertTrue(meta["preserve_scale"])

        topo = json.loads(topo_path.read_text(encoding="utf-8"))
        self.assertEqual(topo["verdict"], "pass")
        self.assertEqual(topo["closed_buildings"], expected_count)
        self.assertEqual(topo["failure_count"], 0)

        orbit_mp4 = ROOT / "artifacts/blender_village/orbit_guide/orbit_guide.mp4"
        orbit_cams = ROOT / "artifacts/blender_village/orbit_guide/orbit_cameras.json"
        self.assertTrue(orbit_mp4.is_file(), f"Missing {orbit_mp4}")
        self.assertTrue(orbit_cams.is_file(), f"Missing {orbit_cams}")


if __name__ == "__main__":
    unittest.main()
