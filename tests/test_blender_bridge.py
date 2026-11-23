import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/blender_bridge")]

import blender_bridge_addon as bba  # noqa: E402
from bridge_client import BridgeClient, find_blender_binary  # noqa: E402


class BlenderBridgeTests(unittest.TestCase):
    def test_coordinate_roundtrip_and_colmap_pose_det(self):
        v = [12.5, 6.25, -43.75]
        b = bba.world_to_blender(v)
        self.assertEqual(b, (12.5, 43.75, 6.25))
        back = bba.blender_to_three(b)
        self.assertEqual(back, v)

        quat = [-0.2042059515, 0.9784986657, -0.0283764140, -0.0059219713]
        t_cw = [-4.12246186, -0.41750191, 95.65342435]
        eye_b, rot_b2w, mat4 = bba.colmap_pose_to_blender_matrix(quat, t_cw)
        r = np.array(rot_b2w)
        self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=6)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-6)
        self.assertEqual(len(mat4), 4)
        self.assertEqual(len(eye_b), 3)

    def test_point_in_polygon_2d(self):
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertTrue(bba.point_in_polygon_2d(5.0, 5.0, square))
        self.assertFalse(bba.point_in_polygon_2d(15.0, 5.0, square))

    def test_blender_headless_doctor_projection_and_audit(self):
        exe = find_blender_binary()
        self.assertTrue(exe.is_file())

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            blend_path = tmp_dir / "scene.blend"
            client = BridgeClient(mode="headless", blend_path=blend_path, base_dir=tmp_dir)

            doc = client.call("doctor")
            self.assertTrue(doc["ok"])
            self.assertEqual(doc["verdict"], "pass")
            self.assertIn("4.", doc["blender_version"])

            # Verify exact COLMAP -> Blender projection with anisotropic fx!=fy and off-center (cx, cy)
            quat = [-0.2042059515, 0.9784986657, -0.0283764140, -0.0059219713]
            t_cw = [-4.12246186, -0.41750191, 95.65342435]
            fx, fy, cx, cy, w, h = 1325.94084821, 1398.85872396, 845.2, 470.8, 1536, 1024
            r_cw = np.array(bba.quat_wxyz_to_matrix(quat))
            pt_w = np.array([-21.78738703, 7.0, 37.32015772])
            xc = r_cw @ pt_w + np.array(t_cw)
            u_col = float(xc[0] / xc[2] * fx + cx)
            v_col = float(xc[1] / xc[2] * fy + cy)

            _, _, mat4 = bba.colmap_pose_to_blender_matrix(quat, t_cw)
            intr = bba.colmap_intrinsics_to_blender_camera(fx, fy, cx, cy, w, h)
            code = f"""
from bpy_extras.object_utils import world_to_camera_view
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
intr = {intr!r}
scene.render.resolution_x = intr['resolution_x']
scene.render.resolution_y = intr['resolution_y']
scene.render.resolution_percentage = 100
scene.render.pixel_aspect_x = intr['pixel_aspect_x']
scene.render.pixel_aspect_y = intr['pixel_aspect_y']
cam_data = bpy.data.cameras.new('C1')
cam_data.type = 'PERSP'
cam_data.sensor_fit = intr['sensor_fit']
cam_data.sensor_width = intr['sensor_width']
cam_data.lens = intr['lens']
cam_data.shift_x = intr['shift_x']
cam_data.shift_y = intr['shift_y']
cam_obj = bpy.data.objects.new('C1', cam_data)
scene.collection.objects.link(cam_obj)
cam_obj.matrix_world = Matrix({mat4!r})
bpy.context.view_layer.update()
ndc = world_to_camera_view(scene, cam_obj, Vector({bba.world_to_blender(pt_w)!r}))
RESULT = [ndc.x * {w}, (1.0 - ndc.y) * {h}]
"""
            proj = client.call("exec_py", {"code": code}, save_on_ok=False)
            self.assertTrue(proj["ok"], proj)
            u_bl, v_bl = proj["result"]
            self.assertLess(abs(u_bl - u_col), 0.02)
            self.assertLess(abs(v_bl - v_col), 0.02)

            # Build closed valid building cube -> snapshot -> audit PASS
            build_closed = """
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add(size=4.0, location=(0.0, 0.0, 2.0))
ob = bpy.context.object
ob.name = 'house_01'
ob['closed_building'] = True
"""
            r1 = client.call("exec_py", {"code": build_closed}, save_on_ok=True)
            self.assertTrue(r1["ok"])
            snap = client.call("snapshot", {"label": "valid"})
            self.assertTrue(snap["ok"])
            pass_audit = client.call("audit", {"expected_closed_buildings": 1})
            self.assertEqual(pass_audit["verdict"], "pass", pass_audit)

            # Delete one face -> open boundary -> audit MUST fail -> rollback restores pass
            break_mesh = """
ob = bpy.data.objects['house_01']
bm = bmesh.new()
bm.from_mesh(ob.data)
bm.faces.ensure_lookup_table()
bm.faces.remove(bm.faces[0])
bm.to_mesh(ob.data)
bm.free()
"""
            client.call("exec_py", {"code": break_mesh}, save_on_ok=True)
            fail_audit = client.call("audit", {"expected_closed_buildings": 1})
            self.assertEqual(fail_audit["verdict"], "fail")
            self.assertTrue(any("boundary_edges" in msg for msg in fail_audit["failures"]))

            rb = client.call("rollback", {"snapshot_path": snap["snapshot_path"]})
            self.assertTrue(rb["ok"])
            after_rb = client.call("audit", {"expected_closed_buildings": 1})
            self.assertEqual(after_rb["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
