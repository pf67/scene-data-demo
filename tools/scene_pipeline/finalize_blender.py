"""Export a manually reviewed Blender scene with the editor's world/camera contract.

Run: blender --factory-startup -b reviewed.blend --python finalize_blender.py --
     --output output/SCENE_CODE.glb --label "Scene label"
This exporter does not invent backsides or repair a raw point cloud.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import bmesh
from mathutils import Vector


parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--reference-url', default='')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
scene = bpy.context.scene
mesh_objects = [o for o in scene.objects if o.type == 'MESH' and not o.hide_render]
if not mesh_objects:
    raise SystemExit('No mesh geometry to export')

def to_three(v):
    return [float(v.x), float(v.z), float(-v.y)]

scene['preserve_scale'] = True
scene['scene_kind'] = 'completed_scene'
scene['reference_image_url'] = args.reference_url
camera = scene.camera
if camera:
    eye = camera.matrix_world.translation
    forward = camera.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    scene['camera_position'] = to_three(eye)
    scene['camera_target'] = to_three(eye + forward * 30)
    scene['camera_fov'] = math.degrees(camera.data.angle_y)
else:
    scene['camera_position'] = [45, 45, 55]
    scene['camera_target'] = [0, 0, 0]
    scene['camera_fov'] = 45

topology = []
for obj in mesh_objects:
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    topology.append({'name': obj.name, 'vertices': len(mesh.verts), 'faces': len(mesh.faces),
                     'boundary_edges': sum(edge.is_boundary for edge in mesh.edges),
                     'requires_closed': bool(obj.get('closed_building', False))})
    mesh.free()
invalid = [item['name'] for item in topology if item['requires_closed'] and item['boundary_edges']]
if invalid:
    raise SystemExit(f'Buildings marked as closed still have boundary edges: {invalid}')
scene['building_count'] = sum(bool(o.get('closed_building')) for o in mesh_objects)
bpy.ops.object.select_all(action='DESELECT')
for obj in mesh_objects:
    obj.select_set(True)
args.output.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.gltf(filepath=str(args.output), export_format='GLB', use_selection=True,
                         export_extras=True, export_animations=False)
args.output.with_suffix('.json').write_text(json.dumps({'code': args.output.stem, 'label': args.label,
    'kind': 'completed_scene', 'preserve_scale': True}, ensure_ascii=False, indent=2), encoding='utf-8')
args.output.with_suffix('.topology.json').write_text(json.dumps(topology, ensure_ascii=False, indent=2), encoding='utf-8')
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix('.blend')))
print(f'Exported {args.output}; check all eight viewing directions before promotion.')
