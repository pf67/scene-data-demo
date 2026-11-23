"""Render a GLB with the reference camera and afternoon light, without editing the source."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=48)
    parser.add_argument('--sun-energy', type=float, default=3.6)
    parser.add_argument('--world-strength', type=float, default=0.65)
    parser.add_argument('--exposure', type=float, default=0.35)
    parser.add_argument('--preview', action='store_true')
    return parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])


def main():
    args = arguments()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.spec.read_text(encoding='utf-8'))
    reference = next(camera for camera in spec['cameras'] if camera['source_clip'] == 'ref')
    intrinsics = reference['blender_intrinsics']
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(source))
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.preview_samples = 16
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 4
    scene.cycles.glossy_bounces = 4
    scene.render.resolution_x = 960 if args.preview else 1920
    scene.render.resolution_y = 640 if args.preview else 1280
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = intrinsics['pixel_aspect_x']
    scene.render.pixel_aspect_y = intrinsics['pixel_aspect_y']
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.exposure = args.exposure

    selected_device = 'CPU'
    preferences = bpy.context.preferences.addons['cycles'].preferences
    for kind in ('OPTIX', 'CUDA'):
        try:
            preferences.compute_device_type = kind
            preferences.get_devices()
            devices = [device for device in preferences.devices if device.type == kind]
            if devices:
                for device in preferences.devices:
                    device.use = device in devices
                scene.cycles.device = 'GPU'
                selected_device = ', '.join(f'{device.type}: {device.name}' for device in devices)
                break
        except (TypeError, RuntimeError):
            continue

    world = bpy.data.worlds.new('Village overcast sky fill')
    world.use_nodes = True
    background = next(node for node in world.node_tree.nodes if node.type == 'BACKGROUND')
    background.inputs['Color'].default_value = (0.70, 0.77, 0.84, 1)
    background.inputs['Strength'].default_value = args.world_strength
    scene.world = world

    sunlight = bpy.data.lights.new('Afternoon sun', 'SUN')
    sunlight.energy = args.sun_energy
    sunlight.color = (1.0, 0.91, 0.78)
    sunlight.angle = math.radians(3.0)
    sun = bpy.data.objects.new('Afternoon sun', sunlight)
    scene.collection.objects.link(sun)
    # Original camera calibration uses Three.js Y-up; Blender is Z-up.
    direction = Vector((0.775, -0.484, 0.407)).normalized()
    sun.rotation_euler = (-direction).to_track_quat('-Z', 'Y').to_euler()

    camera_data = bpy.data.cameras.new('Reference aerial camera')
    camera = bpy.data.objects.new('Reference aerial camera', camera_data)
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix(reference['blender_mat4'])
    camera_data.sensor_fit = intrinsics['sensor_fit']
    camera_data.sensor_width = intrinsics['sensor_width']
    camera_data.lens = intrinsics['lens']
    camera_data.shift_x = intrinsics['shift_x']
    camera_data.shift_y = intrinsics['shift_y']
    camera_data.clip_end = 2000
    scene.camera = camera

    if not args.preview:
        overview_data = camera_data.copy()
        overview_data.name = 'Oblique aerial camera'
        overview = bpy.data.objects.new('Oblique aerial camera', overview_data)
        scene.collection.objects.link(overview)
        overview.location = Vector((72, -60, 70))
        overview.rotation_euler = (Vector((0, 18, 0)) - overview.location).to_track_quat('-Z', 'Y').to_euler()
        overview_data.shift_x = overview_data.shift_y = 0
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str(output / 'village-afternoon.blend'))
    image = output / ('draft-reference.png' if args.preview else 'village-afternoon-reference.png')
    scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)

    if not args.preview:
        # A second viewpoint makes it possible to inspect parts outside the reference camera.
        scene.camera = overview
        scene.render.pixel_aspect_x = 1
        scene.render.filepath = str(output / 'village-afternoon-oblique.png')
        bpy.ops.render.render(write_still=True)
        receipt = {
            'source': str(source), 'source_sha256': before,
            'source_unchanged': before == hashlib.sha256(source.read_bytes()).hexdigest(),
            'device': selected_device, 'engine': scene.render.engine,
            'samples': args.samples, 'resolution': [scene.render.resolution_x, scene.render.resolution_y],
            'camera': 'reference camera calibration plus a second oblique view',
            'sun_energy': sunlight.energy, 'sun_direction_blender': list(direction),
            'world_strength': background.inputs['Strength'].default_value,
            'view_transform': scene.view_settings.view_transform, 'exposure': args.exposure,
        }
        (output / 'render-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        print('RENDER_RECEIPT', json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
