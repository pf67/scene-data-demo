"""Dependency-free GLB delivery check (does not certify unseen reconstruction accuracy)."""
import argparse
import json
import struct
from pathlib import Path


def inspect(path):
    raw = path.read_bytes()
    if len(raw) < 28 or struct.unpack_from('<III', raw) != (0x46546C67, 2, len(raw)):
        raise ValueError('Not a valid GLB 2 container')
    length, kind = struct.unpack_from('<II', raw, 12)
    if kind != 0x4E4F534A:
        raise ValueError('Missing JSON chunk')
    doc = json.loads(raw[20:20 + length])
    external = [item['uri'] for category in ['images', 'buffers'] for item in doc.get(category, [])
                if item.get('uri') and not item['uri'].startswith('data:')]
    primitives = [p for m in doc.get('meshes', []) for p in m['primitives']]
    surfaces = sum(p.get('mode', 4) in (4, 5, 6) for p in primitives)
    points = sum(p.get('mode', 4) == 0 for p in primitives)
    materials = doc.get('materials', [])
    report = {'file': path.name, 'bytes': len(raw), 'nodes': len(doc.get('nodes', [])),
              'surface_primitives': surfaces, 'point_primitives': points,
              'external_dependencies': external, 'materials': len(materials),
              'unlit_materials': sum('KHR_materials_unlit' in m.get('extensions', {}) for m in materials),
              'scene_extras': doc.get('scenes', [{}])[doc.get('scene', 0)].get('extras', {})}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if external:
        raise SystemExit('FAIL: external textures/buffers must be embedded or packaged and mapped explicitly')
    if not surfaces:
        raise SystemExit('FAIL: point-only GLB is not a relightable placement surface')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('glb', type=Path)
    inspect(parser.parse_args().glb)
