"""Check the delivered GLB topology, embedded textures and reference calibration."""
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def verify():
    path = ROOT / "assets/scene_meshes/VILLAGE_COMPLETE.glb"
    data = path.read_bytes()
    magic, version, length = struct.unpack_from("<III", data)
    assert magic == 0x46546C67 and version == 2 and length == len(data)
    json_length = struct.unpack_from("<I", data, 12)[0]
    document = json.loads(data[20:20 + json_length])
    binary = memoryview(data)[28 + json_length:]
    dtypes = {5121: "u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}

    def accessor(index):
        info = document["accessors"][index]
        view = document["bufferViews"][info["bufferView"]]
        count = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[info["type"]]
        dtype = np.dtype(dtypes[info["componentType"]])
        offset = view.get("byteOffset", 0) + info.get("byteOffset", 0)
        stride = view.get("byteStride", dtype.itemsize * count)
        return np.ndarray((info["count"], count), dtype, binary, offset,
                          strides=(stride, dtype.itemsize)).copy()

    houses = [n for n in document["nodes"] if n.get("extras", {}).get("closed_building")]
    assert len(houses) == 64, f"Expected 64 completed buildings, got {len(houses)}"
    for node in houses:
        triangles = []
        for primitive in document["meshes"][node["mesh"]]["primitives"]:
            vertices = accessor(primitive["attributes"]["POSITION"])
            indices = accessor(primitive["indices"]).ravel().reshape(-1, 3)
            triangles.append(vertices[indices])
        triangles = np.concatenate(triangles)
        # glTF splits vertices at material/UV seams. Weld positions for topology only.
        _, inverse = np.unique(np.round(triangles.reshape(-1, 3), 5), axis=0,
                               return_inverse=True)
        faces = inverse.reshape(-1, 3)
        edges = Counter(tuple(sorted((int(a), int(b)))) for face in faces
                        for a, b in zip(face, np.roll(face, -1)))
        assert all(n == 2 for n in edges.values()), f"Open/nonmanifold building: {node['name']}"
        volume = np.einsum("ij,ij->i", triangles[:, 0],
                           np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
        assert volume > 0, f"Inverted building: {node['name']}"

    assert all("bufferView" in image for image in document.get("images", [])), "External texture"
    extras = document["scenes"][0]["extras"]
    assert extras["preserve_scale"] and extras["building_count"] == 64
    calibration = json.loads((ROOT / "assets/village_studio/layout.json").read_text("utf-8"))
    frame = np.array(calibration["frame"])
    origin = np.array(calibration["origin"])
    k = np.array(calibration["intrinsics"])
    image_size = np.array(calibration["imageSize"])
    depth_size = np.array(calibration["depthSize"])
    errors = []
    for building in calibration["buildings"]:
        cv = np.array(building["points"]) / calibration["scale"] @ frame + origin
        pixels = (cv[:, :2] / cv[:, 2:] * [k[0, 0], k[1, 1]]
                  + [k[0, 2], k[1, 2]]) * image_size / depth_size
        errors.append(float(np.abs(pixels - building["roof"]).max()))
    assert max(errors) < .01, f"Reference projection error: {max(errors)}px"
    report = {"closed_buildings": len(houses), "topology": "all watertight, outward-facing",
              "embedded_textures": len(document.get("images", [])),
              "reference_roof_projection_max_error_px": max(errors),
              "glb_megabytes": round(len(data) / 1e6, 2),
              "note": "Topology/projection verification; inferred geometry is not a measured survey."}
    (ROOT / "assets/village_studio/geometry-check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    verify()
