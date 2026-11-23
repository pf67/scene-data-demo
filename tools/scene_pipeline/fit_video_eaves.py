"""Fit existing building footprints to the same visible eaves in independent video shots."""

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from video_bake import base_pixels, color_error, project, remap, tile_grid, transform
from video_materials import load_mesh, scene_faces
from video_surfaces import primitives, signature
from video_tracks import digest
from worker import prepare


def read(path):
    return json.loads(path.read_text("utf-8-sig"))


def frame(video, i):
    stream = cv2.VideoCapture(str(video))
    try:
        stream.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = stream.read()
        if not ok:
            raise ValueError("无法读取标注帧")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        stream.release()


def check_eave(name, drawn, old, survey, tracks, source):
    donor = f"{survey['donor']['clip']}:{survey['donor']['frame']}"
    before = []
    after = []
    clips = set()
    for tag, labels in survey["checks"].items():
        if name not in labels or tag == donor:
            continue
        key, i = tag.split(":")
        data = tracks[key]
        match = data["surfaces"][name].get(i)
        if match is None:
            continue
        target = np.array(labels[name])
        h = np.array(match["h"])
        update = lambda q: transform(
            h, q * np.array(data["firstScale"]) + data["firstShift"]
        )
        prev = np.linalg.norm(update(old) - target, axis=1)
        actual = np.linalg.norm(update(drawn) - target, axis=1)
        if (
            np.median(actual) > 4.5
            or max(actual) > 8
            or np.median(actual) >= 0.65 * np.median(prev)
        ):
            continue
        before.extend(prev)
        after.extend(actual)
        clips.add(key)
    if len(clips) < 2:
        raise ValueError(f"{name} 的檐角未能在两个留出视频中复核")
    return dict(
        heldOutClips=sorted(clips),
        corners=len(after),
        previousPx=round(float(np.median(before)), 3),
        correctedPx=round(float(np.median(after)), 3),
        referenceCorners=drawn.tolist(),
    )


def roof_tile(grid, oldxz, true_ref, name, photos, survey, original):
    world, inside, old = grid
    xz = world[:, :, [0, 2]]

    def rect(photo, quad):
        h = cv2.getPerspectiveTransform(np.float32(oldxz), np.float32(quad))
        return remap(photo, transform(h, xz))

    donor = f"{survey['donor']['clip']}:{survey['donor']['frame']}"
    new = rect(photos[donor], survey["checks"][donor][name])
    first = rect(original, true_ref)
    held = [
        rect(photos[tag], q[name])
        for tag, q in survey["checks"].items()
        if name in q and tag != donor
    ]
    err = np.max([cv2.blur(color_error(new, h), (5, 5)) for h in held], axis=0)
    err_first = np.max([cv2.blur(color_error(first, h), (5, 5)) for h in held], axis=0)
    trusted = inside & (err <= 22)
    fallback = inside & (err_first <= 22)
    coverage = float(trusted.sum() / inside.sum())
    if coverage < 0.84:
        return None
    a = cv2.GaussianBlur(np.uint8(trusted) * 255, (5, 5), 1.0) / 255.0
    a *= trusted
    blended = np.where(fallback[:, :, None], first, old)
    result = np.uint8(np.rint(a[:, :, None] * new + (1 - a[:, :, None]) * blended))
    old_error = float(np.mean([color_error(old, h)[inside].mean() for h in held]))
    new_error = float(np.mean([color_error(result, h)[inside].mean() for h in held]))
    if new_error >= 0.91 * old_error:
        return None
    return Image.fromarray(result), dict(
        mode="surveyedRoof",
        source=survey["donor"]["clip"],
        frame=survey["donor"]["frame"],
        coverage=round(coverage, 3),
        beforeMAE=round(old_error, 3),
        afterMAE=round(new_error, 3),
        heldOutTexels=int(inside.sum()) * len(held),
        heldOutFrames=[
            t for t in survey["checks"] if t != donor and name in survey["checks"][t]
        ],
    )


def checked_wall(
    name,
    label,
    oldxz,
    change,
    faces,
    parts,
    glb,
    cache,
    base,
    original,
    photos,
    survey,
    tracks,
):
    edge = oldxz[label["edge"]]
    vertical = [f for f in faces if f["node"] == name and abs(f["normal"][1]) < 0.12]

    def distance(f):
        return np.max(np.abs((edge - f["corners"][0, [0, 2]]) @ f["normal"][[0, 2]]))

    if not vertical:
        return None
    face = min(vertical, key=distance)
    if distance(face) > 0.08:
        return None
    part = parts[face["primitive"]]
    grid = tile_grid(face, part, base_pixels(glb, part[0], cache))
    if grid is None:
        return None
    world, inside, previous = grid
    fitted = world.copy()
    fitted[:, :, [0, 2]] = transform(
        np.array(change["footprintHomography"]), world[:, :, [0, 2]]
    )
    fitted[:, :, 1] *= change.get("scale", [1, 1, 1])[1]
    xy = project(base, fitted)
    mask = np.zeros(original.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.int32(label["visible"])], 1)
    valid = inside & cv2.remap(mask, xy, None, cv2.INTER_NEAREST).astype(bool)
    donor = f"{survey['donor']['clip']}:{survey['donor']['frame']}"
    samples = {}
    for tag, photo in photos.items():
        clip, number = tag.split(":")
        track = tracks[clip]
        observed = track["surfaces"].get(name + "-wall", {}).get(number)
        if observed is None:
            continue
        pts = transform(
            np.array(observed["h"]),
            xy * np.array(track["firstScale"]) + track["firstShift"],
        )
        valid &= (pts > 1).all(-1) & (pts < np.array(photo.shape[1::-1]) - 2).all(-1)
        samples[tag] = remap(photo, pts)
    if donor not in samples:
        return None
    new = samples.pop(donor)
    held = {
        tag: im
        for tag, im in samples.items()
        if tag.split(":")[0] != survey["donor"]["clip"]
    }
    if len({tag.split(":")[0] for tag in held}) < 2:
        return None
    for check in [remap(original, xy), *held.values()]:
        valid &= cv2.blur(color_error(new, check), (5, 5)) < 22
    valid = cv2.morphologyEx(np.uint8(valid), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = np.minimum(1.0, cv2.distanceTransform(valid, cv2.DIST_L2, 5) / 7.0)
    strong = alpha > 0.4
    coverage = valid.sum() / inside.sum()
    if coverage < 0.08 or strong.sum() < 100:
        return None
    result = np.uint8(
        np.rint(previous * (1 - alpha[:, :, None]) + new * alpha[:, :, None])
    )
    old_error = np.mean(
        [color_error(previous, im)[strong].mean() for im in held.values()]
    )
    new_error = np.mean(
        [color_error(result, im)[strong].mean() for im in held.values()]
    )
    if old_error - new_error < max(2.0, 0.15 * old_error):
        return None
    stats = dict(
        mode="surveyedWall",
        kind="wall",
        source=survey["donor"]["clip"],
        frame=survey["donor"]["frame"],
        coverage=round(float(coverage), 3),
        beforeMAE=round(float(old_error), 3),
        afterMAE=round(float(new_error), 3),
        heldOutFrames=list(held),
        heldOutTexels=int(strong.sum()) * len(held),
        primitive=face["primitive"],
        triangles=face["indices"].tolist(),
    )
    return face, Image.fromarray(result), stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "plan",
        "reference",
        "geometry",
        "survey",
        "tracks",
        "materials",
        "out",
    ):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--videos", nargs="+", type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(4)
    pl = read(a.plan)
    source = pl["views"][0]["source"]
    _, cams, ims, _, _ = prepare(pl, {source: a.reference})
    base = cams[source]
    glb = load_mesh(a.geometry)
    faces = scene_faces(glb)
    survey = read(a.survey)
    videos = {digest(v)[:12]: v for v in a.videos}
    photos = {}
    for tag in survey["checks"]:
        k, i = tag.split(":")
        photos[tag] = frame(videos[k], int(i))
    tracks = {p.stem: read(p) for p in a.tracks.glob("*.json")}
    spec = read(a.materials / "manifest.json")
    atlas = Image.open(a.materials / "atlas.jpg").convert("RGB")
    nodes = {
        glb.doc["nodes"][i]["name"]: glb.doc["nodes"][i]
        for i in glb.doc["scenes"][0]["nodes"]
    }
    roofmap = {}
    parts = {}
    oldprints = {}
    corrections = {}
    heights = {}
    evidence = {}
    for name in sorted({f["node"] for f in faces}):
        if name.startswith("Courtyard walls "):
            continue
        nodeparts = primitives(glb, nodes[name])
        xyz = np.vstack([q[1].reshape(-1, 3) for q in nodeparts])
        oldprints[name] = cv2.convexHull(np.float32(xyz[:, [0, 2]]))
        if name not in survey["roofs"]:
            continue
        choices = [q for q in faces if q["node"] == name and q["normal"][1] > 0.997]
        if not choices:
            raise ValueError(name + " 的平屋面缺失")
        f = max(choices, key=lambda q: q["area"])
        parts[name] = nodeparts
        uv = project(base, f["corners"])
        idx = cv2.approxPolyDP(uv, 1.2, True).reshape(-1, 2)
        if len(idx) != 4:
            raise ValueError(name + " 的檐线不是四边形")
        drawn = np.array(survey["roofs"][name], float)
        order = min(
            itertools.permutations(range(4)),
            key=lambda perm: np.linalg.norm(idx[list(perm)] - drawn),
        )
        corners = f["corners"][
            [np.argmin(np.linalg.norm(uv - pt, axis=1)) for pt in idx[list(order)]]
        ]
        oldxz = corners[:, [0, 2]]
        height = float(corners[:, 1].mean())
        evidence[name] = check_eave(
            name, drawn, project(base, corners), survey, tracks, source
        )
        roofmap[name] = (f, oldxz, drawn)
        heights[name] = height
    shared = []
    for g in survey.get("sharedEaves", []):
        a1, a2 = g["buildings"]
        i1, i2 = g["indices"]
        if not all(n in roofmap for n in (a1, a2)):
            raise ValueError("公共檐线的建筑缺失")
        for tag, q in survey["checks"].items():
            if (
                np.max(
                    np.linalg.norm(np.array(q[a1])[i1] - np.array(q[a2])[i2], axis=1)
                )
                > 2
            ):
                raise ValueError("两片屋面在视频中不共享檐线")
        old_a = roofmap[a1][1][i1]
        old_b = roofmap[a2][1][i2]
        delta = heights[a1] - heights[a2]
        separation = np.sqrt(np.sum((old_a - old_b) ** 2, axis=1) + delta**2)
        common = (heights[a1] + heights[a2]) / 2
        shared.append(
            dict(
                **g,
                estimatedHeightM=round(common, 3),
                previousEndpointGapM=separation.round(3).tolist(),
            )
        )
        heights[a1] = heights[a2] = common
    newprints = {
        name: base.plane(q / base.size, heights[name])[:, [0, 2]]
        for name, (_, _, q) in roofmap.items()
    }
    for name, (f, oldxz, drawn) in roofmap.items():
        poly = cv2.convexHull(np.float32(newprints[name]))
        # Compare every neighbour at its final position, including both halves of a shared edge.
        for other, old in oldprints.items():
            if other == name:
                continue
            new = (
                cv2.convexHull(np.float32(newprints[other]))
                if other in newprints
                else old
            )
            before = max(0, cv2.intersectConvexConvex(oldprints[name], old)[0])
            after = max(0, cv2.intersectConvexConvex(poly, new)[0])
            if after - before > max(0.65, 0.015 * cv2.contourArea(poly)):
                raise ValueError(f"{name} 与 {other} 的新轮廓重叠")
        h = cv2.getPerspectiveTransform(np.float32(oldxz), np.float32(newprints[name]))
        corrections[name] = dict(
            signature=signature(parts[name]),
            faces=[],
            footprintHomography=h.tolist(),
            eaveChecks=evidence[name],
        )
        original = float(f["corners"][:, 1].mean())
        if abs(original - heights[name]) > 0.02:
            corrections[name].update(
                scale=[1.0, heights[name] / original, 1.0],
                translation=[0.0, 0.0, 0.0],
                oldHeight=round(original, 3),
                newHeight=round(heights[name], 3),
            )
    cache = {}
    tiles = []
    a.out.mkdir(exist_ok=True, parents=True)
    for name, (f, oldxz, drawn) in roofmap.items():
        part = parts[name][f["primitive"]]
        grid = tile_grid(f, part, base_pixels(glb, part[0], cache))
        selected = roof_tile(
            grid, oldxz, drawn, name, photos, survey, np.asarray(ims[source])
        )
        if selected:
            tile, stats = selected
            target = dict(
                kind="roof",
                primitive=f["primitive"],
                triangles=f["indices"].tolist(),
                **stats,
            )
            corrections[name]["faces"].append(target)
            tiles.append((f, tile, target))
            tile.save(a.out / (name + "-measured.jpg"), quality=95)
            Image.fromarray(grid[2]).save(a.out / (name + "-previous.jpg"), quality=95)
        if name in survey.get("walls", {}):
            wall = checked_wall(
                name,
                survey["walls"][name],
                oldxz,
                corrections[name],
                faces,
                parts[name],
                glb,
                cache,
                base,
                np.asarray(ims[source]),
                photos,
                survey,
                tracks,
            )
            if wall:
                corrections[name]["faces"].append(wall[2])
                tiles.append(wall)
    spec["nodes"].update(corrections)
    width = atlas.width
    oldheight = atlas.height
    row = 0
    x = 0
    y = oldheight
    pad = 3
    places = []
    for _, tile, _ in tiles:
        w, h = tile.size
        if x + w + 2 * pad > width:
            x = 0
            y += row
            row = 0
        places.append((x + pad, y + pad))
        x += w + 2 * pad
        row = max(row, h + 2 * pad)
    height = y + row
    result = Image.new("RGB", (width, height))
    result.paste(atlas)
    for name, record in spec["nodes"].items():
        if name in corrections:
            continue
        for f in record["faces"]:
            f["basis"][1] = (np.array(f["basis"][1]) * oldheight / height).tolist()
    for (f, tile, entry), (x, y) in zip(tiles, places):
        result.paste(
            Image.fromarray(
                np.pad(np.asarray(tile), ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            ),
            (x - pad, y - pad),
        )
        coef = (np.array(tile.size) - 1) / (f["hi"] - f["lo"]) / result.size
        entry["basis"] = np.column_stack(
            (
                f["basis"] * coef[:, None],
                (np.array([x, y]) + 0.5) / result.size - f["lo"] * coef,
            )
        ).tolist()
    result.save(a.out / "atlas.jpg", quality=97, subsampling=0)
    spec.update(atlasSha256=digest(a.out / "atlas.jpg"), sharedEaves=shared)
    (a.out / "manifest.json").write_bytes(
        (json.dumps(spec, ensure_ascii=False, indent=2) + "\n").encode("utf8")
    )
    print(
        "footprints",
        len(corrections),
        "surveyed surfaces",
        len(tiles),
        "connected eaves",
        len(shared),
    )


if __name__ == "__main__":
    main()
