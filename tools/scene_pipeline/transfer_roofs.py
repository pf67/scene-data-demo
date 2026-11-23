"""Verify traced eaves against interior photo tracks before rectifying a complete roof."""

import argparse
from pathlib import Path
import cv2
import numpy as np
from register_step import mutual_features, read, save, track_corners
from step_view import report_camera
from worker import prepare, read_photo


def validate(a, b, src, dst, images):
    def inside(poly, pt):
        return cv2.pointPolygonTest(poly, tuple(map(float, pt)), True) > 1.0

    pairs = [(u, v) for u, v in zip(a, b) if inside(src, u) and inside(dst, v)]
    transform = cv2.getPerspectiveTransform(src, dst)
    residuals = []
    if pairs:
        x, y = np.float32(pairs).transpose(1, 0, 2)
        residuals = np.linalg.norm(
            cv2.perspectiveTransform(x[None], transform)[0] - y, axis=1
        )
    # Control corners themselves never enter either check. Compare detail in
    # separate interior tiles as additional evidence on roofs with few keypoints.
    basis = np.float32([[0, 0], [159, 0], [159, 111], [0, 111]])
    crops = [
        cv2.warpPerspective(
            cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY),
            cv2.getPerspectiveTransform(q, basis),
            (160, 112),
        )
        for im, q in zip(images, [src, dst])
    ]
    scores = []
    for j in range(3):
        for i in range(4):
            y, x = 8 + 32 * j, 8 + 36 * i
            first = crops[0][y : y + 26, x : x + 28]
            search = crops[1][y - 3 : y + 29, x - 3 : x + 31]
            if min(first.std(), search.std()) < 5:
                continue
            scores.append(
                float(cv2.matchTemplate(search, first, cv2.TM_CCOEFF_NORMED).max())
            )
    good = int(np.sum(np.asarray(residuals) < 7.0))
    tiles = int(np.sum(np.asarray(scores) > 0.68))
    median = float(np.median(residuals)) if len(residuals) else None
    valid = (
        (good >= 3 and median <= 7.0)
        or (tiles >= 5 and good >= 1)
        or (tiles >= 8 and np.median(scores) > 0.73)
    )
    return bool(valid), dict(
        interiorTracks=len(pairs),
        within7px=good,
        medianPx=round(median, 2) if median is not None else None,
        matchingTiles=tiles,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in (
        "plan",
        "report",
        "directory",
        "parent-image",
        "parent-materials",
        "polygons",
    ):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--source", nargs=2, action="append", required=True)
    p.add_argument("--id", help="Source identifier for this view in the scene plan")
    args = p.parse_args()
    d = args.directory
    images = [read_photo(args.parent_image), read_photo(d / "view.jpg")]
    library = read(args.parent_materials)
    from surface_depth import image_digest

    if image_digest(args.parent_image) != library["sha256"]:
        raise ValueError("父图与屋面标注不匹配")
    old = {
        row["building"]: np.float32(
            np.array(row["quad"]) * np.array(images[0].size) / library["size"]
        )
        for row in library["roofs"]
    }
    q = read(d / "track-quality.json")
    cam = report_camera("", q["camera"])
    meta = read(d / "generation.json")
    old_camera = report_camera(
        "",
        next(
            v for v in read(args.report)["supplements"] if v["source"] == meta["parent"]
        )["camera"],
    )
    plan = read(args.plan)
    _, _, _, roofs, _ = prepare(plan, dict(args.source))
    world = {b["id"]: r for b, r in zip(plan["buildings"], roofs)}
    aa, bb, _ = mutual_features(*images)
    ca, cb, _ = track_corners(*images, aa, bb)
    a = np.vstack([aa, ca])
    b = np.vstack([bb, cb])
    accepted = []
    audit = {}
    traced = []
    for label, poly in read(args.polygons).items():
        name = f"house_{int(label):02}" if label.isdigit() else label
        if name not in old:
            continue
        expected = cam.pixels(world[name])[0]
        norm = lambda m: (m - m.mean(0)) / max(1, np.linalg.norm(np.ptp(m, axis=0)))
        eaves = np.float32(poly)
        options = [
            np.roll(eaves[::direction], i, axis=0).copy()
            for direction in (1, -1)
            for i in range(4)
        ]
        dst = min(options, key=lambda v: np.linalg.norm(norm(v) - norm(expected)))
        if (dst < 1).any() or (dst >= np.array(images[1].size) - 2).any():
            continue
        valid, res = validate(a, b, old[name], dst, images)
        lifted = old_camera.plane(
            old[name] / old_camera.size, float(world[name][:, 1].mean())
        )
        predicted = cam.pixels(lifted)[0]
        local_shift = dst[0] - predicted[0]
        errors = np.linalg.norm(predicted[1:] + local_shift - dst[1:], axis=1)
        res.update(
            heldOutEaveErrorsPx=errors.round(2).tolist(),
            shiftPx=round(float(np.linalg.norm(local_shift)), 2),
        )
        roof_span = np.linalg.norm(np.ptp(dst, axis=0))
        geometry_ok = (
            np.median(errors) < 9
            and max(errors) < 12
            and np.linalg.norm(local_shift) < max(7, 0.25 * roof_span)
        )
        valid |= geometry_ok and (res["matchingTiles"] >= 2 or res["within7px"] >= 1)
        audit[name] = dict(accepted=bool(valid), **res)
        traced.append(dict(building=name, quad=dst.astype(int).tolist()))
        if valid:
            accepted.append(dict(building=name, quad=dst.astype(int).tolist()))
    save(d / "roof-audit.json", audit)
    spec = dict(
        source=args.id or f"{plan['views'][0]['source']}-orbit-step-{d.name}",
        reference=plan["views"][0]["source"],
        provider="openai",
        model=meta["model"],
        sha256=meta["sha256"],
        size=list(images[1].size),
        rawSize=list(images[1].size),
        requestedAngle=meta["targetYaw"],
        roofs=accepted,
        walls=[],
        registration=dict(
            accepted=True,
            actualAngle=q["actualAngle"],
            checkMedianPx=q["checkMedianPx"],
        ),
    )
    save(d / "materials.json", spec)
    save(
        d / "traced-roofs.json",
        {**spec, "roofs": traced, "use": "correspondence-review-only"},
    )
    print(
        d.name,
        len(accepted),
        [(k, v["within7px"], v["matchingTiles"]) for k, v in audit.items()],
    )


if __name__ == "__main__":
    main()
