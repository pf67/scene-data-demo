"""Bake a world-aligned ground atlas, its evidence mask and optional inferred relief."""
import math

import cv2
import numpy as np
from PIL import Image


OUTSIDE, OCCLUDED, OBSERVED, UNRESOLVED = 0, 1, 2, 3
COVERAGE_COLORS = [(87, 101, 139), (212, 156, 89), (113, 178, 142), (151, 119, 146)]


def ground_anchors(plan, cameras, roofs):
    points = [p["xz"] for view in plan["views"] for p in view["controls"]]
    for roof in roofs:
        points.extend(roof[:, [0, 2]])
    if not points:
        camera = next(iter(cameras.values()))
        # No surveyed anchors or buildings. Bound the estimated site using
        # downward rays only, so an image touching the horizon stays finite.
        u, v = np.meshgrid(np.linspace(.05, .95, 9), np.linspace(.10, .96, 10))
        rays = camera.rays(np.column_stack((u.ravel(), v.ravel())))
        rays = rays[rays[:, 1] < -.06]
        if len(rays) < 3:
            raise ValueError("画面内可见地面不足，请调整相机或提供地面控制点")
        hits = camera.eye - rays * (camera.eye[1]/rays[:, 1:2])
        points = hits[:, [0, 2]]
    return np.asarray(points)


def ground_bounds(points, margin_percent):
    lo, hi = points.min(0), points.max(0)
    padding = np.maximum(.05 - (hi-lo), 0) / 2
    lo, hi = lo-padding, hi+padding
    margin = (hi - lo) * margin_percent / 100
    return lo - margin, hi + margin


def collect_observations(cameras, images, buffers, roofs, lo, hi, width, height, ground_masks=None):
    """Project the same world texel into every calibrated camera, depth-test, then blend."""
    shape = (height, width)
    rgb = np.zeros((*shape, 3), np.float32)
    weights = np.zeros(shape, np.float32)
    framed = np.zeros(shape, bool)
    clear_ray = np.zeros(shape, bool)
    texel_step = (hi-lo) / [width-1, height-1]
    footprint = np.zeros(shape, np.uint8)
    for roof in roofs:
        vertices = (roof[:, [0, 2]] - lo) / (hi - lo) * [width - 1, height - 1]
        cv2.fillConvexPoly(footprint, np.int32(np.round(vertices)), 1)
    x = np.linspace(lo[0], hi[0], width)
    z = np.linspace(lo[1], hi[1], height)
    per_view = {}
    for source, cam in cameras.items():
        src = images[source]
        support = (ground_masks or {}).get(source)
        zbuf, scale = buffers[source]
        conservative = cv2.erode(np.nan_to_num(zbuf, posinf=1e12), np.ones((3, 3), np.uint8))
        count = 0
        for start in range(0, height, 96):
            sl = slice(start, min(start + 96, height))
            xx, zz = np.meshgrid(x, z[sl])
            world = np.column_stack((xx.ravel(), np.zeros(xx.size), zz.ravel()))
            with np.errstate(divide="ignore", invalid="ignore"):
                uv, depth = cam.pixels(world)
            inside = (depth > .02) & np.isfinite(uv).all(1) & (uv >= 0).all(1) & (uv < cam.size - 1).all(1)
            uv[~inside] = 0
            framed[sl] |= inside.reshape(xx.shape)
            ij = np.clip((uv * scale).astype(int), 0, np.array(zbuf.shape[::-1]) - 1)
            visible = inside & (depth <= conservative[ij[:, 1], ij[:, 0]] + .03) & (footprint[sl].ravel() == 0)
            clear_ray[sl] |= visible.reshape(xx.shape)
            if support is not None:
                px = uv.astype(int)
                visible &= support[px[:, 1], px[:, 0]]
            # Reject severely undersampled far terrain using the shorter
            # singular axis of the local ground-to-image projection.
            safe_z = np.maximum(depth, .02)
            delta_uv = uv - cam.k[:2, 2]
            jac = (np.diag(cam.k[:2, :2])[:, None] * cam.r[:2, [0,2]] - delta_uv[:, :, None] * cam.r[2, [0,2]]) * texel_step
            quality = (cam.k[0, 0] * cam.k[1, 1] * cam.eye[1]) / safe_z**3
            shorter = quality * np.prod(texel_step) * safe_z / np.maximum(np.sqrt((jac**2).sum((1,2))), 1e-8)
            height_ratio = cam.eye[1] / np.maximum(np.linalg.norm(world - cam.eye, axis=1), .02)
            visible &= (shorter >= .28) | (height_ratio >= .23)
            count += int(visible.sum())
            edge = np.minimum(uv, cam.size - 1 - uv).min(axis=1)
            weight = (visible * quality * np.clip((edge + 1) / 16, .05, 1)).reshape(xx.shape).astype(np.float32)
            mapped = cv2.remap(src, uv.astype(np.float32).reshape(*xx.shape, 2), None, cv2.INTER_LINEAR)
            rgb[sl] += mapped * weight[:, :, None]
            weights[sl] += weight
        per_view[source] = count
    known = weights > 0
    rgb[known] /= weights[known, None]
    evidence = np.where(known, OBSERVED, np.where(framed, OCCLUDED, OUTSIDE)).astype(np.uint8)
    evidence[~known & clear_ray] = UNRESOLVED
    return np.uint8(np.clip(np.rint(rgb), 0, 255)), evidence, footprint, per_view


def neighborhood_color(rgb, known):
    # Normalized area averages filter out holes at every scale. Coarse levels
    # supply the low-frequency color where there are no local observations.
    mask = known.astype(np.float32)
    colors = rgb.astype(np.float32) * mask[:, :, None]
    levels = [(colors, mask)]
    while min(mask.shape) > 2 and max(mask.shape) > 12:
        colors, mask = cv2.pyrDown(colors), cv2.pyrDown(mask)
        levels.append((colors, mask))
    average = rgb[known].mean(axis=0) if known.any() else np.array([145, 139, 119])
    guide = np.broadcast_to(average, colors.shape).astype(np.float32).copy()
    for colors, mask in reversed(levels):
        guide = cv2.resize(guide, mask.shape[::-1], interpolation=cv2.INTER_LINEAR)
        local = colors / np.maximum(mask[:, :, None], 1e-6)
        strength = np.clip(mask * 3, 0, 1)[:, :, None]
        guide = guide * (1 - strength) + local * strength
    return guide


def variation(rng, shape, cell):
    ny, nx = (max(3, math.ceil(length / cell) + 1) for length in shape)
    return cv2.resize(rng.normal(0, 1, (ny, nx)).astype(np.float32), shape[::-1], interpolation=cv2.INTER_CUBIC)


def field_pattern(x, z, rng, shape, unit, angle, materials=None):
    # Deterministic parcels in ground coordinates, with distinct fallow and
    # planted strips. The scale follows the patch size for estimated scenes.
    cs, sn = math.cos(angle), math.sin(angle)
    u, v = (cs*x + sn*z)/unit, (-sn*x + cs*z)/unit
    v = v + 1.15*np.sin(u/26) + 1.8*np.sin(v/19)
    band = np.floor(v / 23).astype(int)
    offset = np.sin(band * 23.8) * 19
    length = 33 + 10*np.cos(band*7.1)
    u = u + offset + 3*np.sin(u/24)
    parcel = np.floor(u / length).astype(int)
    family = np.mod(np.abs(np.sin(band * 15.71 + parcel * 78.23 + 2.7) * 4703).astype(int), 7)
    palette = np.array([[125,131,90], [145,143,96], [139,123,91], [143,141,110],
                        [113,124,87], [156,139,105], [128,137,98]], np.float32)
    furrow = np.where(family % 3 == 0, u + .04*v, v + .07*u)
    grain = 3.6 * np.cos(furrow * 4.5) + 4.4 * variation(rng, shape, 24)
    color = palette[family] + grain[:, :, None]
    if materials:
        color = materials.farmland((cs*x+sn*z),(-sn*x+cs*z),family,color)
    edge = np.minimum(np.minimum(v % 23, 23 - v % 23), np.minimum(u % length, length - u % length))
    track = np.clip((.64 - edge) / .55, 0, .82)[:, :, None]
    return color*(1-track) + (np.array([153,142,111]) + grain[:, :, None]) * track


def exemplar_detail(rgb, known, guide, rng):
    """Transfer fine texture from clean, fully visible ground patches only."""
    height, width = known.shape
    tile = min(32, min(height, width) // 3)
    flat = np.uint8(known) * 255
    eligible = cv2.erode(flat, np.ones((tile + 2, tile + 2), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.nonzero(eligible)
    result = np.zeros_like(guide)
    if not len(ys):
        return result, 0
    sample = rng.choice(len(ys), min(350, len(ys)), replace=False)
    patches = np.array([rgb[y-tile//2:y-tile//2+tile, x-tile//2:x-tile//2+tile] for y, x in zip(ys[sample], xs[sample])], dtype=np.float32)
    means = patches.mean(axis=(1, 2))
    # Desaturate sharp structure (cars, isolated objects) at the donor
    # selection stage. Remove the DC component to prevent copying silhouettes.
    contrast = patches.std(axis=(1, 2)).mean(1)
    keep = contrast <= max(12, np.percentile(contrast, 72))
    patches, means = patches[keep], means[keep]
    details = np.clip(patches - means[:, None, None, :], -22, 22)
    step = tile // 2
    window = (.06 + .94 * np.hanning(tile)).astype(np.float32)
    window = window[:, None] * window[None, :]
    total = np.zeros(known.shape, np.float32)
    for y in range(-step, height, step):
        for x in range(-step, width, step):
            sy, sx = slice(max(y,0), min(y+tile,height)), slice(max(x,0), min(x+tile,width))
            if known[sy, sx].all():
                continue
            py, px = slice(sy.start-y,sy.stop-y), slice(sx.start-x,sx.stop-x)
            target = guide[min(y+step,height-1), min(x+step,width-1)]
            distance = ((means - target) ** 2).sum(1)
            k = min(5, len(distance))
            choice = rng.choice(np.argpartition(distance, k-1)[:k])
            weight = window[py, px]
            result[sy, sx] += details[choice, py, px] * weight[:, :, None]
            total[sy, sx] += weight
    return result / np.maximum(total[:, :, None], .001), len(details)


def road_distance(x, z, guide):
    points = np.array(guide["points"])
    best = np.full(x.shape, np.inf, np.float32)
    along = np.zeros(x.shape, np.float32)
    previous = 0
    for i, (a, b) in enumerate(zip(points, points[1:])):
        d = b - a
        length2 = d @ d
        if length2 < 1e-8:
            raise ValueError("道路引导线中有重合的连续点")
        t = ((x-a[0])*d[0] + (z-a[1])*d[1]) / length2
        extend = guide.get("extendEnds", True)
        if i or not extend:
            t = np.maximum(t, 0)
        if i < len(points)-2 or not extend:
            t = np.minimum(t, 1)
        distance = np.hypot(x-a[0]-t*d[0], z-a[1]-t*d[1])
        length = math.sqrt(length2)
        along = np.where(distance < best, previous+t*length, along)
        best = np.minimum(best, distance)
        previous += length
    return best, along


def reviewed_streets(settings, marks, cameras):
    """Rectify a traced street centerline at ground level; sample its actual pavement and verge."""
    roads = [dict(road) for road in settings.get("roads", [])]
    for key, mark in marks.items():
        if key not in cameras:
            continue
        for street in mark.data.get("streets", []):
            index = street["road"]
            if not 0 <= index < len(roads):
                continue
            center = mark.cam.plane(np.asarray(street["center"]) / mark.size, 0)
            road = roads[index]
            road["points"] = center[:, [0, 2]].tolist()
            for name in ("asphalt", "shoulder"):
                points = np.rint(mark.pixels(street[name])).astype(int)
                h, w = mark.image.shape[:2]
                samples = [mark.image[max(0,y-2):min(h,y+3), max(0,x-2):min(w,x+3)].reshape(-1,3)
                           for x,y in points if 0 <= x < w and 0 <= y < h]
                if samples:
                    road[name] = np.median(np.concatenate(samples), axis=0)
    return {**settings, "roads": roads}


def ground_surface(rgb, evidence, footprint, lo, hi, settings, materials=None):
    shape = evidence.shape
    height, width = shape
    rng = np.random.default_rng(612)
    x, z = np.meshgrid(np.linspace(lo[0], hi[0], width, dtype=np.float32), np.linspace(lo[1], hi[1], height, dtype=np.float32))
    step = max((hi-lo) / [width-1, height-1])
    style = settings.get("surface", "local")
    roads = settings.get("roads", [])
    # A ground ray can still hit a parked vehicle or a tree shadow at a misregistered
    # curb. Exclude incompatible samples before marking them as original evidence.
    streets = [(road, *road_distance(x, z, road)) for road in roads]
    for road, dist, _ in streets:
        if "asphalt" in road:
            on_lane = dist < road["widthM"] / 2
            difference = np.max(np.abs(rgb - road["asphalt"]), axis=2)
            evidence[(evidence == OBSERVED) & on_lane & (difference > 39)] = UNRESOLVED
    known = evidence == OBSERVED
    guide = neighborhood_color(rgb, known)
    verges = [r["shoulder"] for r in roads if "shoulder" in r]
    if verges:
        soil = np.median(verges, axis=0) + (3.5*variation(rng, shape, 28))[:, :, None]
        guide = .18*guide + .82*soil
    direction = np.subtract(roads[0]["points"][-1], roads[0]["points"][0]) if roads else [1, .28]
    angle = math.atan2(direction[1], direction[0])
    if materials:
        u = math.cos(angle)*x+math.sin(angle)*z
        v = -math.sin(angle)*x+math.cos(angle)*z
        distance_to_wall = cv2.distanceTransform((footprint==0).astype(np.uint8),cv2.DIST_L2,5)*step
        guide = materials.settlement(u,v,distance_to_wall,streets,guide)
    if style == "fields":
        fields = field_pattern(x, z, rng, shape, max(.3, min(hi-lo)/260), angle, materials)
        # Occlusion silhouettes can extend far behind a roof. Use distance to
        # the settlement in world space, not the shape of the camera's shadow.
        street_block = footprint.copy()
        footprint_pixels = cv2.findNonZero(footprint)
        if footprint_pixels is not None:
            cv2.fillConvexPoly(street_block, cv2.convexHull(footprint_pixels), 1)
        settlement = known | street_block.astype(bool)
        distance = cv2.distanceTransform((~settlement).astype(np.uint8), cv2.DIST_L2, 5)
        mix = np.clip((distance*step - 1.8) / max(3., min(hi-lo)*.035), 0, 1)
        mix *= mix * (3 - 2*mix)
        guide = guide * (1-mix[:, :, None]) + fields * mix[:, :, None]
    else:
        guide += (3.8 * variation(rng, shape, 36))[:, :, None]
    protected = known | footprint.astype(bool)
    for road, dist, along in streets:
        half = road["widthM"] / 2
        asphalt = known & (dist < half * .72)
        samples = rgb[asphalt]
        tone = np.array([89, 92, 90])
        if "asphalt" in road:
            tone = road["asphalt"]
        elif len(samples) >= 20:
            luminance = samples.mean(1)
            tone = np.median(samples[luminance <= np.percentile(luminance, 48)], axis=0)
        verge = road.get("shoulder", tone + 19)
        shoulder = np.clip((half + max(.35,half*.27) - dist) / max(.15,step), 0, 1)
        road_alpha = np.clip((half - dist) / max(.12,step), 0, 1)
        if materials:
            shoulder_texture = materials.sample("yard",along,dist)
            if shoulder_texture is not None:
                verge = shoulder_texture
        guide = guide*(1-shoulder[:, :, None]) + verge * shoulder[:, :, None]
        asphalt = tone + (1.8*variation(rng,shape,18))[:, :, None]
        guide = guide*(1-road_alpha[:, :, None]) + asphalt * road_alpha[:, :, None]
        if road["widthM"] >= 4:
            stripe = np.clip((.075+step*.44-dist)/step, 0, .56)
            if "asphalt" not in road:
                stripe *= along % 7 < 2.8
            guide = guide*(1-stripe[:, :, None]) + np.array([151,137,102])*stripe[:, :, None]
        protected |= dist <= half + max(1, step*3)
    detail, donors = exemplar_detail(rgb, known, guide, rng)
    grain = rng.normal(0, 2.4, shape).astype(np.float32)
    strength = .19 if verges else ((.70 - .42*mix[:, :, None]) if style == "fields" else .70)
    synthesized = np.uint8(np.clip(guide + strength*detail + grain[:, :, None], 0, 255))
    synthesized[known] = rgb[known]
    clearance = cv2.distanceTransform((~protected).astype(np.uint8), cv2.DIST_L2, 5)
    return synthesized, clearance, donors


def terrain_mesh(lo, hi, clearance, amplitude):
    h, w = clearance.shape
    if amplitude:
        nx, nz = np.maximum(3, np.rint((hi-lo) / max(hi-lo) * 112).astype(int) + 1)
    else:
        nx = nz = 2
    u, v = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,nz))
    x, z = lo[0]+u*(hi-lo)[0], lo[1]+v*(hi-lo)[1]
    heights = np.zeros_like(x)
    if amplitude:
        pixels = np.rint(u*(w-1)).astype(int)
        rows = np.rint(v*(h-1)).astype(int)
        diagonal = math.hypot((w-1)/(nx-1), (h-1)/(nz-1))
        fade = np.clip((clearance[rows, pixels] - diagonal - 2) / max(12, diagonal*3), 0, 1)
        noise = .58*np.sin(12*u+2*v)*np.cos(9*v-1.8*u) + .27*np.sin(24*v+5*u) + .15*np.cos(29*u-9*v)
        heights = amplitude * fade**2 * (3-2*fade) * noise
    dz, dx = np.gradient(heights, (hi-lo)[1]/(nz-1), (hi-lo)[0]/(nx-1))
    normals = np.stack([-dx, np.ones_like(dx), -dz], -1).reshape(-1,3)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    corners = np.arange(nx*nz).reshape(nz,nx)
    a, b, c, d = corners[:-1,:-1], corners[1:,:-1], corners[1:,1:], corners[:-1,1:]
    indices = np.stack([a,b,c,a,c,d], -1).ravel().astype("<u2")
    vertices = np.column_stack((x.ravel(), heights.ravel()-.015, z.ravel()))
    uv = np.column_stack(((.5+u.ravel()*(w-1))/w, (.5+v.ravel()*(h-1))/h))
    return vertices, normals, uv, indices, [round(float(heights.min()),4), round(float(heights.max()),4)]


def make_ground(plan, cameras, images, buffers, roofs, directory=None, ground_masks=None, supplements=(), marks=None, locked=None):
    from ground_material import GroundMaterials
    settings = reviewed_streets(plan.get("ground", {}), marks or {}, cameras)
    lo, hi = ground_bounds(ground_anchors(plan, cameras, roofs), settings.get("marginPercent", 25))
    width, height = np.maximum(64, np.rint((hi-lo) / max(hi-lo) * 1536).astype(int))
    rgb, evidence, footprint, per_view = collect_observations(cameras, images, buffers, roofs, lo, hi, width, height, ground_masks)
    materials = GroundMaterials(marks or {})
    final, clearance, donors = ground_surface(rgb, evidence, footprint, lo, hi, settings, materials if materials.tiles else None)
    supplemented = 0
    if supplements:
        x = np.linspace(lo[0],hi[0],width)
        z = np.linspace(lo[1],hi[1],height)
        for row in range(0,height,64):
            sl = slice(row,min(row+64,height))
            xx,zz = np.meshgrid(x,z[sl])
            world = np.stack([xx,np.zeros_like(xx),zz],axis=2)
            best = np.zeros(xx.shape)
            base_color = final[sl].astype(np.float32)
            for view in supplements:
                visible, color = view.sample(world)
                distance = np.linalg.norm(world-view.camera.eye,axis=2)
                quality = view.camera.eye[1] / np.maximum(distance,1e-6)
                use = visible & (quality>.19) & (quality>best) & (evidence[sl]!=OBSERVED) & (footprint[sl]==0)
                if getattr(view, "surface_depth", None) is None:
                    # A bare shell z-buffer cannot distinguish an untraced rooftop
                    # from the street below it. Only borrow plausible road colors.
                    corridor = np.zeros(xx.shape, bool)
                    for road in settings.get("roads", []):
                        dist, _ = road_distance(xx, zz, road)
                        corridor |= dist < road["widthM"]*.46
                    difference = np.abs(color.astype(float)-final[sl]).mean(axis=2)
                    use &= corridor & (difference < 38)
                difference=np.abs(color.astype(float)-base_color).mean(axis=2)
                for road in settings["roads"]:
                    if "asphalt" in road:
                        dist, _ = road_distance(xx, zz, road)
                        use &= (dist > road["widthM"]*.55) | (np.max(np.abs(color-road["asphalt"]),axis=2) < 29)
                use &= difference<48
                alpha=(.72*np.clip(1-difference/62,0,1))[:,:,None]
                blended=base_color*(1-alpha)+color*alpha
                final[sl][use] = np.uint8(np.rint(blended[use]))
                best[use] = quality[use]
            supplemented += int(np.count_nonzero(best))
    image = Image.fromarray(final)
    mesh = terrain_mesh(lo, hi, clearance, settings.get("reliefM", 0))
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
        image.save(directory / "texture.png", compress_level=3)
        Image.fromarray(np.uint8(evidence != OBSERVED)*255).save(directory / "mask.png")
        coverage = Image.fromarray(evidence).convert("P")
        coverage.putpalette([v for color in COVERAGE_COLORS for v in color])
        coverage.save(directory / "coverage.png")
    counts = np.bincount(evidence.ravel(), minlength=4)
    info = {"method": "local", "surface": settings.get("surface", "local"), "size": [int(width), int(height)],
            "boundsXZ": [lo.tolist(), hi.tolist()], "observed": int(counts[OBSERVED]), "occluded": int(counts[OCCLUDED]),
            "outside": int(counts[OUTSIDE]), "unresolved": int(counts[UNRESOLVED]), "visibleByView": per_view, "donorPatches": donors,
            "heightRangeM": mesh[-1], "groundTriangles": len(mesh[3])//3}
    if supplements:
        info["syntheticTexels"] = supplemented
    if materials.tiles:
        info["materialSwatches"] = {name:len(tiles) for name,tiles in materials.tiles.items()}
    if locked is not None:
        locked["observed"] = evidence == OBSERVED
    return image, mesh[:4], info
