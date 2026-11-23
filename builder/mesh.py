"""Export closed building shells with photo-projected faces and an embedded ground map."""
import io
import json
import struct
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from ground import ground_anchors, make_ground


def building_faces(roof, rise, hip=False):
    base = roof.copy()
    base[:, 1] = 0
    center = .5 * (roof.mean(0) + base.mean(0))
    n = len(roof)
    faces = [base]
    for i in range(n):
        j = (i + 1) % n
        faces.append(np.array([base[i], base[j], roof[j], roof[i]]))
    if rise:
        # Put the ridge along the longer pair of opposing eave edges.
        q = roof
        lengths = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
        if lengths[0] + lengths[2] < lengths[1] + lengths[3]:
            q = np.roll(q, -1, axis=0)
        a, b, c, d = q
        p, s = (a + d) / 2 + [0, rise, 0], (b + c) / 2 + [0, rise, 0]
        if hip:
            inset = min(.32, .40*np.linalg.norm(a-d)/np.linalg.norm(s-p))
            p, s = p+inset*(s-p), s+inset*(p-s)
        faces += [np.array(f) for f in ([a, b, s, p], [p, s, c, d], [a, p, d], [b, c, s])]
    else:
        faces.append(roof)
    result = []
    for face in faces:
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        if normal @ (face.mean(0) - center) < 0:
            face, normal = face[::-1], -normal
        result.append((face, normal / np.linalg.norm(normal)))
    return result


def visibility_buffer(cam, faces):
    # Interpolate inverse camera-space Z over each projected triangle.
    scale = 640 / max(cam.size)
    width, height = np.maximum(1, np.ceil(cam.size * scale).astype(int))
    buffer = np.full((height, width), np.inf, np.float32)
    for face, _ in faces:
        uv, z = cam.pixels(face)
        if np.min(z) <= .01:
            continue
        uv *= scale
        for i in range(1, len(face) - 1):
            idx = [0, i, i + 1]
            tri = uv[idx]
            lo = np.maximum(0, np.floor(tri.min(0))).astype(int)
            hi = np.minimum([width - 1, height - 1], np.ceil(tri.max(0))).astype(int)
            if (hi < lo).any():
                continue
            a, b, c = tri
            edges = np.stack([b - a, c - a], axis=1)
            if abs(np.linalg.det(edges)) < 1e-6:
                continue
            y, x = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            weights = (np.stack([x + .5, y + .5], -1) - a) @ np.linalg.inv(edges).T
            w = np.dstack([1 - weights.sum(2), weights])
            inside = w.min(2) >= 0
            local = buffer[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            local[inside] = np.minimum(local[inside], 1 / (w[inside] @ (1 / z[idx])))
    return buffer, scale


def photo_for_face(face, normal, cameras, buffers):
    best, score = None, 0
    center = face.mean(0)
    probes = np.vstack([center, .72 * face + .28 * center])
    for source, cam in cameras.items():
        delta = cam.eye - center
        distance = np.linalg.norm(delta)
        cosine = normal @ delta / distance
        if cosine <= .12:
            continue
        uv, depth = cam.pixels(face)
        if np.min(depth) <= .01 or (uv < 0).any() or (uv > cam.size).any():
            continue
        zbuf, scale = buffers[source]
        pixels, z = cam.pixels(probes)
        xy = np.clip((pixels * scale).astype(int), 0, np.array(zbuf.shape[::-1]) - 1)
        visible = np.mean(z <= zbuf[xy[:, 1], xy[:, 0]] + np.maximum(.15, .012 * z))
        weight = visible * cosine * (cam.k[1, 1] / distance) ** 2
        if visible == 1 and weight > score:
            best, score = cam, weight
    return best


class SurfaceAtlas:
    """Pack partially visible and inferred faces into small, embedded texture pages."""
    tile = 256
    pad = 4
    columns = 12

    def __init__(self, glb):
        self.glb = glb
        self.pages = []
        self.count = 0
        self.stride = self.tile + 2 * self.pad
        self.width = self.stride * self.columns

    def put(self, colors, uv):
        slot = self.count % self.columns ** 2
        if slot == 0:
            page = Image.new("RGB", (self.width, self.width), (143, 139, 127))
            material = self.glb.material("Surface details " + str(len(self.pages) + 1), [1, 1, 1])
            self.pages.append((page, material))
        page, material = self.pages[-1]
        origin = np.array([slot % self.columns, slot // self.columns]) * self.stride
        padded = np.pad(colors, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode="edge")
        page.paste(Image.fromarray(padded), tuple(origin))
        self.count += 1
        return material, (origin + self.pad + .5 + uv * (self.tile - 1)) / self.width

    def finish(self):
        for page, material in self.pages:
            pbr = self.glb.doc["materials"][material]["pbrMetallicRoughness"]
            pbr["baseColorTexture"] = {"index": self.glb.texture(page)}


def bake_face(face, normal, vertices, cameras, buffers, images, variation, supplements, priors=None, material_bank=None):
    wall = abs(normal[1]) < .1
    axis = np.cross([0, 1, 0], normal) if wall else face[1] - face[0]
    axis /= np.linalg.norm(axis)
    vertical = np.cross(axis, normal)
    basis = np.stack([axis, vertical])
    local = (face - face[0]) @ basis.T
    lo, hi = local.min(0), local.max(0)
    ticks = np.linspace(0, 1, SurfaceAtlas.tile)
    u, v = np.meshgrid(ticks, ticks)
    xy = lo + np.stack([u, v], axis=2) * (hi - lo)
    world = face[0] + xy @ basis
    x, y = xy[:, :, 0] - lo[0], world[:, :, 1]
    grain = 2 * np.sin(85*x + 47*y) + 1.8 * np.cos(29*x - 59*y)
    if wall:
        base = np.array([(166, 158, 138), (161, 146, 126), (174, 167, 151), (154, 151, 139)][variation % 4])
        colors = np.clip(base + grain[:, :, None], 0, 255).astype(np.uint8)
        colors[y < .45] = [128, 119, 105]
        if hi[0] - lo[0] >= 3 and y.max() > 2.6:
            bays = max(1, round((hi[0]-lo[0]) / 3.6))
            dx = np.abs((u * bays) % 1 - .5) * (hi[0]-lo[0]) / bays
            dy = np.abs((y % 3.1) - 1.65)
            colors[(dx < .80) & (dy < .74)] = [188, 181, 159]
            colors[(dx < .69) & (dy < .61)] = [88, 99, 93]
            colors[(dx < .60) & (dx > .06) & (dy < .53) & (dy > .04)] = [73, 86, 84]
    else:
        tone = 126 + variation % 15 + grain + np.where((x % .42) < .05, -13, 0)
        colors = np.uint8(np.clip(tone[:, :, None] + [5, 2, -2], 0, 255))
    if material_bank:
        colors = material_bank.fill(face, normal, variation, u, v, colors)
    roof_fallback = colors.copy() if not wall and material_bank and material_bank.roofs else None
    scores = np.zeros(u.shape)
    for source, cam in cameras.items():
        delta = cam.eye - face.mean(0)
        cosine = normal @ delta / np.linalg.norm(delta)
        if cosine <= .17:
            continue
        uv, z = cam.pixels(world.reshape(-1, 3))
        zbuf, scale = buffers[source]
        indices = np.clip((uv * scale).astype(int), 0, np.array(zbuf.shape[::-1]) - 1)
        front = (z > .01) & (z <= zbuf[indices[:, 1], indices[:, 0]] + np.maximum(.18, .009*z))
        inside = (uv >= 0).all(1) & (uv < cam.size - 1).all(1)
        if source in (priors or {}):
            front &= priors[source].check(uv,z,normal,cam.size)
        score = cosine * cam.k[1, 1] / np.maximum(z, .01)
        use = ((front & inside) & (score > scores.ravel())).reshape(u.shape)
        if not use.any():
            continue
        mapped = cv2.remap(images[source], uv.astype(np.float32).reshape(*u.shape, 2), None, cv2.INTER_LINEAR)
        colors[use] = mapped[use]
        scores[use] = score.reshape(u.shape)[use]
    polygon = np.zeros(u.shape, np.uint8)
    cv2.fillConvexPoly(polygon, np.int32(np.round((local - lo) / (hi - lo) * (SurfaceAtlas.tile - 1))), 1)
    fraction = np.mean(scores[polygon > 0] > 0)
    if roof_fallback is not None and fraction < .32:
        # Sparse disconnected projections on an unreviewed roof read as a streaked
        # facade or canopy. Use one rectified roof covering for this whole slope.
        return roof_fallback, ((vertices-face[0])@basis.T-lo)/(hi-lo), 0., [0]*len(supplements), []
    synthetic = np.full(u.shape,-1,np.int16)
    best_score = np.zeros(u.shape)
    missing = (scores == 0) & (polygon > 0)
    for index, view in enumerate(supplements):
        delta = view.camera.eye - face.mean(0)
        cosine = normal @ delta / np.linalg.norm(delta)
        if cosine < .12 or not missing.any():
            continue
        visible, color = view.sample(world,normal) if getattr(view,"surface_depth",None) is not None else view.sample(world)
        quality = cosine * view.camera.k[1,1] / np.maximum(np.linalg.norm(world-view.camera.eye,axis=2),.1)
        use = missing & visible & (quality > best_score)
        colors[use] = color[use]
        synthetic[use] = index
        best_score[use] = quality[use]
    details = window_surrounds(colors,synthetic,world,normal,hi-lo) if wall and (synthetic>=0).any() else []
    used = [int(np.count_nonzero(synthetic==i)) for i in range(len(supplements))]
    if wall and material_bank and (material_bank.walls or material_bank.patches) and (missing & (synthetic<0)).any():
        material_bank.used+=1
    return colors, ((vertices - face[0]) @ basis.T - lo) / (hi - lo), fraction, used, details


def window_surrounds(color, synthetic, world, normal, extent):
    """Fit shallow sill/lintel geometry to enclosed dark openings in a rectified novel-view facade."""
    gray = cv2.cvtColor(color,cv2.COLOR_RGB2GRAY)
    dark = (cv2.GaussianBlur(gray,(0,0),4).astype(float)-gray > 12) & (synthetic>=0)
    binary = cv2.morphologyEx(dark.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    result = []
    size = len(gray)
    for contour in sorted(contours,key=cv2.contourArea,reverse=True):
        x,y,w,h = cv2.boundingRect(contour)
        width,height = extent*np.array([w,h])/(size-1)
        if not (.45<width<2.8 and .45<height<2.5 and .4<width/height<2.8 and min(w,h)>=5):
            continue
        if min(x,y,size-x-w,size-y-h) < 3 or cv2.contourArea(contour) < .62*w*h:
            continue
        if (synthetic[y-1:y+h+1,x-1:x+w+1]<0).any():
            continue
        outer = gray[y-2:y+h+2,x-2:x+w+2]
        rim = np.r_[outer[0],outer[-1],outer[:,0],outer[:,-1]]
        if np.median(rim)-np.median(gray[y:y+h,x:x+w]) < 12:
            continue
        left,right = sorted((world[y,x],world[y+h-1,x]),key=lambda p:p[1])
        dx = world[y,x+w-1]-world[y,x]
        # The lower and upper edges of each detected opening get a shallow reveal.
        for anchor in (left,right):
            a,b = anchor+normal*.012, anchor+dx+normal*.012
            vertical,depth = np.array([0,.075,0]),normal*.105
            quad = np.array([a,b,b+vertical,a+vertical])
            sides = [quad+depth,quad] + [np.array([quad[i],quad[(i+1)%4],quad[(i+1)%4]+depth,quad[i]+depth]) for i in range(4)]
            for side in sides:
                n = np.cross(side[1]-side[0],side[2]-side[0])
                if n @ (side.mean(0)-(quad.mean(0)+depth/2)) < 0:
                    side,n=side[::-1],-n
                verts = subdivide(side,0)
                result.append((verts,np.tile(n/np.linalg.norm(n),(len(verts),1)),np.zeros((len(verts),2))))
        if len(result)>=120:
            break
    return result


class Glb:
    def __init__(self, extras):
        self.blob = bytearray()
        self.doc = {"asset": {"version": "2.0", "generator": "Scene Studio"}, "scene": 0,
                    "scenes": [{"nodes": [], "extras": extras}], "nodes": [], "meshes": [], "materials": [],
                    "images": [], "textures": [], "bufferViews": [], "accessors": [], "buffers": []}

    def add(self, key, value):
        self.doc[key].append(value)
        return len(self.doc[key]) - 1

    def buffer(self, data, **attrs):
        self.blob.extend(bytes(-len(self.blob) % 4))
        offset = len(self.blob)
        self.blob.extend(data)
        return self.add("bufferViews", dict(buffer=0, byteOffset=offset, byteLength=len(data), **attrs))

    def material(self, name, color, image=None, lossless=False):
        pbr = {"baseColorFactor": [*color, 1], "roughnessFactor": .93, "metallicFactor": 0}
        if image is not None:
            pbr["baseColorTexture"] = {"index": self.texture(image, lossless)}
        return self.add("materials", {"name": name, "pbrMetallicRoughness": pbr})

    def texture(self, image, lossless=False):
        out = io.BytesIO()
        if lossless:
            image.save(out, format="PNG", compress_level=3)
        else:
            image.save(out, format="JPEG", quality=91)
        index = self.add("images", {"bufferView": self.buffer(out.getvalue()), "mimeType": "image/png" if lossless else "image/jpeg"})
        return self.add("textures", {"source": index})

    def mesh(self, name, groups, indices=None, scene=0):
        primitives = []
        for material, parts in groups.items():
            positions, normals, uv = (np.concatenate(block) for block in zip(*parts))
            attributes = {}
            for key, values in (("POSITION", positions), ("NORMAL", normals), ("TEXCOORD_0", uv)):
                arr = np.asarray(values, dtype="<f4")
                acc = {"componentType": 5126, "type": f"VEC{arr.shape[1]}", "count": len(arr),
                       "bufferView": self.buffer(arr.tobytes(), target=34962)}
                if key == "POSITION":
                    acc.update(min=arr.min(0).tolist(), max=arr.max(0).tolist())
                attributes[key] = self.add("accessors", acc)
            primitive = {"attributes": attributes, "material": material}
            if indices is not None:
                primitive["indices"] = self.add("accessors", {"componentType": 5123, "type": "SCALAR", "count": len(indices),
                    "bufferView": self.buffer(indices.astype("<u2", copy=False).tobytes(), target=34963)})
            primitives.append(primitive)
        index = self.add("meshes", {"name": name, "primitives": primitives})
        self.doc["scenes"][scene]["nodes"].append(self.add("nodes", {"mesh": index, "name": name}))

    def write(self, path):
        write_glb(path, self.doc, self.blob)


def write_glb(path, doc, blob):
    doc["buffers"] = [{"byteLength": len(blob)}]
    meta = json.dumps(doc, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    meta += b" " * (-len(meta) % 4)
    blob.extend(bytes(-len(blob) % 4))
    with path.open("wb") as f:
        f.write(struct.pack("<4sIII4s", b"glTF", 2, 28 + len(meta) + len(blob), len(meta), b"JSON"))
        f.write(meta)
        f.write(struct.pack("<I4s", len(blob), b"BIN\0"))
        f.write(blob)


def subdivide(face, levels):
    triangles = np.array([[face[0], face[i], face[i + 1]] for i in range(1, len(face) - 1)])
    for _ in range(levels):
        a, b, c = triangles.transpose(1, 0, 2)
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        triangles = np.concatenate([np.stack(t, axis=1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return triangles.reshape(-1, 3)


def export_scene(path, plan, cameras, photos, roofs, ground_dir=None, ground_masks=None, supplements=(), priors=None, marks=None):
    first = next(iter(cameras.values()))
    direction = first.r[2]
    if roofs:
        middle = np.median(np.concatenate(roofs), axis=0)
    else:
        x, z = np.median(ground_anchors(plan, cameras, roofs), axis=0)
        middle = np.array([x, 0, z])
    target_distance = max(1., np.linalg.norm(middle - first.eye))
    target = first.eye + direction * target_distance
    glb = Glb({"preserve_scale": True, "camera_position": first.eye.tolist(), "camera_target": target.tolist(),
               "camera_up": (-first.r[1]).tolist(), "reference_size": first.size.tolist(),
               "focal_ratio": float(first.k[0, 0] / first.k[1, 1]), "principal": (first.k[:2, 2] / first.size).tolist(),
               "camera_fov": plan["views"][0]["fovY"], "building_count": len(roofs), "scale_basis": plan["scaleBasis"]})
    marks = marks or {}
    surfaces, mapped, inferred_ridges = [], [], 0
    for b,roof in zip(plan["buildings"],roofs):
        mappings = [m for mark in marks.values() if (m:=mark.match(roof,b["id"])) is not None]
        mapped.append(mappings)
        rise = b.get("ridgeRiseM", 0)
        hip = False
        if not rise:
            for m in mappings:
                profile = m.marks.buildings[b["id"]].get("hipRiseM")
                if profile:
                    rise, hip = profile, True
                    inferred_ridges += 1
                    break
        surfaces.append(building_faces(roof,rise,hip))
    faces = [f for shell in surfaces for f in shell]
    buffers = {key: visibility_buffer(cam, faces) for key, cam in cameras.items()}
    photos_mat = {key: glb.material(key, [1, 1, 1], photo) for key, photo in photos.items()} if roofs else {}
    underside = glb.material("Foundation", [.30, .27, .22]) if roofs else None
    atlas = SurfaceAtlas(glb)
    images = {key: np.asarray(photo) for key, photo in photos.items()}
    from facade_material import FacadeMaterials
    from surface_context import add_context, slab_edge
    from global_surface import add_global_surface
    from landscape import add_landscape
    material_bank = FacadeMaterials(surfaces, cameras, images, buffers, priors or {}, supplements, marks,
                                    building_ids=[b["id"] for b in plan["buildings"]])
    marked_materials=dict(photos_mat)
    reviewed_by_view=defaultdict(int)
    coping = glb.material("Roof coping",[.42,.34,.27]) if roofs and any(m.data.get("eaves") for m in marks.values()) else None
    counts = {"photoFaces": 0, "partialFaces": 0, "inferredFaces": 0}
    novel = {"buildingFaces":0,"openingFrames":0,"byView":{v.camera.source:0 for v in supplements}}
    detail_material = None
    for variation, (b, shell, roof) in enumerate(zip(plan["buildings"], surfaces, roofs)):
        groups = defaultdict(list)
        mappings=mapped[variation]
        for face, normal in shell:
            cam = photo_for_face(face, normal, cameras, buffers)
            if cam and cam.source in (priors or {}):
                probes = subdivide(face,2).reshape(-1,3,3).mean(axis=1)
                uv,z = cam.pixels(probes)
                cosine = normal @ (cam.eye-face.mean(0)) / np.linalg.norm(cam.eye-face.mean(0))
                minimum = .84 if abs(normal[1])<.1 else .70
                if cosine<.22 or np.mean(priors[cam.source].check(uv,z,normal,cam.size)) < minimum:
                    cam = None
            # A projective photo warp is nonlinear in world coordinates. Subdivide
            # visible faces to reduce affine UV interpolation error when orbiting.
            vertices = subdivide(face, 2)
            reviewed=None
            for mapping in mappings:
                coords=mapping.texture(face,normal,vertices)
                if coords is not None:
                    reviewed=(mapping.marks, coords)
                    break
            roof_patch = (material_bank.reviewed_roof_uv(variation, roof, vertices)
                          if reviewed is None and cam is None and normal[1] > .35 else None)
            if reviewed is not None:
                mark,tex=reviewed
                if mark.source not in marked_materials:
                    marked_materials[mark.source]=glb.material("Reviewed view "+mark.source,[1,1,1],Image.fromarray(mark.image))
                material=marked_materials[mark.source]
                reviewed_by_view[mark.source]+=1
                if mark.source in cameras:
                    counts["photoFaces"]+=1
                else:
                    novel["buildingFaces"]+=1
                    counts["inferredFaces"]+=1
            elif cam:
                material, tex = photos_mat[cam.source], cam.pixels(vertices)[0] / cam.size
                counts["photoFaces"] += 1
            elif roof_patch is not None:
                source, photo, tex = roof_patch
                if source not in marked_materials:
                    marked_materials[source] = glb.material("Reviewed roof "+source, [1,1,1], Image.fromarray(photo))
                material = marked_materials[source]
                material_bank.roof_sources[variation] = source
                counts["inferredFaces"] += 1
            elif normal[1] < -.99:
                material, tex = underside, np.zeros((len(vertices), 2))
                counts["inferredFaces"] += 1
            else:
                tile, uv, observed, samples, details = bake_face(face, normal, vertices, cameras, buffers, images, variation, supplements, priors, material_bank)
                material, tex = atlas.put(tile, uv)
                counts["partialFaces" if observed > 0 else "inferredFaces"] += 1
                if any(samples):
                    novel["buildingFaces"] += 1
                    for view,n in zip(supplements,samples):
                        novel["byView"][view.camera.source] += n
                if details:
                    if detail_material is None:
                        detail_material = glb.material("Inferred facade trim",[.55,.52,.47])
                    groups[detail_material].extend(details)
                    novel["openingFrames"] += len(details)//12
            groups[material].append((vertices, np.tile(normal, (len(vertices), 1)), tex))
        if coping is not None:
            groups[coping].extend(slab_edge(roof))
        glb.mesh(b["id"], groups)
    atlas.finish()
    counts["surfaceDetail"] = {"wallDonors":len(material_bank.walls or material_bank.patches), "inferredWalls":material_bank.used,
                               "reviewedFaces":sum(reviewed_by_view.values()), "reviewedFacesByView":dict(reviewed_by_view),
                               "inferredRidges":inferred_ridges,
                               "context":add_context(glb, plan, cameras, roofs, marks, marked_materials, material_bank),
                               "depthAnchorMedianPercent":{key:prior.anchor_error for key,prior in (priors or {}).items()}}
    glb.doc["scenes"][0]["extras"]["reference_source"] = first.source
    locked = {}
    tex, (vertices, normals, uv, indices), ground = make_ground(plan, cameras, images, buffers, roofs, ground_dir, ground_masks, supplements, marks, locked=locked)
    fusion,tex = add_global_surface(glb, priors or {}, marks, {**images, **{v.camera.source:v.image for v in supplements}},
                                    marked_materials, roofs, material_bank, ground, tex, locked["observed"], plan)
    counts["surfaceDetail"]["depthViews"] = {}
    if fusion is not None:
        counts["surfaceDetail"]["globalSurface"] = fusion
    from local_ground_patch import apply_local_yards
    tex, yard_counts = apply_local_yards(tex, ground, locked["observed"], material_bank, roofs)
    if ground_dir and (fusion is not None or yard_counts):
        tex.save(ground_dir/"texture.png",compress_level=3)
    ground["localYardTexelsBySource"] = yard_counts
    if fusion:
        registered_sources = set(cameras) | {v.camera.source for v in supplements}
        counts["surfaceDetail"]["localMaterialReferences"] = [
            {"source": source, "angle": spec.get("requestedAngle",spec.get("angle")),
             "model": spec.get("model"), "walls": fusion["localWallsBySource"].get(source,0),
             "roofs": fusion["localRoofsBySource"].get(source,0), "yardTexels": yard_counts.get(source,0),
             "checkMedianPx": spec.get("registration",{}).get("checkMedianPx")}
            for source,spec in material_bank.reference_specs.items() if source not in registered_sources
        ]
    glb.doc["scenes"][0]["extras"]["footprint_bounds_xz"] = ground["boundsXZ"]
    counts["surfaceDetail"]["landscape"] = add_landscape(glb,plan,marks,ground,tex,vertices)
    if plan.get("videoEvidence"):
        from video_surfaces import add_evidence
        counts["surfaceDetail"]["videoSurfaces"] = add_evidence(glb, plan, photos[first.source])
    if roofs:
        from photo_projectors import embed_projectors
        # Reserve full-image projection for source photographs and registered
        # generated views with independent check points. Other views may still
        # contribute individual surfaces where their local evidence agrees.
        checked = {s["source"] for s in plan.get("supplements", [])
                   if sum(bool(p.get("check")) for p in s.get("points", [])) >= 4}
        views = {**cameras, **{v.camera.source: v.camera for v in supplements if v.camera.source in checked}}
        project_images = {**images, **{v.camera.source: v.image for v in supplements if v.camera.source in checked}}
        regions = {v.camera.source: v.registration_region for v in supplements
                   if v.camera.source in marks and marks[v.camera.source].data.get("projectionRegion") == "matched"}
        counts["surfaceDetail"]["photoProjectors"] = embed_projectors(
            glb, views, project_images, marked_materials, vertices, indices, priors or {}, regions,
            {k for k in views if k in marks and marks[k].paired_mask is not None}, [b["id"] for b in plan["buildings"]],
            {plan["buildings"][i]["id"] for i in material_bank.roof_sources}
        )
    counts["ground"] = ground
    if supplements:
        counts["supplements"] = novel
    mat = glb.material("Ground", [1, 1, 1], tex, lossless=True)
    glb.mesh("Ground", {mat: [(vertices, normals, uv)]}, indices)
    glb.write(path)
    return counts
