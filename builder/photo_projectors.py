"""Bake reference-camera visibility of the actual exported scene for projective materials."""

import cv2
import numpy as np
from PIL import Image


def render_triangles(glb, ground_vertices, ground_indices, roof_nodes=None):
    def read(index):
        acc = glb.doc["accessors"][index]
        view = glb.doc["bufferViews"][acc["bufferView"]]
        dtype = {5126: "<f4", 5123: "<u2", 5125: "<u4"}[acc["componentType"]]
        dims = {"VEC3": 3, "SCALAR": 1}[acc["type"]]
        return np.frombuffer(glb.blob, dtype, acc["count"] * dims, view["byteOffset"]).reshape(-1, dims)

    pieces = [ground_vertices[np.asarray(ground_indices).reshape(-1, 3)]]
    owners = [np.zeros(len(pieces[0]),np.int32)] if roof_nodes is not None else None
    for node in glb.doc["scenes"][glb.doc["scene"]]["nodes"]:
        item = glb.doc["nodes"][node]
        mesh = glb.doc["meshes"][item["mesh"]]
        for prim in mesh["primitives"]:
            verts = read(prim["attributes"]["POSITION"])
            if "indices" in prim:
                verts = verts[read(prim["indices"]).ravel()]
            tri = verts.reshape(-1, 3, 3)
            pieces.append(tri)
            if owners is not None:
                label = np.zeros(len(tri),np.int32)
                if item["name"].startswith("Photo building ") or item["name"] in roof_nodes:
                    normal = np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
                    roof = (np.abs(normal[:,1])>.50*np.linalg.norm(normal,axis=1)) & (tri[:,:,1].min(1)>.45)
                    label[roof] = node+1
                owners.append(label)
    return np.concatenate(pieces) if owners is None else (np.concatenate(pieces),np.concatenate(owners))


def clip_near(camera_triangles, near=0.08, owners=None):
    z = camera_triangles[:, :, 2]
    visible = z > near
    clipped = []
    labels = []
    # Only the broad landscape at the image edge normally crosses the camera.
    for index in np.flatnonzero(visible.any(1) & ~visible.all(1)):
        tri = camera_triangles[index]
        poly = []
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            if a[2] > near:
                poly.append(a)
            if (a[2] > near) != (b[2] > near):
                poly.append(a + (b - a) * ((near - a[2]) / (b[2] - a[2])))
        clipped.extend([[poly[0], poly[i], poly[i + 1]] for i in range(1, len(poly) - 1)])
        if owners is not None: labels.extend([owners[index]]*(len(poly)-2))
    front = camera_triangles[visible.all(1)]
    result = np.concatenate([front, clipped]) if clipped else front
    return result if owners is None else (result,np.r_[owners[visible.all(1)],np.asarray(labels,dtype=owners.dtype)])


DEPTH_ERROR_RANGE = 0.30


def visibility_image(cam, tris, longest=896, reference_depth=None, owners=None):
    """Rasterize nearest perspective-correct Z at pixel centers, including disocclusion edges."""
    w, h = np.maximum(1, np.rint(cam.size * min(1.0, longest / max(cam.size))).astype(int))
    clipped = clip_near((tris - cam.eye) @ cam.r.T,owners=owners)
    if owners is None:
        tri=clipped
    else:
        tri,owners=clipped
    pixel = tri @ cam.k.T
    xy = pixel[:, :, :2] / pixel[:, :, 2:] * ([w, h] / cam.size) - 0.5
    z = tri[:, :, 2]
    lo = np.maximum(0, np.ceil(xy.min(1)).astype(int))
    hi = np.minimum([w - 1, h - 1], np.floor(xy.max(1)).astype(int))
    a, b, c = xy.transpose(1, 0, 2)
    ab, ac = b - a, c - a
    det = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
    keep = (hi >= lo).all(1) & (np.abs(det) > 1e-8)
    if owners is not None: owners=owners[keep]
    a, ab, ac, det, lo, hi, z = (v[keep] for v in (a, ab, ac, det, lo, hi, z))
    box = np.ceil(np.log2(hi - lo + 1)).astype(int)
    depth = np.full((h, w), np.inf, np.float32)
    ownership=np.zeros((h,w),np.int32) if owners is not None else None
    # Batch similar projected sizes so a distant eave and a whole field need
    # neither the same work nor a per-triangle Python rasterization loop.
    for bucket in np.unique(box, axis=0):
        indices = np.flatnonzero(np.all(box == bucket, axis=1))
        bw, bh = 2 ** bucket
        chunk = max(1, 200000 // (bw * bh))
        for start in range(0, len(indices), chunk):
            k = indices[start:start + chunk]
            x = lo[k, 0, None, None] + np.arange(bw)[None, None, :]
            y = lo[k, 1, None, None] + np.arange(bh)[None, :, None]
            dx, dy = x - a[k, 0, None, None], y - a[k, 1, None, None]
            wb = (dx * ac[k, 1, None, None] - dy * ac[k, 0, None, None]) / det[k, None, None]
            wc = (dy * ab[k, 0, None, None] - dx * ab[k, 1, None, None]) / det[k, None, None]
            wa = 1.0 - wb - wc
            inside = ((np.minimum(np.minimum(wa, wb), wc) >= -1e-7)
                      & (x <= hi[k, 0, None, None]) & (y <= hi[k, 1, None, None]))
            t, yy, xx = np.nonzero(inside)
            inv_z = wa[t, yy, xx] / z[k[t], 0] + wb[t, yy, xx] / z[k[t], 1] + wc[t, yy, xx] / z[k[t], 2]
            rows,cols=lo[k[t],1]+yy,lo[k[t],0]+xx
            values=(1.0/inv_z).astype(np.float32)
            np.minimum.at(depth,(rows,cols),values)
            if ownership is not None:
                front=values==depth[rows,cols]
                ownership[rows[front],cols[front]]=owners[k[t[front]]]
    valid = np.isfinite(depth)
    if not valid.any():
        return None
    units = max(0.002, float(depth[valid].max()) / 65534.0)
    code = np.zeros((h, w), np.uint16)
    code[valid] = np.maximum(1, np.rint(depth[valid] / units)).astype(np.uint16)
    mismatch = np.zeros((h, w), np.uint8)
    if reference_depth is not None:
        observed = cv2.resize(reference_depth, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
        usable = valid & np.isfinite(observed) & (observed > 0.1)
        # Together with the baked Z this encodes |1/Z_photo - 1/Z_surface|.
        # At a source eye the predicted disparity is exactly zero even when
        # the photograph depicts an object in front of the assigned surface.
        err = np.abs(depth[usable] - observed[usable]) / observed[usable]
        mismatch[usable] = np.rint(255 * np.minimum(1, err / DEPTH_ERROR_RANGE)).astype(np.uint8)
        mismatch[valid & ~usable] = 255
    rgb = np.stack([code >> 8, code & 255, mismatch], axis=-1).astype(np.uint8)
    record = Image.fromarray(rgb), units
    return record if ownership is None else (*record, ownership)


def embed_projectors(glb, cameras, images, materials, ground_vertices, ground_indices, priors, regions=None,
                     paired=None, roof_nodes=(), rectified_roofs=()):
    triangles, owners = render_triangles(glb, ground_vertices, ground_indices,set(roof_nodes))
    records = []
    roof_counts={}
    for key, cam in cameras.items():
        field = priors.get(key)
        has_prior = field is not None and field.anchor_error <= 6.5
        region = (regions or {}).get(key)
        check_roofs = bool(paired) and region is not None
        baked = visibility_image(cam, triangles, reference_depth=field.depth if has_prior else None,
                                 owners=owners if check_roofs else None)
        if baked is None:
            continue
        depth, units = baked[:2]
        if region is not None:
            evidence = cv2.resize(np.uint8(region), depth.size, interpolation=cv2.INTER_NEAREST_EXACT)
            if key in (paired or ()) and has_prior:
                from orbit import scaled
                from surface_depth import rays_for
                packed=np.asarray(depth).astype(np.float32)
                z=(256*packed[:,:,0]+packed[:,:,1])*units
                observed=cv2.resize(field.depth,depth.size,interpolation=cv2.INTER_NEAREST_EXACT)
                evidence &= np.abs(z-observed)<np.maximum(.85,.045*z)
                ground_y=cam.eye[1]+rays_for(scaled(cam,depth.size))[:,:,1]*z
                photo_y=cv2.resize(field.height,depth.size,interpolation=cv2.INTER_NEAREST_EXACT)
                normal_y=cv2.resize(np.abs(field.normals[:,:,1]),depth.size,interpolation=cv2.INTER_NEAREST_EXACT)
                evidence[(np.abs(ground_y)<.65)&((np.abs(photo_y)>1.35)|(normal_y<.55))]=0
            if check_roofs:
                # An incomplete photograph must not paint a sharp seam through
                # the middle of a single roof. Let the other camera or its
                # rectified mesh material supply this entire visible surface.
                ids=baked[2]
                area=np.bincount(ids.ravel())
                hits=np.bincount(ids.ravel(),evidence.ravel(),minlength=len(area))
                partial=(area>=80)&(hits>0)&(hits<.90*area)
                partial[0]=False
                if key not in paired:
                    trusted={i+1 for i,node in enumerate(glb.doc["nodes"]) if node.get("name","").removeprefix("Photo building ") in rectified_roofs}
                    partial &= np.isin(np.arange(len(partial)),list(trusted))
                evidence[partial[ids]]=0
                roof_counts[key]=int(partial.sum())
            distance = cv2.distanceTransform(evidence, cv2.DIST_L2, 5)
            alpha = np.uint8(255 * np.clip(distance / (2.5 if key in (paired or ()) else 6), 0, 1))
            depth.putalpha(Image.fromarray(alpha))
        if key in materials:
            pbr = glb.doc["materials"][materials[key]]["pbrMetallicRoughness"]
            texture = pbr["baseColorTexture"]["index"]
        else:
            texture = glb.texture(Image.fromarray(images[key]))
        record = {
            "source": key, "texture": texture, "size": cam.size.tolist(), "eye": cam.eye.tolist(),
            "rotation": cam.r.ravel().tolist(),
            "intrinsics": [cam.k[0, 0], cam.k[1, 1], cam.k[0, 2], cam.k[1, 2]],
            "depthTexture": glb.texture(depth, lossless=True),
            "depthSize": list(depth.size), "depthScale": units,
        }
        if has_prior:
            record["depthErrorRange"] = DEPTH_ERROR_RANGE
        if region is not None:
            record["matchedPercent"] = round(float(np.mean(region) * 100), 2)
        if key in (paired or ()):
            record["paired"] = True
        records.append(record)
    if records:
        glb.doc["scenes"][0]["extras"]["photo_projectors"] = {"version": 1, "cameras": records}
    return {"sources": [c["source"] for c in records], "depthSizes": {c["source"]: c["depthSize"] for c in records},
            "partialRoofsUsingBase":roof_counts,
            "pairedAreaPercent":{c["source"]:c["matchedPercent"] for c in records if c.get("paired")}}
