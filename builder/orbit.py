"""Render a target-camera layout, register novel photographs, and keep synthetic evidence separate."""
from dataclasses import dataclass
import math
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from geometry import Camera
from mesh import building_faces
from ground import field_pattern, road_distance, variation


def scaled(cam, size):
    size = np.array(size, dtype=float)
    k = cam.k.copy()
    k[:2] *= (size/cam.size)[:, None]
    return Camera(cam.source, size, k, cam.r, cam.t)


def clip_near(vertices, cam):
    depth = vertices @ cam.r[2] + cam.t[2]
    if (depth > .2).all():
        return vertices
    points = []
    for i, a in enumerate(vertices):
        b, za, zb = vertices[i-1], depth[i], depth[i-1]
        if (za > .2) != (zb > .2):
            points.append(a + (b-a)*((.2-za)/(zb-za)))
        if za > .2:
            points.append(a)
    return np.asarray(points)


def surface_map(cam, faces):
    """World coordinates of the front-most surface at each pixel, with perspective-correct depth."""
    width, height = cam.size.astype(int)
    yy, xx = np.mgrid[:height,:width].astype(np.float32)
    rays = np.stack([(xx-cam.k[0,2])/cam.k[0,0], (yy-cam.k[1,2])/cam.k[1,1], np.ones_like(xx)],-1) @ cam.r
    depth = np.full((height,width), np.inf, np.float32)
    np.divide(-cam.eye[1], rays[:,:,1], out=depth, where=rays[:,:,1] < -.035)
    ids = np.where(np.isfinite(depth), -1, -2).astype(np.int32)
    for i, (face, normal) in enumerate(faces):
        if normal @ (cam.eye-face.mean(0)) <= 0:
            continue
        polygon = clip_near(face, cam)
        if len(polygon) < 3:
            continue
        uv, _ = cam.pixels(polygon)
        x0,y0 = np.maximum(0,np.floor(uv.min(0))).astype(int)
        x1,y1 = np.minimum([width-1,height-1],np.ceil(uv.max(0))).astype(int)
        if x1 <= x0 or y1 <= y0:
            continue
        window = np.s_[y0:y1+1,x0:x1+1]
        mask = np.zeros(depth[window].shape, np.uint8)
        cv2.fillConvexPoly(mask,np.int32(np.rint(uv-[x0,y0])),1)
        denom = rays[window] @ normal
        z = float(normal @ (face[0]-cam.eye)) / np.where(np.abs(denom)>1e-8,denom,1e-8)
        visible = (mask != 0) & (z > .19) & (z < depth[window])
        depth[window][visible] = z[visible]
        ids[window][visible] = i
    world = (cam.eye + rays * np.where(np.isfinite(depth),depth,0)[:,:,None]).astype(np.float32)
    return world, depth, ids


@dataclass
class Layout:
    camera: Camera
    pivot: np.ndarray
    image: Image.Image
    world: np.ndarray
    ids: np.ndarray
    ref_camera: Camera
    ref_world: np.ndarray
    ref_ids: np.ndarray
    novel_mask: Image.Image


def seed_hidden_facades(guide, ids, world, observed, source_ids, base, ref, shells):
    """Give the image model a wall material prior from the same building or a nearby visible facade."""
    faces = [(building,face,normal) for building,shell in enumerate(shells) for face,normal in shell]
    donors = []
    walls = {}
    source_counts = np.bincount(source_ids[source_ids>=0],minlength=len(faces))
    for i,(building,face,normal) in enumerate(faces):
        if abs(normal[1])>.1:
            continue
        axis = np.cross([0.,1.,0.],normal)
        axis /= np.linalg.norm(axis)
        along = face @ axis
        width, height = np.ptp(along), np.ptp(face[:,1])
        if min(width,height) < .3:
            continue
        origin = face[np.argmin(along)].copy()
        origin[1] = face[:,1].min()
        center = face.mean(0)
        walls[i] = (building,center,axis,origin,width,height)
        if source_counts[i]<160:
            continue
        uv,z = base.pixels(face)
        if np.any(z<=.2):
            continue
        area = abs(cv2.contourArea(uv.astype(np.float32)))
        if source_counts[i]/max(area,1)>.55 and normal@(base.eye-center)>.22*np.linalg.norm(base.eye-center):
            donors.append(i)
    if not donors:
        return
    unseen = ~observed & (ids>=0)
    for i in np.unique(ids[unseen]):
        if i not in walls:
            continue
        b,c,axis,p,w,h = walls[i]
        def cost(j):
            db,dc,_,_,dw,dh = walls[j]
            return (12 if db!=b else 0)+.055*np.linalg.norm(c-dc)+3*abs(math.log((dw/dh)/(w/h)))
        donor = min(donors,key=cost)
        _,_,da,dp,dw,dh = walls[donor]
        yy,xx = np.nonzero(unseen & (ids==i))
        points = world[yy,xx]-p
        u = np.clip((points@axis)/w,.06,.94)
        v = np.clip(points[:,1]/h,.09,.91)
        other = dp + (dw*u)[:,None]*da
        other[:,1] += v*dh
        xy,_ = base.pixels(other)
        xy = np.clip(np.rint(xy),0,base.size-1).astype(int)
        valid = source_ids[xy[:,1],xy[:,0]]==donor
        guide[yy[valid],xx[valid]] = ref[xy[valid,1],xy[valid,0]]


def orbit_camera(base, pivot, angle):
    a = math.radians(angle)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    eye = pivot + rot @ (base.eye - pivot)
    rotation = base.r @ rot.T
    return Camera("orbit", base.size, base.k, rotation, -rotation @ eye)


def make_layout(plan, base, photo, roofs, angle, size=None):
    if not roofs:
        raise ValueError("请先在参考图标注建筑，作为旋转机位的空间参照")
    if abs(angle) < 10 or abs(angle) > 180:
        raise ValueError("旋转角度须在左右 10° 至 180° 之间")
    if size is None:
        size = np.maximum(32, np.rint(base.size * min(1,1152/max(base.size)))).astype(int)
    base = scaled(base, size)
    # Keep the street at the same position in the frame when orbiting its centre.
    pivot = np.median(np.concatenate(roofs),axis=0)
    target = orbit_camera(base, pivot, angle)
    shells = [building_faces(roof,b.get("ridgeRiseM",0)) for b,roof in zip(plan["buildings"],roofs)]
    faces = [f for shell in shells for f in shell]
    src_world, src_depth, src_ids = surface_map(base,faces)
    world, _, ids = surface_map(target,faces)
    ref = np.asarray(photo.resize(tuple(base.size.astype(int)),Image.Resampling.LANCZOS))
    h,w = ids.shape
    # A textured projective guide makes matching individual roofs and street directions possible.
    guide = np.full((h,w,3), [150,146,126],np.uint8)
    # Supply a continuous ground context for the new camera. Flat blank canvas
    # in disocclusions is easily interpreted as a literal background by an image model.
    gx,gz = world[:,:,0],world[:,:,2]
    rng = np.random.default_rng(924)
    roads = plan.get("ground",{}).get("roads",[])
    if plan.get("ground",{}).get("surface") == "fields":
        direction = np.subtract(roads[0]["points"][-1],roads[0]["points"][0]) if roads else [1.,.3]
        soil = field_pattern(gx,gz,rng,(h,w),1.25,math.atan2(direction[1],direction[0]))
    else:
        tone = np.percentile(ref[h*2//3:],45,axis=(0,1))
        soil = tone + (5*variation(rng,(h,w),18))[:,:,None]
    grain = rng.normal(0,2,(h,w)).astype(np.float32)
    for road in roads:
        dist,_ = road_distance(gx,gz,road)
        soil[dist < road["widthM"]/2] = [85,87,85]
    guide[:] = np.uint8(np.clip(soil+grain[:,:,None],0,255))
    guide[ids == -2] = [178,192,195]
    palette = np.array([np.clip(np.array([168,159,141]) * (.83+.15*n[1]+.12*n[0]),0,255)
                        for _,n in faces],np.uint8)
    buildings = ids>=0
    guide[buildings] = palette[ids[buildings]]
    with np.errstate(divide="ignore",invalid="ignore"):
        uv, z = base.pixels(world.reshape(-1,3))
    uv = np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1).reshape(h,w,2).astype(np.float32)
    z = z.reshape(h,w)
    inside = (uv>=1).all(2) & (uv < base.size-2).all(2) & (ids != -2) & (z>.2)
    indices = np.clip(np.rint(uv),0,base.size-1).astype(int)
    sampled_depth = src_depth[indices[:,:,1],indices[:,:,0]]
    match = inside & (ids == src_ids[indices[:,:,1],indices[:,:,0]]) & (z <= sampled_depth + np.maximum(.12,z*.006))
    # The ground projective warp is only reliable near the camera.
    color = cv2.remap(ref, uv, None, cv2.INTER_LINEAR)
    guide[match] = color[match]
    if abs(angle)>=60:
        seed_hidden_facades(guide,ids,world,match,src_ids,base,ref,shells)
    if base.source == "village":
        # The bundled photo also ships its depth and fitted scene-unit scale.
        # Splat real vegetation and terrain into the otherwise empty guide background.
        assets = Path(__file__).resolve().parents[1]/"assets"
        depth_file = assets/"da3_out/village_road_near/depth.npy"
        if depth_file.is_file():
            scale = json.loads((assets/"village_studio/layout.json").read_text("utf-8"))["scale"]
            depth = cv2.resize(np.load(depth_file).astype(np.float32),(w,h))*scale
            yy,xx = np.mgrid[:h,:w]
            rays = np.stack([(xx-base.k[0,2])/base.k[0,0],(yy-base.k[1,2])/base.k[1,1],np.ones_like(xx)],-1) @ base.r
            xyz = base.eye + rays*depth[:,:,None]
            proj,tz = target.pixels(xyz.reshape(-1,3))
            valid = (tz>.2)&np.isfinite(proj).all(1)&(proj>=0).all(1)&(proj<[w-1,h-1]).all(1)
            idx = np.flatnonzero(valid)
            order = idx[np.argsort(tz[idx])[::-1]]
            px,py = np.int32(np.rint(proj[order])).T
            layer = np.zeros_like(ref); seen = np.zeros((h,w),np.uint8)
            layer[py,px] = ref.reshape(-1,3)[order]; seen[py,px] = 1
            # Interpolate small gaps from finite sampling without spreading a whole roof into a courtyard.
            weight = cv2.GaussianBlur(seen.astype(np.float32),(9,9),1.8)
            soft = cv2.GaussianBlur(layer.astype(np.float32),(9,9),1.8)/np.maximum(weight[:,:,None],.001)
            context = ~buildings & (weight>.08)
            guide[context] = np.uint8(np.clip(soft[context],0,255))
            direct = ~buildings & (seen!=0)
            guide[direct] = layer[direct]
    missing = np.uint8((~match) & (ids != -2)) * 255
    mask = Image.fromarray(cv2.GaussianBlur(missing, (5, 5), .9))
    return Layout(target,pivot,Image.fromarray(guide),world,ids,base,src_world,src_ids,mask)


def rotation_prompt(angle):
    direction = "right" if angle>0 else "left"
    return (f"Generate one new photograph of the real scene in Image 2, moving the camera {abs(angle):g} degrees to the {direction}. "
            "Image 1 gives the EXACT target camera composition, roof locations and road perspective. Output this first "
            "viewpoint. Keep its building corners, apparent size and tight spacing pixel-aligned. Image 2 is the original "
            "photograph of these same houses, with a DIFFERENT viewpoint. Use it for identity, materials, light and color. "
            "Show the newly revealed side walls, courtyard ground, street and roof detail from the target camera. "
            "Produce a single natural aerial photograph, preserving every building. No collage, text or 2D rotation.")


@dataclass
class RegisteredView:
    camera: Camera
    image: np.ndarray
    depth: np.ndarray
    support: np.ndarray
    surface_depth: object = None
    ground_mask: object = None
    registration_region: object = None

    def sample(self, world, normal=None):
        shape = world.shape[:-1]
        cam = self.camera
        with np.errstate(divide="ignore",invalid="ignore"):
            uv,z = cam.pixels(world.reshape(-1,3))
        uv = np.nan_to_num(uv,nan=-1,posinf=-1,neginf=-1)
        inside = (z>.2) & (uv>=1).all(1) & (uv<cam.size-2).all(1)
        ij = np.clip(np.rint(uv),0,cam.size-1).astype(int)
        front = self.depth[ij[:,1],ij[:,0]]
        valid = inside & (np.abs(z-front) < np.maximum(.16,z*.004)) & self.support[ij[:,1],ij[:,0]]
        if self.surface_depth is not None:
            valid &= self.surface_depth.check(uv, z, normal, cam.size)
        if normal is None and self.ground_mask is not None:
            valid &= self.ground_mask[ij[:,1],ij[:,0]]!=0
        color = cv2.remap(self.image,uv.astype(np.float32).reshape(*shape,2),None,cv2.INTER_LINEAR)
        return valid.reshape(shape), color


def root_sift(desc):
    return np.sqrt(desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-8))


def keypoint_pairs(detector, ref, world, ids, target_keys, target_desc, normalized=False):
    # Restrict matches to stable surfaces with actual 3D support.
    width = 5 if normalized else 3
    valid = cv2.erode((ids>=0).astype(np.uint8),np.ones((width,width),np.uint8))
    keys, desc = detector.detectAndCompute(cv2.cvtColor(ref,cv2.COLOR_RGB2GRAY),valid*255)
    if desc is None or target_desc is None or min(len(desc),len(target_desc)) < 2:
        return [], []
    if normalized:
        desc = root_sift(desc)
    ratio = .79 if normalized else .77
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc,target_desc,k=2)
    inverse = {m.queryIdx:m.trainIdx for m in matcher.match(target_desc,desc)}
    xyz, uv = [], []
    for m,n in matches:
        if m.distance >= ratio*n.distance or inverse.get(m.trainIdx) != m.queryIdx:
            continue
        x,y = np.rint(keys[m.queryIdx].pt).astype(int)
        xyz.append(world[y,x])
        uv.append(target_keys[m.trainIdx].pt)
    return xyz, uv


def match_image_api_view(layout, reference, candidate, faces, angle):
    # Samples from intermediate virtual cameras help match roofs when the model
    # turns less than requested. Only surfaces visible in the real photo can vote.
    detector = cv2.SIFT_create(nfeatures=8500, contrastThreshold=.014)
    keys, desc = detector.detectAndCompute(cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY), None)
    if desc is None:
        return [], []
    desc = root_sift(desc)
    def correspondences(rgb, world, ids):
        return keypoint_pairs(detector, rgb, world, ids, keys, desc, normalized=True)
    xyz, uv = correspondences(reference, layout.ref_world, layout.ref_ids)
    other, pixels = correspondences(np.asarray(layout.image), layout.world, layout.ids)
    xyz.extend(other)
    uv.extend(pixels)
    base = layout.ref_camera
    for fraction in (.25, .5):
        camera = orbit_camera(base, layout.pivot, angle * fraction)
        world, _, ids = surface_map(camera, faces)
        with np.errstate(divide="ignore", invalid="ignore"):
            xy, z = base.pixels(world.reshape(-1, 3))
        xy = np.nan_to_num(xy, nan=-1, posinf=-1, neginf=-1).reshape(*ids.shape, 2).astype(np.float32)
        ij = np.clip(np.rint(xy), 0, base.size-1).astype(int)
        visible = ((ids >= 0) & (z.reshape(ids.shape) > .2) & (xy >= 1).all(2) & (xy < base.size-2).all(2)
                   & (ids == layout.ref_ids[ij[:,:,1], ij[:,:,0]]))
        projected = cv2.remap(reference, xy, None, cv2.INTER_LINEAR)
        other, pixels = correspondences(projected, world, np.where(visible, ids, -1))
        xyz.extend(other)
        uv.extend(pixels)
    if not uv:
        return [], []
    # A feature rediscovered through multiple reprojections is one observation.
    _, unique = np.unique(np.rint(np.asarray(uv)/2), axis=0, return_index=True)
    return np.asarray(xyz)[unique], np.asarray(uv)[unique]


def signed_yaw(origin, target):
    a,b = origin[[0,2]],target[[0,2]]
    return math.degrees(math.atan2(a[1]*b[0]-a[0]*b[1],a@b))


def appearance_support(layout, rgb, world, fitted_ids, region):
    """Reject blank areas and texture copied from the target camera guide after pose alignment."""
    guide = np.asarray(layout.image)
    fitted_guide = np.empty_like(rgb)
    comparable = np.zeros(fitted_ids.shape, bool)
    for top in range(0, len(rgb), 64):
        band = slice(top, top+64)
        surface = world[band]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv, z = layout.camera.pixels(surface.reshape(-1,3))
        uv = np.nan_to_num(uv, nan=-1, posinf=-1, neginf=-1)
        inside = (z>.2) & (uv>=1).all(1) & (uv<layout.camera.size-2).all(1)
        xy = np.clip(np.rint(uv),0,layout.camera.size-1).astype(int)
        ids = fitted_ids[band].ravel()
        ids_shape = surface.shape[:2]
        comparable[band] = (inside & (ids!=-2) & (layout.ids[xy[:,1],xy[:,0]]==ids)).reshape(ids_shape)
        fitted_guide[band] = cv2.remap(guide, uv.astype(np.float32).reshape(*ids_shape,2), None, cv2.INTER_LINEAR)
    gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    guide_gray = cv2.cvtColor(fitted_guide,cv2.COLOR_RGB2GRAY).astype(np.float32)
    kernel = (11,11)
    variance = cv2.blur(gray*gray,kernel)-cv2.blur(gray,kernel)**2
    # Compare local contrast so a uniform exposure or tint change is insufficient.
    delta = gray-guide_gray
    contrast_delta = cv2.blur(np.abs(delta-cv2.blur(delta,kernel)),(7,7))
    texture = region & (fitted_ids!=-2) & (variance>9)
    original_detail = texture & (~comparable | (contrast_delta>4.5))
    fraction = int(original_detail.sum())/max(1,int(texture.sum()))
    return original_detail, fraction


def estimate_pose(xyz, uv, intrinsics, tolerance, relative_view):
    # A category-based camera instruction can also change the field of view.
    # Select the focal scale with the strongest geometric support across the image.
    best = None
    for scale in ((1., .8, 1.25, .65, 1.5) if relative_view else (1.,)):
        k = intrinsics.copy()
        k[0,0] *= scale
        k[1,1] *= scale
        cv2.setRNGSeed(319)
        ok, rv, tv, keep = cv2.solvePnPRansac(xyz,uv,k,None,iterationsCount=2200,
            reprojectionError=tolerance,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if ok and keep is not None and (best is None or len(keep)>len(best[3])):
            best = (k,rv,tv,keep)
    return best


def newly_visible_faces(layout, world, ids, detail):
    """Count supported interior pixels on building faces hidden in the real reference."""
    candidates = (ids >= 0) & detail
    # Exclude silhouettes; tiny camera-fit errors commonly shift roof edges by a pixel.
    spread = cv2.dilate(ids.astype(np.float32), np.ones((5,5),np.uint8))
    lower = cv2.erode(ids.astype(np.float32), np.ones((5,5),np.uint8))
    candidates &= spread == lower
    if not candidates.any():
        return 0, 0
    with np.errstate(divide="ignore",invalid="ignore"):
        uv, z = layout.ref_camera.pixels(world[candidates])
    uv = np.nan_to_num(uv, nan=-1, posinf=-1, neginf=-1)
    inside = (z>.2) & (uv>=1).all(1) & (uv<layout.ref_camera.size-2).all(1)
    xy = np.clip(np.rint(uv),0,layout.ref_camera.size-1).astype(int)
    faces = ids[candidates]
    hidden = ~inside | (layout.ref_ids[xy[:,1],xy[:,0]] != faces)
    _, counts = np.unique(faces[hidden],return_counts=True)
    reliable = counts[counts>=24]
    return len(reliable), int(reliable.sum())


def local_evidence(region, inliers, disputed, diagonal):
    # An edited photograph may contain locally shifted buildings under an otherwise
    # valid camera. Limit its contribution to neighborhoods supported by landmarks.
    seeds = np.ones(region.shape, np.uint8)
    width, height = np.array(region.shape[::-1])
    xy = np.clip(np.rint(inliers).astype(int), 0, [width-1, height-1])
    seeds[xy[:, 1], xy[:, 0]] = 0
    local = np.uint8(region & (cv2.distanceTransform(seeds, cv2.DIST_L2, 3) <= diagonal*.08))
    for pixel, error in disputed:
        radius = round(np.clip(2*error, diagonal*.035, diagonal*.08))
        cv2.circle(local, tuple(np.rint(pixel).astype(int)), radius, 0, -1)
    return local.astype(bool)


def register(layout, reference, image, angle, faces, points=None, *, relative_view=False, image_api=False):
    image = image.convert("RGB")
    diag = float(np.linalg.norm(image.size))
    expected = scaled(layout.camera,image.size)
    rgb = np.asarray(image)
    if abs((image.width/image.height)/(layout.image.width/layout.image.height)-1) > .09:
        raise ValueError("生成图的宽高比偏离参考图，请使用完整画面")
    method = "control-points" if points else "feature-match"
    check_world, check_uv = [], []
    if points:
        xyz, uv = [], []
        for pair in points:
            x,y = np.rint(np.array(pair["guide"])*(layout.camera.size-1)).astype(int)
            if "world" in pair:
                world_point = np.asarray(pair["world"], dtype=float)
                pixel, z = layout.camera.pixels([world_point])
                if z[0] <= .2 or not np.isfinite(pixel).all() or np.linalg.norm(pixel[0]-[x,y]) > 2.:
                    raise ValueError("已重建表面上的三维点与结构图所标位置不一致")
            elif layout.ids[y,x] == -2:
                raise ValueError("结构图上的对应点落在天空中，请选择建筑转角或道路交点")
            else:
                world_point = layout.world[y,x]
            image_point = np.asarray(pair["image"])*(expected.size-1)
            if pair.get("check"):
                check_world.append(world_point)
                check_uv.append(image_point)
            else:
                xyz.append(world_point)
                uv.append(image_point)
        if len(xyz)<6:
            raise ValueError("人工配准至少需要六对参与拟合的点，检查点单独计数")
    elif image_api:
        method = "reprojected-features"
        ref = np.asarray(reference.resize(layout.image.size, Image.Resampling.LANCZOS))
        xyz, uv = match_image_api_view(layout, ref, rgb, faces, angle)
    else:
        detector = cv2.SIFT_create(nfeatures=6000,contrastThreshold=.018)
        tk, td = detector.detectAndCompute(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),None)
        r = np.asarray(reference.resize(layout.image.size,Image.Resampling.LANCZOS))
        xyz, uv = keypoint_pairs(detector,r,layout.ref_world,layout.ref_ids,tk,td)
        gx,gu = keypoint_pairs(detector,np.asarray(layout.image),layout.world,layout.ids,tk,td)
        xyz.extend(gx); uv.extend(gu)
    xyz,uv = np.float64(xyz).reshape(-1,3),np.float64(uv).reshape(-1,2)
    summary = {"status":"needs-points","method":method,"matches":len(uv),"inliers":0,"rmsPx":None,
               "actualAngle":None,"coveragePercent":0.0,"message":"自动对应不足，请在结构图和生成图标出相同的建筑角点。"}
    accepted = None
    active = expected
    active_ids = None
    min_points = 6 if points else 12
    if len(xyz)>=min_points:
        tolerance = max(3.,diag*(.008 if points else .006))
        estimated = estimate_pose(xyz,uv,expected.k,tolerance,relative_view)
        if estimated and len(estimated[3]) >= min_points:
            k,rv,tv,keep = estimated
            selected = keep.ravel()
            rv,tv = cv2.solvePnPRefineLM(xyz[selected],uv[selected],k,None,rv,tv)
            fitted = Camera("orbit",expected.size,k,cv2.Rodrigues(rv)[0],tv.ravel())
            projected,z = fitted.pixels(xyz)
            errors = np.linalg.norm(projected-uv,axis=1)
            # Matches from both references sometimes describe the same keypoint.
            used = np.flatnonzero((errors<=tolerance)&(z>0))
            _,indices = np.unique(np.rint(uv[used]/2),axis=0,return_index=True)
            used = used[indices]
            if len(used)>=min_points:
                hull = cv2.convexHull(np.float32(uv[used]))
                coverage = cv2.contourArea(hull) / np.prod(expected.size)
                yaw = signed_yaw(layout.ref_camera.eye-layout.pivot,fitted.eye-layout.pivot)
                distance = np.linalg.norm(fitted.eye-layout.pivot)/np.linalg.norm(expected.eye-layout.pivot)
                summary.update(inliers=len(used),rmsPx=round(float(np.sqrt(np.mean(errors[used]**2))),2),
                               actualAngle=round(yaw,1),coveragePercent=round(float(100*coverage),2))
                reference_ray = layout.ref_camera.eye-layout.pivot
                candidate_ray = fitted.eye-layout.pivot
                orbit_delta = math.degrees(math.acos(float(np.clip(
                    reference_ray@candidate_ray/(np.linalg.norm(reference_ray)*np.linalg.norm(candidate_ray)), -1, 1))))
                angle_error = abs((yaw-angle+180) % 360-180)
                differs = angle_error>min(25,max(9,.45*abs(angle)))
                if relative_view:
                    summary.update(orbitDeltaDeg=round(orbit_delta,1), targetMismatch=bool(differs))
                # Small camera moves can contribute to narrow alleys. Require strong,
                # distributed matches and verify an actual visibility gain below.
                strong_fit = len(used)>=40 and coverage>=.12 and summary["rmsPx"]<=diag*.004
                min_delta = 5 if strong_fit else 8
                disputed = []
                poor_checks = False
                if check_world:
                    predicted, depths = fitted.pixels(np.asarray(check_world))
                    check_errors = np.linalg.norm(predicted-np.asarray(check_uv), axis=1)
                    check_errors = np.where((depths>.2) & np.isfinite(check_errors), check_errors, diag)
                    median = float(np.median(check_errors))
                    summary.update(checkPoints=len(check_world), checkInliers=int((check_errors<=tolerance).sum()),
                                   checkMedianPx=round(median,2), checkMaxPx=round(float(check_errors.max()),2))
                    poor_checks = len(check_world)>=3 and median>tolerance
                    disputed.extend((pt,err) for pt,err in zip(check_uv,check_errors) if err>tolerance)
                if fitted.eye[1]<=0 or not .55<distance<1.65 or (orbit_delta<min_delta if relative_view else differs):
                    summary["message"]="生成图的实际机位偏离所选旋转角度，请核对对应点或重新生成。"
                elif coverage < .035:
                    summary["message"]="对应点过于集中，请在街区的前后、左右分散标点。"
                elif poor_checks:
                    summary["message"]="独立检查点偏差过大，请核对生成图和结构图中的同名屋角。"
                else:
                    active = fitted
                    world,depth,active_ids = surface_map(fitted,faces)
                    support = np.zeros(depth.shape,np.uint8)
                    cv2.fillConvexPoly(support,np.int32(np.rint(hull)),1)
                    dist = cv2.distanceTransform(1-support,cv2.DIST_L2,3)
                    region = dist<=diag*.12
                    if image_api or check_world:
                        if points:
                            disputed.extend((pt,err) for pt,err in zip(uv,errors) if err>tolerance and np.isfinite(err))
                        region = local_evidence(region,uv[used],disputed,diag)
                        summary["localSupportPercent"] = round(float(region.mean()*100),2)
                    detail, fraction = appearance_support(layout,rgb,world,active_ids,region)
                    summary["detailPercent"] = round(100*fraction,2)
                    if relative_view:
                        new_faces,new_pixels = newly_visible_faces(layout,world,active_ids,detail)
                        summary.update(newlyVisibleFaces=new_faces,newlyVisiblePixels=new_pixels)
                    look_distance = np.linalg.norm(fitted.eye-layout.pivot)
                    summary["camera"] = {"position": fitted.eye.tolist(), "target": (fitted.eye+fitted.r[2]*look_distance).tolist(),
                                         "up": (-fitted.r[1]).tolist(), "size": list(image.size),
                                         "principal": (fitted.k[:2,2]/fitted.size).tolist(), "focalRatio": float(fitted.k[0,0]/fitted.k[1,1]),
                                         "fov": math.degrees(2*math.atan(image.height/(2*fitted.k[1,1])))}
                    if fraction < .05:
                        summary.update(status="needs-detail",message="机位已对齐，但生成图主要重复结构图，请重新生成有表面细节的机位。")
                    elif relative_view and orbit_delta<8 and (new_faces<2 or new_pixels<256):
                        summary.update(status="needs-detail",message="小幅转动后可辨认的新墙面不足，请选择变化更明显的机位。")
                    else:
                        accepted = RegisteredView(fitted,rgb,depth,detail,registration_region=region)
                        summary.update(status="aligned",message="机位已对齐，可用于补充遮挡表面。")
                        if relative_view and differs:
                            summary["message"] = "实测机位与请求方向有偏差，请对照叠加图后决定是否按实测机位采用。"
    if active_ids is None:
        _,_,active_ids = surface_map(active,faces)
    edges = (cv2.dilate(active_ids.astype(np.float32),np.ones((3,3),np.uint8)) != active_ids) & (active_ids>=0)
    overlay = rgb.copy()
    overlay[edges] = .72*np.array([244,193,85]) + .28*rgb[edges]
    return accepted,summary,Image.fromarray(overlay)
