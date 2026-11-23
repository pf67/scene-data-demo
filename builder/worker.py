"""Offline CLI and HTTP worker. All modelling inputs use the same versioned plan."""
import argparse
import io
import json
import os
import secrets
import shutil
import sys
from pathlib import Path

try:
    from jsonschema import Draft7Validator
    from PIL import Image, ImageOps, UnidentifiedImageError
    import numpy as np
    import cv2
    from geometry import fit_camera, reconstruct_building
    from mesh import export_scene
except ImportError:
    sys.exit("请为 SCENE_PYTHON 安装 builder/requirements.txt 中的建模依赖")

HERE = Path(__file__).resolve().parent
VALIDATOR = Draft7Validator(json.loads((HERE / "plan.schema.json").read_text("utf-8")))


def read_photo(path):
    try:
        with Image.open(path) as original:
            if original.format not in ("JPEG", "PNG") or original.width * original.height > 32_000_000 or min(original.size) < 32:
                raise ValueError("照片需为 PNG/JPEG，短边至少 32 px，总像素至多 3200 万")
            img = ImageOps.exif_transpose(original).convert("RGB")
            img.thumbnail((3072, 3072), Image.Resampling.LANCZOS)
            return img
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("照片无法读取，请提供完整的 PNG 或 JPEG") from error


def prepare(plan, paths):
    error = next(VALIDATOR.iter_errors(plan), None)
    if error:
        field = ".".join(str(x) for x in error.absolute_path) or "plan"
        raise ValueError(f"{field} · {error.message[:320]}")
    if not plan["name"].strip():
        raise ValueError("请输入场景名称")
    for road in plan.get("ground", {}).get("roads", []):
        if any(a == b for a, b in zip(road["points"], road["points"][1:])):
            raise ValueError("道路引导线中有重合的连续点")
    photos, cameras, views, rows, roofs = {}, {}, [], [], []
    ground_masks = {}
    for view in plan["views"]:
        key = view["source"]
        if "approxPose" in view and (len(plan["views"]) != 1 or plan["scaleBasis"] != "estimated" or view["controls"]):
            raise ValueError("估算相机仅用于单图草模，请选择估算尺度并清空控制点。多视图需统一地面标定。")
        if key in cameras or key not in paths:
            raise ValueError(f"图片 {key} 重复或未提供")
        photos[key] = read_photo(paths[key])
        mask_id = view.get("groundMask")
        if mask_id:
            mask_path = paths.get(mask_id) or (HERE / "village-ground.png" if mask_id == "village-depth-ground" else None)
            if not mask_path:
                raise ValueError(f"{key} 的地面支持掩膜未提供")
            mask = read_photo(mask_path).convert("L")
            w, h = photos[key].size
            if abs(mask.width*h/(mask.height*w)-1) > .01:
                raise ValueError(f"{key} 的地面支持掩膜须与照片方向和宽高比一致")
            ground_masks[key] = np.asarray(mask.resize((w,h), Image.Resampling.NEAREST)) >= 128
        cam, row = fit_camera(view, photos[key].size)
        cameras[key] = cam
        views.append(row)
    ids = set()
    for building in plan["buildings"]:
        key = building["id"]
        if key in ids:
            raise ValueError(f"建筑编号 {key} 重复")
        ids.add(key)
        try:
            roof, row = reconstruct_building(building, cameras)
        except ValueError as error:
            raise ValueError(f"{key} · {error}") from error
        roofs.append(roof)
        rows.append(row)
    notes = []
    if plan["scaleBasis"] == "estimated":
        notes.append("场景采用估算尺度，米制尺寸仅供参考。")
    if any(v["cameraMethod"] == "estimated" for v in views):
        notes.append("相机位置和俯角来自估算，可切换地面标定并增加独立检查点。")
    if not any(v["checkPoints"] for v in views):
        notes.append("尚无独立检查点，拟合误差只反映输入点的一致性。")
    if any(v["checkPoints"] and v["checkRmsPx"] > max(5, max(photos[v["source"]].size) * .01) for v in views):
        notes.append("独立检查点偏差较大，建议复核地面坐标、视场角和标点。")
    if any(r["heightMethod"] == "estimated" for r in rows):
        notes.append("部分建筑使用估算檐高，可补标墙脚或增加侧向视图。")
    result = {"version": 1, "name": plan["name"], "scaleBasis": plan["scaleBasis"],
              "views": views, "buildings": rows, "notes": notes}
    return result, cameras, photos, roofs, ground_masks


def scene_faces(plan, roofs):
    from mesh import building_faces
    return [face for b, roof in zip(plan["buildings"], roofs) for face in building_faces(roof,b.get("ridgeRiseM",0))]


def registration_options(spec):
    if spec.get("registration") == "target":
        return {"relative_view": False, "image_api": False}
    measured = spec.get("registration") == "measured" or spec.get("provider") == "openai"
    return {"relative_view": measured or spec.get("provider") == "qma", "image_api": measured}


def orbit_coverage(cameras, roofs):
    """Report gaps between camera positions on a horizontal orbit, in the first camera's frame."""
    if not roofs:
        return None
    from orbit import signed_yaw
    center = np.median(np.concatenate(roofs), axis=0)
    rays = {key:cam.eye-center for key,cam in cameras.items() if np.linalg.norm((cam.eye-center)[[0,2]])>.05}
    if not rays:
        return None
    base = next(iter(rays.values()))
    angles = {key:round(signed_yaw(base,d),1) for key,d in rays.items()}
    alpha = sorted(set(a%360 for a in angles.values()))
    gaps = [(b-a,a) for a,b in zip(alpha,alpha[1:]+[alpha[0]+360])]
    largest, start = max(gaps)
    targets = []
    for gap,start_angle in sorted(gaps,reverse=True):
        pieces = int(np.ceil(gap/50))
        targets.extend(round((start_angle+gap*i/pieces+180)%360-180,1) for i in range(1,pieces))
    return {"anglesDeg":angles,"largestGapDeg":round(largest,1),"gapStartDeg":round((start+180)%360-180,1),
            "suggestedAnglesDeg":targets}


def run(plan, paths, output=None, ground_dir=None):
    result, cameras, photos, roofs, ground_masks = prepare(plan, paths)
    depth_files = dict(paths)
    registered = []
    if plan.get("supplements"):
        from orbit import make_layout, register
        faces = scene_faces(plan,roofs)
        taken = set(cameras)
        result["supplements"] = []
        for spec in plan["supplements"]:
            key, reference = spec["source"], spec["reference"]
            source = paths.get(key)
            if source is None and key.startswith("village-orbit-"):
                catalog = json.loads((HERE / "sample-views/manifest.json").read_text("utf-8"))
                if key in {item["source"] for item in catalog["views"]}:
                    source = HERE / "sample-views" / (key + ".jpg")
            if source is None or reference not in cameras or key in taken:
                raise ValueError("补充视图需要独立图像和有效的原始参考图")
            taken.add(key)
            layout = make_layout(plan,cameras[reference],photos[reference],roofs,spec["angle"])
            view, quality, _ = register(layout,photos[reference],read_photo(source),spec["angle"],faces,spec.get("points"),
                                        **registration_options(spec))
            del layout
            if view is None:
                raise ValueError(f'{key} · {quality["message"]}')
            view.camera.source = key
            depth_files[key] = source
            registered.append(view)
            result["supplements"].append({**{k:spec[k] for k in ("source","reference","provenance","angle","provider","model","registration") if k in spec}, **quality})
    all_cameras = {**cameras, **{v.camera.source:v.camera for v in registered}}
    coverage = orbit_coverage(all_cameras, roofs)
    if coverage:
        result["orbitCoverage"] = coverage
    if output:
        from surface_depth import cached_depth, fit_surface_depth, image_digest
        from surface_marks import SurfaceMarks
        faces = scene_faces(plan, roofs)
        priors, marks = {}, {}
        supplements_by_source = {v.camera.source:v for v in registered}
        all_images={**{k:np.asarray(im) for k,im in photos.items()}, **{v.camera.source:v.image for v in registered}}
        for key, cam in all_cameras.items():
            digest=image_digest(depth_files[key])
            review=SurfaceMarks.load(digest,key,cam,all_images[key])
            if review is not None:
                marks[key]=review
                if key in cameras and review.ground is not None:
                    ground_masks[key]=(review.ground!=0)&ground_masks.get(key,True)
            region = None
            view = supplements_by_source.get(key)
            if view and review and review.data.get("projectionRegion") == "matched":
                mask = np.uint8(view.registration_region)
                if review.paired_mask is not None:
                    mask &= cv2.resize(np.uint8(review.paired_mask),mask.shape[::-1],interpolation=cv2.INTER_NEAREST_EXACT)
                if review.paired_ground is not None:
                    mask |= cv2.resize(np.uint8(review.paired_ground),mask.shape[::-1],interpolation=cv2.INTER_NEAREST_EXACT)
                scale = np.array(mask.shape[::-1]) / review.size
                for polygon in review.data.get("projectionExclusions", []):
                    cv2.fillPoly(mask, [np.int32(np.rint(np.asarray(polygon) * scale))], 0)
                region = view.registration_region = mask.astype(bool)
                view.support &= region
            field = fit_surface_depth(cam, cached_depth(depth_files[key],digest), faces,
                                      plan.get("ground",{}).get("roads",[]), ground_masks.get(key), region)
            if field is not None:
                priors[key] = field
                if key in cameras:
                    ground_like = (np.abs(field.height) < 1.35) & (np.abs(field.normals[:, :, 1]) > .65)
                    supported = cv2.resize(ground_like.astype(np.uint8), photos[key].size,
                                           interpolation=cv2.INTER_NEAREST).astype(bool)
                    ground_masks[key] = supported & ground_masks.get(key, True)
        for view in registered:
            view.surface_depth = priors.get(view.camera.source)
            review=marks.get(view.camera.source)
            view.ground_mask=review.ground if review else None
        result["mesh"] = export_scene(output, plan, cameras, photos, roofs, ground_dir, ground_masks, registered, priors, marks)
    return result


def orbit_candidate(args):
    from orbit import make_layout, register, rotation_prompt
    plan = json.loads(args.plan.read_text("utf-8-sig"))
    spec = {"reference":args.reference,"source":"candidate","angle":args.angle,
            "provenance":"imported-synthetic" if args.image else "qwen-image-2.1"}
    if args.points:
        spec["points"] = json.loads(args.points.read_text("utf-8"))
    if args.registration:
        spec["registration"] = args.registration
    if args.provider == "qma" or (args.provider is None and not args.image and os.environ.get("QMA_API_URL")):
        spec["provider"] = "qma"
    elif args.provider in ("openai", "comfyui"):
        spec["provider"] = args.provider
    if args.provider == "openai":
        if not args.image:
            spec["provenance"] = "image-api"
        if args.model:
            spec["model"] = args.model
    _,cameras,photos,roofs,_ = prepare({**plan,"supplements":[spec]},dict(args.source))
    if args.reference not in cameras:
        raise ValueError("请选择原始参考图")
    photo = photos[args.reference]
    layout = make_layout(plan,cameras[args.reference],photo,roofs,args.angle)
    directory = args.directory
    directory.mkdir(exist_ok=True,parents=True)
    layout.image.save(directory/"guide.jpg",quality=94)
    generation = None
    if args.image:
        if args.reuse_candidate:
            shutil.copyfile(args.image,directory/"view.jpg")
        else:
            read_photo(args.image).save(directory/"view.jpg",quality=96,subsampling=0)
    elif spec.get("provider") == "openai":
        sys.path.insert(0, str(HERE.parent))
        from refinement.image_api import generate_view
        data, generation = generate_view(photo, layout.image, args.angle, args.model, args.guidance, args.quality, args.instruction)
        spec["model"] = generation["model"]
        read_photo(io.BytesIO(data)).save(directory/"view.jpg",quality=96,subsampling=0)
    elif spec.get("provider") == "qma":
        sys.path.insert(0,str(HERE.parent))
        from refinement.qma import generate_view
        if not os.environ.get("QMA_API_URL"):
            raise ValueError("请配置 QMA_API_URL")
        data, generation = generate_view(photo, args.angle, 20261118 if args.seed is None else args.seed, args.strength)
        read_photo(io.BytesIO(data)).save(directory/"view.jpg",quality=96,subsampling=0)
    else:
        sys.path.insert(0,str(HERE.parent))
        from refinement.qwen import generate
        seed = args.seed if args.seed is not None else secrets.randbelow(2**53)
        generation = {"model":"Qwen-Image-2.1","seed":seed,"steps":36,"resolution":896,"cfg":1}
        large_turn = abs(args.angle) >= 60
        prompt = rotation_prompt(args.angle)
        if large_turn:
            prompt += (" Fill newly exposed walls with natural masonry, window openings and doors like Image 2."
                       " Restore courtyard ground and street detail. Preserve all fixed building corners.")
        generation["layoutMask"] = large_turn
        data = generate(layout.image, prompt, directory, steps=36, reference=photo, resolution=896, seed=seed,
                        novel_mask=layout.novel_mask if large_turn else None)
        read_photo(io.BytesIO(data)).save(directory/"view.jpg",quality=96,subsampling=0)
    # Registration and subsequent builds see exactly the same saved pixels.
    _,quality,overlay = register(layout,photo,read_photo(directory/"view.jpg"),args.angle,scene_faces(plan,roofs),
                                 spec.get("points"), **registration_options(spec))
    overlay.save(directory/"overlay.jpg",quality=92)
    result = {"quality":quality,"spec":spec}
    if generation:
        result["generation"] = generation
    return result


def main():
    cv2.setNumThreads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="action", required=True)
    cmd.add_parser("orbit-options", help="Read QMA directions and strength suggestions from the connected service")
    normalize = cmd.add_parser("photo", help="Orient, validate and normalize a reference photo")
    normalize.add_argument("input", type=Path)
    normalize.add_argument("output", type=Path)
    draft = cmd.add_parser("draft", help="Check proposed outlines before calibrating the views")
    draft.add_argument("--plan", required=True, type=Path)
    edit = cmd.add_parser("ground-edit", help="Write an updated model and ground maps to a new directory")
    for name in ("model", "ground-dir", "out", "out-ground"):
        edit.add_argument("--" + name, required=True, type=Path)
    method = edit.add_mutually_exclusive_group(required=True)
    method.add_argument("--image", type=Path)
    method.add_argument("--prompt")
    orbit = cmd.add_parser("orbit-view", help="Generate or register a new camera view of an original photograph")
    orbit.add_argument("--plan", required=True, type=Path)
    orbit.add_argument("--source", nargs=2, action="append", default=[])
    orbit.add_argument("--angle", required=True, type=float)
    orbit.add_argument("--reference", required=True)
    orbit.add_argument("--directory", required=True,type=Path)
    orbit.add_argument("--image", type=Path)
    orbit.add_argument("--seed", type=int)
    orbit.add_argument("--provider", choices=("openai","qma","comfyui"))
    orbit.add_argument("--model")
    orbit.add_argument("--guidance", choices=("layout","photo"))
    orbit.add_argument("--quality")
    orbit.add_argument("--instruction", default="")
    orbit.add_argument("--registration", choices=("target","measured"))
    orbit.add_argument("--strength", type=float, help="Override the recommended QMA LoRA strength (0 to 1)")
    orbit.add_argument("--reuse-candidate", action="store_true", help=argparse.SUPPRESS)
    orbit.add_argument("--points", type=Path)
    for action in ("validate", "build"):
        sub = cmd.add_parser(action)
        sub.add_argument("--plan", required=True, type=Path)
        sub.add_argument("--source", nargs=2, action="append", default=[], metavar=("ID", "FILE"))
        sub.add_argument("--report", type=Path)
        if action == "build":
            sub.add_argument("--out", required=True, type=Path)
            sub.add_argument("--ground-dir", type=Path, help="Save the ground atlas, editable mask and coverage map")
    args = parser.parse_args()
    if args.action == "orbit-options":
        sys.path.insert(0,str(HERE.parent))
        import requests
        from refinement.qma import camera_options
        try:
            with requests.Session() as session:
                result = camera_options(session)
        except requests.RequestException:
            raise ValueError("无法读取 QMA 视角选项，请确认拍摄角度服务已启动并可从场景服务访问") from None
    elif args.action == "photo":
        photo = read_photo(args.input)
        photo.save(args.output, "JPEG", quality=94)
        result = {"width": photo.width, "height": photo.height}
    elif args.action == "orbit-view":
        result = orbit_candidate(args)
    elif args.action == "ground-edit":
        from ground_edit import edit_ground
        result = edit_ground(args.model, args.ground_dir, args.out, args.out_ground, args.image, args.prompt)
    elif args.action == "draft":
        plan = json.loads(args.plan.read_text("utf-8"))
        buildings = plan["buildings"]
        error = next(VALIDATOR.descend(buildings, VALIDATOR.schema["properties"]["buildings"]), None)
        if error:
            raise ValueError(f"建筑草案格式错误 · {error.message[:220]}")
        sources = {v["source"] for v in plan["views"]}
        if len(set(b["id"] for b in buildings)) != len(buildings):
            raise ValueError("建筑草案中有重复编号")
        for b in buildings:
            keys = [o["source"] for o in b["observations"]]
            if not set(keys) <= sources or len(set(keys)) != len(keys):
                raise ValueError(f"{b['id']} 的观测图片编号无效")
        result = {"valid": True}
    else:
        plan = json.loads(args.plan.read_text("utf-8-sig"))
        result = run(plan, dict(args.source), getattr(args, "out", None), getattr(args, "ground_dir", None))
        if args.report:
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(2)
