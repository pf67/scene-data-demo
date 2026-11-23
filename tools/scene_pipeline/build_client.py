"""Upload references, optionally request a model draft, validate and build over HTTP."""
import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8090")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source", nargs=2, action="append", default=[], metavar=("PLAN_ID", "PHOTO"))
    parser.add_argument("--photo", type=Path, action="append", default=[], help="Register an image whose content ID is already in the plan")
    parser.add_argument("--prompt", help="Request outlines from the server-configured vision model")
    parser.add_argument("--draft-only", action="store_true", help="Save a plan for editing before geometry validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    server = args.server.rstrip("/")

    def request(route, payload=None, mime="application/json"):
        data = json.dumps(payload).encode() if payload is not None and not isinstance(payload, bytes) else payload
        req = urllib.request.Request(server + route, data=data, headers={"Content-Type": mime})
        with urllib.request.urlopen(req, timeout=150) as response:
            return response.read()

    def call(route, payload, mime="application/json"):
        return json.loads(request(route, payload, mime))["data"]

    plan = json.loads(args.plan.read_text("utf-8-sig"))
    ids = {}
    known = {v["source"] for v in plan["views"]}
    known.update(v["groundMask"] for v in plan["views"] if v.get("groundMask"))
    known.update(v["source"] for v in plan.get("supplements", []))
    inputs = list(args.source)
    for photo in args.photo:
        inputs.append(("src_" + hashlib.sha256(photo.read_bytes()).hexdigest(), photo))
    for alias, filename in inputs:
        if alias not in known or alias in ids:
            parser.error(f"Unknown or duplicate --source alias: {alias}")
        info = call("/api/builder/sources", Path(filename).read_bytes(), "application/octet-stream")
        ids[alias] = info["source"]
    for v in plan["views"]:
        v["source"] = ids.get(v["source"], v["source"])
        if "groundMask" in v:
            v["groundMask"] = ids.get(v["groundMask"], v["groundMask"])
    for b in plan["buildings"]:
        for o in b["observations"]:
            o["source"] = ids.get(o["source"], o["source"])
    for spec in plan.get("supplements",[]):
        for key in ("source","reference"):
            spec[key] = ids.get(spec[key],spec[key])
    if args.prompt:
        plan = call("/api/builder/propose", {"plan": plan, "instruction": args.prompt})["plan"]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), "utf-8")
    if args.draft_only:
        return
    call("/api/builder/validate", plan)
    result = call("/api/builder/build", plan)
    scene = result["scene"]
    (args.output / "scene.glb").write_bytes(request(scene["url"]))
    (args.output / "report.json").write_bytes(request(scene["report_url"]))
    if scene.get("ground_url"):
        atlas = args.output / "ground"
        atlas.mkdir(exist_ok=True)
        for name in ("texture", "mask", "coverage"):
            (atlas / (name + ".png")).write_bytes(request(scene["ground_url"] + "/" + name))
    print(server + "/?scene=" + scene["code"])


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read())["detail"]
        except (ValueError, KeyError):
            detail = e.reason
        sys.exit(f"HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        sys.exit(f"Server connection failed: {e.reason}")
