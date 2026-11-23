"""Unified CLI and Python client for Antigravity Blender Bridge (GUI socket + headless Blender 4.5)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

ADDON_PATH = Path(__file__).resolve().parent / "blender_bridge_addon.py"

DEFAULT_BLENDER_CANDIDATES = (
    r"D:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"D:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
    r"C:\Program Files\Blender Foundation\Blender 4.2\blender.exe",
)


def find_blender_binary(preferred: str | Path | None = None) -> Path:
    """Locate Blender 4.5/4.2 executable from explicit arg, BLENDER_BIN env var, standard paths, or PATH."""
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    env_bin = os.environ.get("BLENDER_BIN", "").strip()
    if env_bin:
        candidates.append(Path(env_bin))
    for item in DEFAULT_BLENDER_CANDIDATES:
        candidates.append(Path(item))
    which_bin = shutil.which("blender")
    if which_bin:
        candidates.append(Path(which_bin))
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "Could not locate blender.exe (checked BLENDER_BIN, D:/Program Files/Blender Foundation/Blender 4.5|4.2, PATH)"
    )


def _extract_marked_json(stdout: str) -> dict[str, Any]:
    start = stdout.rfind("BRIDGE_JSON_BEGIN")
    end = stdout.rfind("BRIDGE_JSON_END")
    if start < 0 or end <= start:
        raise RuntimeError(f"Blender output missing BRIDGE_JSON markers:\n{stdout[-1500:]}")
    raw = stdout[start + len("BRIDGE_JSON_BEGIN") : end].strip()
    return json.loads(raw)


class BridgeClient:
    """Client that speaks the same JSON contract in GUI TCP mode or headless `--factory-startup -b` mode."""

    def __init__(
        self,
        *,
        mode: str = "headless",
        blend_path: Path | None = None,
        blender_bin: str | Path | None = None,
        host: str = "127.0.0.1",
        port: int = 9876,
        base_dir: Path | None = None,
        timeout_sec: int = 240,
    ) -> None:
        self.mode = mode
        self.blend_path = Path(blend_path) if blend_path else None
        self.blender_bin = Path(blender_bin) if blender_bin else None
        self.host = host
        self.port = int(port)
        self.base_dir = Path(base_dir) if base_dir else (self.blend_path.parent if self.blend_path else None)
        self.timeout_sec = int(timeout_sec)

    def call(
        self,
        cmd: str,
        params: dict[str, Any] | None = None,
        *,
        save_on_ok: bool | None = None,
    ) -> dict[str, Any]:
        params = dict(params or {})
        if self.mode == "gui":
            return self._call_gui(cmd, params)
        if self.mode == "headless":
            return self._call_headless(cmd, params, save_on_ok=save_on_ok)
        raise ValueError(f"Unsupported mode: {self.mode}")

    def _call_gui(self, cmd: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps({"id": uuid.uuid4().hex[:10], "cmd": cmd, "params": params}, ensure_ascii=False)
        with socket.create_connection((self.host, self.port), timeout=self.timeout_sec) as sock:
            sock.sendall((payload + "\n").encode("utf-8"))
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
        if not buf:
            raise RuntimeError("Closed connection from GUI Blender Bridge without response")
        return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))

    def _call_headless(
        self,
        cmd: str,
        params: dict[str, Any],
        *,
        save_on_ok: bool | None = None,
    ) -> dict[str, Any]:
        exe = find_blender_binary(self.blender_bin)
        if save_on_ok is None:
            save_on_ok = cmd in ("exec_py",) and self.blend_path is not None
        if self.blend_path and "blend_path" not in params:
            params["blend_path"] = str(self.blend_path)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            req_path = Path(tmp.name)
            json.dump(
                {
                    "cmd": cmd,
                    "params": params,
                    "base_dir": str(self.base_dir) if self.base_dir else None,
                    "save_on_ok": bool(save_on_ok),
                },
                tmp,
                ensure_ascii=False,
            )

        try:
            args = [str(exe), "--factory-startup", "--python-exit-code", "1", "-b"]
            if self.blend_path and self.blend_path.is_file() and cmd != "rollback":
                args.append(str(self.blend_path))
            args.extend(["--python", str(ADDON_PATH), "--", "--req", str(req_path)])
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_sec,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Blender exited {proc.returncode}:\nSTDOUT:\n{proc.stdout[-1200:]}\nSTDERR:\n{proc.stderr[-1200:]}"
                )
            return _extract_marked_json(proc.stdout)
        finally:
            req_path.unlink(missing_ok=True)


def run_blender_script(
    script_path: Path,
    script_args: list[str] | None = None,
    *,
    blend_path: Path | None = None,
    blender_bin: str | Path | None = None,
    timeout_sec: int = 900,
) -> subprocess.CompletedProcess[str]:
    """Execute a dedicated Blender modeling/baking/export script with --factory-startup -b."""
    exe = find_blender_binary(blender_bin)
    args = [str(exe), "--factory-startup", "--python-exit-code", "1", "-b"]
    if blend_path and blend_path.is_file():
        args.append(str(blend_path))
    args.extend(["--python", str(script_path)])
    if script_args:
        args.append("--")
        args.extend(script_args)
    proc = subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Blender script failed ({proc.returncode}):\nSTDOUT:\n{proc.stdout[-1200:]}\nSTDERR:\n{proc.stderr[-1200:]}"
        )
    return proc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("headless", "gui"), default="headless")
    parser.add_argument("--blender", default=None, help="Path to blender.exe")
    parser.add_argument("--blend", type=Path, default=None, help="Target .blend file for headless mode")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9876)

    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    sub.add_parser("ping")

    p_ins = sub.add_parser("inspect")
    p_ins.add_argument("--limit", type=int, default=250)

    p_py = sub.add_parser("py")
    p_py.add_argument("--code", default=None)
    p_py.add_argument("--file", type=Path, default=None)
    p_py.add_argument("--see", action="store_true")
    p_py.add_argument("--out", type=Path, default=Path("artifacts/blender_village/see.png"))
    p_py.add_argument("--save", action="store_true")

    p_see = sub.add_parser("see")
    p_see.add_argument("--camera", default=None)
    p_see.add_argument("--eye", nargs=3, type=float, default=None)
    p_see.add_argument("--target", nargs=3, type=float, default=None)
    p_see.add_argument("--fov", type=float, default=42.0)
    p_see.add_argument("--res", nargs=2, type=int, default=[1024, 640])
    p_see.add_argument("--out", type=Path, default=Path("artifacts/blender_village/capture.png"))

    p_snap = sub.add_parser("snapshot")
    p_snap.add_argument("--label", default="ckpt")

    p_roll = sub.add_parser("rollback")
    p_roll.add_argument("--snapshot", type=Path, default=None)

    p_aud = sub.add_parser("audit")
    p_aud.add_argument("--expected-buildings", type=int, default=None)
    p_aud.add_argument("--out", type=Path, default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    client = BridgeClient(
        mode=args.mode,
        blend_path=args.blend,
        blender_bin=args.blender,
        host=args.host,
        port=args.port,
    )
    if args.cmd in ("doctor", "ping"):
        res = client.call(args.cmd)
    elif args.cmd == "inspect":
        res = client.call("inspect", {"limit": args.limit})
    elif args.cmd == "py":
        code = args.file.read_text(encoding="utf-8") if args.file else (args.code or "")
        res = client.call(
            "exec_py",
            {"code": code, "see": args.see, "out_path": str(args.out)},
            save_on_ok=bool(args.save or args.blend),
        )
    elif args.cmd == "see":
        params: dict[str, Any] = {
            "camera": args.camera,
            "fov": args.fov,
            "resolution": args.res,
            "out_path": str(args.out),
        }
        if args.eye is not None:
            params["eye"] = args.eye
        if args.target is not None:
            params["target"] = args.target
        res = client.call("capture_view", params)
    elif args.cmd == "snapshot":
        res = client.call("snapshot", {"label": args.label})
    elif args.cmd == "rollback":
        res = client.call(
            "rollback",
            {"snapshot_path": str(args.snapshot) if args.snapshot else None},
        )
    elif args.cmd == "audit":
        res = client.call("audit", {"expected_closed_buildings": args.expected_buildings})
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raise SystemExit(f"Unhandled command: {args.cmd}")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if not res.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
