"""Package a committed source tree, bundled assets and a freshly built frontend."""
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    deps = ROOT / "frontend/node_modules"
    vite = deps / "vite/bin/vite.js"
    if not vite.is_file():
        raise SystemExit("Install the locked build dependencies with: npm --prefix frontend ci")
    target = ROOT / "artifacts"
    target.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target, prefix="release-") as scratch:
        work = Path(scratch)
        archive = work / "release.zip"
        subprocess.run(["git", "archive", "--format=zip", "-o", str(archive), commit], cwd=ROOT, check=True)
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if member.startswith("frontend/"):
                    z.extract(member, work)
            version = json.loads(z.read("package.json"))["version"]
        # Build from the exported commit; local configuration and untracked
        # public files never enter Vite's source directory.
        link = work / "frontend/node_modules"
        subprocess.run(["node", "-e", "require('node:fs').symlinkSync(process.argv[1],process.argv[2],'junction')",
                        str(deps), str(link)], check=True)
        try:
            subprocess.run(["node", str(vite), "build"], cwd=work / "frontend", check=True)
        finally:
            if os.name == "nt":
                os.rmdir(link)
            else:
                link.unlink()
        with zipfile.ZipFile(archive, "a", compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted((work / "web").rglob("*")):
                if file.is_file():
                    z.write(file, file.relative_to(work).as_posix())
            z.writestr("RELEASE.json", json.dumps({"version": version, "commit": commit}, indent=2))
        result = target / f"scene-data-demo-{version}-{commit[:7]}.zip"
        archive.replace(result)
    with result.open("rb") as f:
        digest = hashlib.sha256()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    print(f"{result.name}  SHA256 {digest.hexdigest()}")


if __name__ == "__main__":
    main()
