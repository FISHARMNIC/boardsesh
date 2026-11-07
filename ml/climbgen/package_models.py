"""Package trained climb generator models for download (see
packages/mobile/src/lib/climb-generator/remote-models.ts).

Reads each model's export from out/<model>/ (what train.py and grader.py write)
and writes the hosted layout:

  <dest>/v2/manifest.json
  <dest>/v2/<model>/<version>/model-meta.json
  <dest>/v2/<model>/<version>/weights.bin      raw int8/float32 blob (no base64)
  <dest>/v2/<model>/<version>/grader.json

Manifest paths are relative to the repo root (v2/...), like v3's. An existing
<dest>/v2/manifest.json is kept and updated, so packaging one board leaves the
others listed. Files of a version are written before the manifest
names them. Upload the folder with the Hugging Face CLI, e.g.

  .venv/bin/hf upload <user>/boardsesh-climbgen out/hub/v2 v2 --repo-type model

Usage (from ml/climbgen/):
  .venv/bin/python -I package_models.py --version 2026-11-06 tension-9 tension-10 tension-11
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from pathlib import Path

WEIGHTS_PATTERN = re.compile(r"'([A-Za-z0-9+/=]+)'")


def file_entry(dest: Path, relative: str, data: bytes) -> dict:
    path = dest / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def put(v2: Path, relative: str, data: bytes) -> dict:
    info = file_entry(v2, relative, data)
    info["path"] = f"v2/{info['path']}"  # paths are relative to the models URL root
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+", help="model keys, e.g. tension-9")
    parser.add_argument("--version", required=True, help="a new version name, e.g. the training date")
    parser.add_argument("--source", type=Path, default=Path("out"))
    parser.add_argument("--dest", type=Path, default=Path("out/hub"))
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+", args.version):
        parser.error("--version may use letters, digits, '.', '_' and '-' only")

    v2 = args.dest / "v2"
    manifest_path = v2 / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"formatVersion": 1, "models": {}}

    for key in args.models:
        source = args.source / key
        meta = (source / "model-meta.generated.json").read_bytes()
        grader = (source / "grader.generated.json").read_bytes()
        weights_ts = (source / "model-weights.generated.ts").read_text()
        match = WEIGHTS_PATTERN.search(weights_ts)
        if not match:
            raise SystemExit(f"no base64 weights in {source / 'model-weights.generated.ts'}")
        weights = base64.b64decode(match.group(1))
        if json.loads(meta).get("key") not in (None, key):
            raise SystemExit(f"{source} holds a different model than {key}")

        prefix = f"{key}/{args.version}"
        manifest["models"][key] = {
            "version": args.version,
            "files": {
                "meta": put(v2, f"{prefix}/model-meta.json", meta),
                "weights": put(v2, f"{prefix}/weights.bin", weights),
                "grader": put(v2, f"{prefix}/grader.json", grader),
            },
        }
        total = len(meta) + len(weights) + len(grader)
        print(f"{key} {args.version}: {total / 1e3:.0f} kB")

    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path} ({', '.join(sorted(manifest['models']))})")


if __name__ == "__main__":
    main()
