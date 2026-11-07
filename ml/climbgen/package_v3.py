"""Package a v3 model for download, beside (never inside) the v2 manifest.

Writes into <dest>/v3/:
  manifest.json                          formatVersion 2
  <model>/<version>/model-meta.json      writer metadata (vocabulary incl. the quality token)
  <model>/<version>/weights.bin          writer weights (int8 blob)
  <model>/<version>/grader.json          grader (attention or deepsets, see "kind")
  <model>/<version>/critic.json          realism critic, paired to the writer by sha256
  <model>/<version>/geometry.json        real-classic geometry limits (features.py)
  <model>/<version>/known-climbs.bin     every known climb's holds, for near-copy rejection

known-climbs.bin: little-endian uint16 records, the hold count then that many
model placement indices, one record per distinct hold set.

The v2 root manifest is left alone, so phones on the v2 build keep working.

Usage (from ml/climbgen/):
  .venv/bin/python -I package_v3.py --version v3-2026-11-07 --grader out/v3/kilter-1-grader/grader.generated.json
  .venv/bin/hf upload fisharmnic/boardsesh-climbgen out/hub/v3 v3 --repo-type model
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

MAX_KNOWN_BYTES = 2_500_000


def entry(dest: Path, relative: str, data: bytes) -> dict:
    path = dest / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def known_climbs_blob(examples, min_ascents: int = 0) -> bytes:
    sets = sorted({tuple(sorted({p for p, _ in e.holds})) for e in examples if e.ascents >= min_ascents})
    out = bytearray()
    for holds in sets:
        out += struct.pack(f"<{len(holds) + 1}H", len(holds), *holds)
    return bytes(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kilter-1")
    parser.add_argument("--version", required=True)
    parser.add_argument("--writer", type=Path, default=Path("out/v3/kilter-1"))
    parser.add_argument("--grader", type=Path, required=True, help="grader.generated.json to ship")
    parser.add_argument("--critic", type=Path, default=Path("out/v3/kilter-1-critic/critic.generated.json"))
    parser.add_argument("--geometry", type=Path, default=Path("out/v3/geometry.json"))
    parser.add_argument("--climbs", type=Path, default=Path("data/kilter1.db"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.db"))
    parser.add_argument("--dest", type=Path, default=Path("out/hub"))
    args = parser.parse_args()

    train.configure(args.model)
    meta = (args.writer / "model-meta.generated.json").read_bytes()
    encoded = re.search(r"'([A-Za-z0-9+/=]+)'", (args.writer / "model-weights.generated.ts").read_text()).group(1)
    weights = base64.b64decode(encoded)
    critic = args.critic.read_bytes()
    if json.loads(critic)["writerSha256"] != hashlib.sha256(weights).hexdigest():
        raise SystemExit("the critic was trained against a different writer; retrain critic.py first")

    placement_ids, coords = train.load_placements(args.catalog)
    examples = train.load_examples(args.climbs, placement_ids, coords)
    known = known_climbs_blob(examples)
    if len(known) > MAX_KNOWN_BYTES:
        known = known_climbs_blob(examples, min_ascents=10)
        print("known-climbs index over budget; kept climbs with 10+ ascents")

    v3 = args.dest / "v3"
    manifest_path = v3 / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"formatVersion": 2, "models": {}}
    prefix = f"{args.model}/{args.version}"

    def put(name: str, data: bytes) -> dict:
        info = entry(v3, f"{prefix}/{name}", data)
        info["path"] = f"v3/{info['path']}"  # paths are relative to the models URL root
        return info

    manifest["models"][args.model] = {
        "version": args.version,
        "files": {
            "meta": put("model-meta.json", meta),
            "weights": put("weights.bin", weights),
            "grader": put("grader.json", args.grader.read_bytes()),
            "critic": put("critic.json", critic),
            "geometry": put("geometry.json", args.geometry.read_bytes()),
            "known": put("known-climbs.bin", known),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for name, info in manifest["models"][args.model]["files"].items():
        print(f"  {name:9} {info['bytes'] / 1e3:8.0f} kB  {info['path']}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
