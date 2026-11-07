"""Write a tiny, randomly initialised generator + grader for the mobile unit tests.

Every model now downloads, so the tests need a model of their own. Its weights
are random: the tests check the decoding rules (allowed holds, roles, start and
finish counts, grade ranges), which hold whatever the weights say. A few kB.

Usage (from ml/climbgen/):
  .venv/bin/python -I make_test_fixture.py --catalog data/catalog.db
"""

from __future__ import annotations

import argparse
import base64
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader  # noqa: E402
import train  # noqa: E402

DEST = Path(__file__).resolve().parents[2] / "packages/mobile/src/lib/climb-generator/__tests__/fixtures/tiny-model"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(0)
    train.configure("kilter-1")
    placement_ids, coords = train.load_placements(args.catalog)
    placement_ids = placement_ids[:60]
    coords = {pid: coords[pid] for pid in placement_ids}

    vocab = train.HOLD_BASE + len(placement_ids) * 4
    config = {"dim": 8, "layers": 1, "heads": 2, "ff": 16, "vocabSize": vocab, "maxLen": 3 + len(train.TAGS) + 1 + train.MAX_HOLDS + 1}
    model = train.ClimbGen(vocab, 8, 1, 2, 16, config["maxLen"])
    grade_model = grader.Grader(len(placement_ids), placement_dim=4, role_dim=2, hidden=8)

    with tempfile.TemporaryDirectory() as work:
        out = Path(work)
        train.export(model, config, placement_ids, coords, out)
        grader.export(grade_model, np.zeros(grader.NUM_GRADES), 1.0, out)
        DEST.mkdir(parents=True, exist_ok=True)
        (DEST / "model-meta.json").write_bytes((out / "model-meta.generated.json").read_bytes())
        (DEST / "grader.json").write_bytes((out / "grader.generated.json").read_bytes())
        encoded = re.search(r"'([A-Za-z0-9+/=]+)'", (out / "model-weights.generated.ts").read_text()).group(1)
        (DEST / "weights.bin").write_bytes(base64.b64decode(encoded))
    for path in sorted(DEST.iterdir()):
        print(f"{path.name}: {path.stat().st_size / 1e3:.1f} kB")


if __name__ == "__main__":
    main()
