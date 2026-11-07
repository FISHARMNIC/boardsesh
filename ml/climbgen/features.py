"""Geometry of a climb's hand moves, in board units (inches on Kilter, 4-inch grid).

A climb is a list of (placement index, role index) with role 0 start, 1 hand,
2 finish, 3 foot. Hands (roles 0-2) are walked bottom to top, then left to right.

- max_step: the longest straight-line distance between consecutive hands
- max_vgap: the biggest height gain between consecutive hands
- unsupported: hand holds above the starts with no other hold (any role) 20-70
  units below and within 30 sideways, i.e. no plausible foot for that move

These do not model a body; they catch moves rarer than nearly every real climb.
`thresholds` holds the real-classic 99th percentile per grade band and angle band;
the generator rejects a candidate that exceeds any of them.
"""

from __future__ import annotations

import math

import numpy as np

FEATURES = ("max_step", "max_vgap", "unsupported")
GRADE_BANDS = ((0, 4), (5, 7), (8, 11), (12, 15))
ANGLE_BANDS = ((0, 30), (35, 45), (50, 70))
FOOT_BELOW = (20.0, 70.0)
FOOT_SIDEWAYS = 30.0


def climb_features(holds, xy) -> tuple[float, float, int]:
    """`xy` is a sequence of (x, y) indexed by placement index."""
    points = [(xy[p][0], xy[p][1], r) for p, r in holds]
    hands = sorted(((x, y, r) for x, y, r in points if r != 3), key=lambda point: (point[1], point[0]))
    max_step = max((math.dist(a[:2], b[:2]) for a, b in zip(hands, hands[1:])), default=0.0)
    max_vgap = max((b[1] - a[1] for a, b in zip(hands, hands[1:])), default=0.0)
    unsupported = 0
    for x, y, role in hands:
        if role == 0:
            continue
        supported = any(
            FOOT_BELOW[0] <= y - other_y <= FOOT_BELOW[1] and abs(x - other_x) <= FOOT_SIDEWAYS
            for other_x, other_y, _ in points
            if (other_x, other_y) != (x, y)
        )
        unsupported += not supported
    return float(max_step), float(max_vgap), unsupported


def band_key(v_grade: int, angle: int) -> str:
    grade_band = next(i for i, (low, high) in enumerate(GRADE_BANDS) if low <= v_grade <= high)
    angle_band = next(i for i, (low, high) in enumerate(ANGLE_BANDS) if low <= angle <= high)
    return f"{grade_band}-{angle_band}"


def thresholds(examples, xy, percentile: float = 99.0) -> dict:
    """Per band, the `percentile` of each feature over the given (real classic) climbs."""
    by_band: dict[str, list[tuple[float, float, int]]] = {}
    for example in examples:
        by_band.setdefault(band_key(example.v_grade, example.angle), []).append(climb_features(example.holds, xy))
    return {
        "percentile": percentile,
        "gradeBands": GRADE_BANDS,
        "angleBands": ANGLE_BANDS,
        "bands": {
            key: [round(float(np.percentile([row[i] for row in rows], percentile)), 2) for i in range(len(FEATURES))]
            for key, rows in sorted(by_band.items())
            if len(rows) >= 50
        },
    }


def exceeds(features: tuple[float, float, int], limits: dict, v_grade: int, angle: int) -> bool:
    band = limits["bands"].get(band_key(v_grade, angle))
    return band is not None and any(value > limit for value, limit in zip(features, band))


def main():
    import argparse
    import json
    import sys
    import zlib
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import train

    parser = argparse.ArgumentParser(description="Write the real-classic geometry limits for one board")
    parser.add_argument("--model", default="kilter-1")
    parser.add_argument("--climbs", type=Path, default=Path("data/kilter1.db"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.db"))
    parser.add_argument("--out", type=Path, default=Path("out/v3/geometry.json"))
    args = parser.parse_args()
    train.configure(args.model)
    placement_ids, coords = train.load_placements(args.catalog)
    xy = [coords[pid] for pid in placement_ids]
    examples = train.load_examples(args.climbs, placement_ids, coords)
    # Classic climbs from the training split only, so the held-out climbs stay untouched.
    reference = [e for e in examples if e.classic and zlib.crc32(e.climb_uuid.encode()) % 20 > 1]
    limits = thresholds(reference, xy)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(limits, indent=2) + "\n")
    print(f"{len(reference)} classic climbs -> {len(limits['bands'])} bands, wrote {args.out}")
    for key, values in limits["bands"].items():
        print(f"  band {key}: " + ", ".join(f"{name} {value}" for name, value in zip(FEATURES, values)))


if __name__ == "__main__":
    main()
