"""Scorecard for a generation pipeline on Kilter Original: the v3 ship gates.

For each target grade (V3, V5, V7, V9, V11 at 40 degrees) it runs `--taps`
simulated Generate taps through the full pipeline (sample -> geometry filter ->
near-copy filter -> rank -> keep) and reports, over the climbs a tap shows:

- hit: the JUDGE grader's median within 1 V of the target (the judge is fixed,
  by default the v2 grader, so v2 and v3 are scored the same way)
- KS distance of each geometry feature to real classic climbs at that grade and angle
- copies: share of candidates with an existing climb at Jaccard >= 0.8 (and exact),
  before filtering, and among the shown climbs after it
- variety: mean pairwise Jaccard among one tap's shown climbs (lower = more varied)
- rejected: share of candidates the geometry safety net threw out

Usage (from ml/climbgen/):
  .venv/bin/python -I -u evaluate_v3.py --writer data/climbgen-v2.pt --label v2 \
     --rank-grader data/grader.pt --rank-grader-json out/kilter-1/grader.generated.json \
     --temperature 0.9 --top-p 1.0 --candidates 24
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import pipeline  # noqa: E402
import train  # noqa: E402

TARGETS = (3, 5, 7, 9, 11)
ANGLE = 40


def ks_distance(sample: np.ndarray, reference: np.ndarray) -> float:
    if len(sample) == 0 or len(reference) == 0:
        return float("nan")
    values = np.union1d(sample, reference)
    cdf_a = np.searchsorted(np.sort(sample), values, side="right") / len(sample)
    cdf_b = np.searchsorted(np.sort(reference), values, side="right") / len(reference)
    return float(np.abs(cdf_a - cdf_b).max())


def jaccard(a, b) -> float:
    sa, sb = {p for p, _ in a}, {p for p, _ in b}
    return len(sa & sb) / max(len(sa | sb), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--writer", type=Path, required=True)
    parser.add_argument("--rank-grader", type=Path, required=True)
    parser.add_argument("--rank-grader-json", type=Path, required=True)
    parser.add_argument("--judge-grader", type=Path, default=Path("data/grader.pt"))
    parser.add_argument("--judge-grader-json", type=Path, default=Path("out/kilter-1/grader.generated.json"))
    parser.add_argument("--critic", type=Path, default=None, help="critic checkpoint (critic.py); omit for none")
    parser.add_argument("--critic-json", type=Path, default=None)
    parser.add_argument("--critic-weight", type=float, default=0.0)
    parser.add_argument("--quality", type=int, choices=[0, 1], default=None, help="1 = ask for classic")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--keep", type=int, default=6)
    parser.add_argument("--taps", type=int, default=100)
    parser.add_argument("--geometry", type=Path, default=None, help="thresholds JSON; omit to skip the safety net")
    parser.add_argument("--report-geometry-only", action="store_true",
                        help="measure what the safety net would reject without filtering (v2 baseline)")
    parser.add_argument("--reject-copies", action="store_true")
    parser.add_argument("--climbs", type=Path, default=Path("data/kilter1.db"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.db"))
    parser.add_argument("--out", type=Path, default=None, help="write the scorecard JSON here")
    args = parser.parse_args()

    train.configure("kilter-1")
    dev = pipeline.device()
    torch.manual_seed(0)
    placement_ids, coords = train.load_placements(args.catalog)
    xy = [coords[pid] for pid in placement_ids]
    examples = train.load_examples(args.climbs, placement_ids, coords)
    known = pipeline.known_climbs(examples, len(placement_ids))
    rank = pipeline.order_rank(placement_ids, coords)
    allowed = np.ones(len(placement_ids), dtype=bool)

    writer, writer_config = pipeline.load_writer(args.writer, dev)
    rank_grader = pipeline.load_grader(args.rank_grader, args.rank_grader_json, len(placement_ids), dev)
    judge = pipeline.load_grader(args.judge_grader, args.judge_grader_json, len(placement_ids), dev)
    critic = None
    if args.critic:
        import critic as critic_module
        critic = critic_module.load_critic(args.critic, args.critic_json, len(placement_ids), dev)
    limits = json.loads(args.geometry.read_text()) if args.geometry else None

    report = {"label": args.label, "config": {k: str(v) for k, v in vars(args).items()}, "grades": {}}
    print(f"[{args.label}] writer quality token: {writer_config.get('qualityToken', False)}")
    print("target | hit±1 | KS step / vgap / unsup | copies cand (exact) | copies shown | variety | rejected")
    for target in TARGETS:
        prefix = pipeline.prefix(ANGLE, target, args.quality)
        candidates = pipeline.sample(writer, [prefix] * (args.taps * args.candidates), rank, allowed, len(train.ROLES),
                                     args.temperature, args.top_p, dev)
        valid = [c for c in candidates if c is not None]
        rank_probs = pipeline.grade_probabilities(*rank_grader, valid, [ANGLE] * len(valid), coords, placement_ids, dev)
        critic_logp = (critic_module.log_p_real(critic, valid, [ANGLE] * len(valid), coords, placement_ids, dev)
                       if critic else np.zeros(len(valid)))
        geometry = [features.climb_features(c, xy) for c in valid]
        copy_score = np.array([known.max_jaccard(c) for c in valid])
        rejected = np.array([bool(limits) and features.exceeds(g, limits, target, ANGLE) for g in geometry])

        shown, varieties = [], []
        per_tap = len(valid) // args.taps
        for tap in range(args.taps):
            rows = range(tap * per_tap, (tap + 1) * per_tap)
            seen, pool = set(), []
            for row in rows:
                key = tuple(sorted(valid[row]))
                filtered = rejected[row] and not args.report_geometry_only
                if key in seen or filtered or (args.reject_copies and copy_score[row] >= 0.8):
                    continue
                seen.add(key)
                score = np.log(rank_probs[row, target] + 1e-9) + args.critic_weight * max(critic_logp[row], -5.0)
                pool.append((score, row))
            picked = [row for _, row in sorted(pool, reverse=True)[: args.keep]]
            shown += picked
            if len(picked) > 1:
                varieties.append(np.mean([jaccard(valid[a], valid[b]) for a, b in itertools.combinations(picked, 2)]))

        shown_climbs = [valid[row] for row in shown]
        judge_median = pipeline.median_grade(
            pipeline.grade_probabilities(*judge, shown_climbs, [ANGLE] * len(shown), coords, placement_ids, dev))
        hit = float(np.mean(np.abs(judge_median - target) <= 1)) if shown else float("nan")
        reference = [features.climb_features(e.holds, xy) for e in examples
                     if e.v_grade == target and e.angle == ANGLE and e.classic]
        ks = [ks_distance(np.array([geometry[row][i] for row in shown]), np.array([r[i] for r in reference]))
              for i in range(len(features.FEATURES))]
        result = {
            "hit": hit,
            "ks": ks,
            "copyCandidates": float(np.mean(copy_score >= 0.8)),
            "exactCandidates": float(np.mean(copy_score >= 0.999)),
            "copyShown": float(np.mean(copy_score[shown] >= 0.8)) if shown else float("nan"),
            "variety": float(np.mean(varieties)) if varieties else float("nan"),
            "rejected": float(rejected.mean()),
            "valid": len(valid) / (args.taps * args.candidates),
            "shownPerTap": len(shown) / args.taps,
        }
        report["grades"][f"V{target}"] = result
        print(f"V{target:<5} | {hit:5.2f} | {ks[0]:.2f} / {ks[1]:.2f} / {ks[2]:.2f} | "
              f"{result['copyCandidates']:5.1%} ({result['exactCandidates']:4.1%}) | {result['copyShown']:5.1%} | "
              f"{result['variety']:.3f} | {result['rejected']:5.1%}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
