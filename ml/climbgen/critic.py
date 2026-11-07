"""v3 realism critic: P(this climb is a real classic) for Kilter Original.

Positives: real classic climbs (train.is_classic). Negatives:
- climbs sampled from the v3 writer with the app's exact decoding rules and
  sampling settings, at the positives' grade/angle mix, with near-copies of real
  climbs (Jaccard >= 0.6) removed so the same climb is never labelled both ways
- real climbs rated below the board's low-rated cutoff (train.LOW_RATED_STARS; 2.25 on Kilter) with at least 10 ascents

Splits by climb as everywhere else (crc32 % 20: 0 test, 1 calibration, rest
train); generated negatives for each split come from separate sampling runs.

Gates printed at the end: held-out AUC (real classic vs generated, and real
classic vs real low-quality) next to a logistic-regression baseline on simple
statistics. The critic must beat the baseline by >= 0.05 AUC on generated
negatives, or it is only spotting artefacts like hold counts.

The export records the writer's sha256: the app refuses a critic paired with a
different writer, because a critic is a discriminator for one writer.

Usage (from ml/climbgen/):
  .venv/bin/python -I -u critic.py --writer data/v3-writer.pt --out out/v3/kilter-1-critic
  .venv/bin/python -I -u critic.py --model tension-9 --climbs data/tension9.db --writer data/v3-tension-9-writer.pt \
    --out out/v3/tension-9-critic --checkpoint data/v3-tension-9-critic.pt
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import grader as grader_module  # noqa: E402
import pipeline  # noqa: E402
import train  # noqa: E402

# Negatives are sampled the way the app samples (V3_PIPELINE in generate.ts). The
# Kilter Original critic was trained at 0.85 / 0.9, before the app moved to 0.9 / 1.0.
TEMPERATURE, TOP_P = 0.9, 1.0


def split_of(uuid: str) -> str:
    bucket = zlib.crc32(uuid.encode()) % 20
    return "test" if bucket == 0 else "calibration" if bucket == 1 else "train"


def generate_negatives(writer, positives, count, rank, placements, known, dev, seed,
                       temperature=TEMPERATURE, top_p=TOP_P):
    """`count` generated climbs at the positives' (angle, grade) mix, minus near-copies of real climbs."""
    rng = random.Random(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    out = []
    allowed = np.ones(placements, dtype=bool)
    while len(out) < count:
        picks = [rng.choice(positives) for _ in range(1024)]
        prefixes = [pipeline.prefix(p.angle, p.v_grade, 1) for p in picks]
        climbs = pipeline.sample(writer, prefixes, rank, allowed, len(train.ROLES), temperature, top_p, dev,
                                 generator if dev == "cpu" else None)
        for pick, climb in zip(picks, climbs):
            if climb is not None and known.max_jaccard(climb) < 0.6:
                out.append(train.Example("generated", pick.angle, pick.v_grade, 0.0, [], climb))
    return out[:count]


def auc(scores_positive: np.ndarray, scores_negative: np.ndarray) -> float:
    scores = np.concatenate([scores_positive, scores_negative])
    ranks = scores.argsort().argsort() + 1
    positive_ranks = ranks[: len(scores_positive)].sum()
    n_pos, n_neg = len(scores_positive), len(scores_negative)
    return float((positive_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def simple_features(example, xy, popularity) -> list[float]:
    roles = [r for _, r in example.holds]
    ys = [xy[p][1] for p, _ in example.holds]
    xs = [xy[p][0] for p, _ in example.holds]
    step, vgap, unsupported = features.climb_features(example.holds, xy)
    return [
        len(roles), roles.count(0), roles.count(1), roles.count(2), roles.count(3),
        step, vgap, unsupported, float(np.mean(ys)), max(ys) - min(ys), max(xs) - min(xs),
        float(np.mean(np.abs(np.array(xs) - 72))), example.angle / 70, example.v_grade / 15,
        float(np.mean([popularity[p] for p, _ in example.holds])),
    ]


def logistic_baseline(train_rows, train_labels, test_sets):
    x = torch.tensor(train_rows, dtype=torch.float32)
    mean, std = x.mean(0), x.std(0) + 1e-6
    x = (x - mean) / std
    y = torch.tensor(train_labels, dtype=torch.float32)
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + 1e-3 * w.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return [((torch.tensor(rows, dtype=torch.float32) - mean) / std @ w + b).numpy() for rows in test_sets]


@torch.no_grad()
def logits_for(model, examples, coords, placement_ids, dev):
    out = []
    for start in range(0, len(examples), 4096):
        batch = grader_module.tensors_for(examples[start:start + 4096], coords, placement_ids, dev)
        out.append(model(*batch[:5]).squeeze(-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def load_critic(checkpoint: Path, export_json: Path, placements: int, dev: str):
    exported = json.loads(export_json.read_text())
    model = grader_module.AttentionGrader(placements, outputs=1)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(dev).eval(), float(exported["temperature"])


def log_p_real(critic, climbs, angles, coords, placement_ids, dev) -> np.ndarray:
    model, temperature = critic
    examples = [train.Example("x", angle, 0, 0.0, [], holds) for holds, angle in zip(climbs, angles)]
    logits = logits_for(model, examples, coords, placement_ids, dev)
    return -np.logaddexp(0, -logits / temperature)  # log sigmoid


def export(model, temperature: float, writer_sha256: str, out: Path):
    tensors = {}
    for name, param in model.state_dict().items():
        array = param.detach().float().cpu().numpy().astype(np.float32)
        tensors[name] = {"shape": list(array.shape), "base64": base64.b64encode(array.tobytes()).decode()}
    payload = {
        "kind": "attention-critic",
        "key": train.MODEL_KEY,
        "writerSha256": writer_sha256,
        "maxAngle": grader_module.MAX_ANGLE,
        "edges": dict(train.EDGES),
        "temperature": temperature,
        "tensors": tensors,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "critic.generated.json"
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kilter-1")
    parser.add_argument("--writer", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument("--writer-weights", type=Path, default=None,
                        help="the exported model-weights.generated.ts the app downloads; its decoded blob's sha256 "
                             "pairs critic and writer. Defaults to <writer export dir>/model-weights.generated.ts")
    parser.add_argument("--climbs", type=Path, default=Path("data/kilter1.db"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog.db"))
    parser.add_argument("--out", type=Path, default=Path("out/v3/kilter-1-critic"))
    parser.add_argument("--checkpoint", type=Path, default=Path("data/v3-critic.pt"))
    parser.add_argument("--generated", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    train.configure(args.model)
    dev = pipeline.device()
    random.seed(2)
    torch.manual_seed(2)
    placement_ids, coords = train.load_placements(args.catalog)
    xy = [coords[pid] for pid in placement_ids]
    examples = train.load_examples(args.climbs, placement_ids, coords)
    known = pipeline.known_climbs(examples, len(placement_ids))
    rank = pipeline.order_rank(placement_ids, coords)
    writer, _ = pipeline.load_writer(args.writer, dev)

    popularity = np.zeros(len(placement_ids))
    for example in examples:
        for p, _ in example.holds:
            popularity[p] += 1
    popularity /= popularity.max()

    splits: dict[str, dict[str, list]] = {name: {"pos": [], "low": []} for name in ("train", "calibration", "test")}
    for example in examples:
        bucket = splits[split_of(example.climb_uuid)]
        if example.classic:
            bucket["pos"].append(example)
        elif example.quality is not None and example.quality < train.LOW_RATED_STARS and example.ascents >= 10:
            bucket["low"].append(example)
    sizes = {"train": args.generated, "calibration": args.generated // 18, "test": args.generated // 18}
    for seed, (name, bucket) in enumerate(splits.items()):
        bucket["gen"] = generate_negatives(writer, bucket["pos"], sizes[name], rank, len(placement_ids), known, dev,
                                           seed + 10, args.temperature, args.top_p)
        print(f"{name}: {len(bucket['pos'])} classic, {len(bucket['gen'])} generated, {len(bucket['low'])} low-rated")

    train_rows = [(e, 1.0) for e in splits["train"]["pos"]] + [(e, 0.0) for e in splits["train"]["gen"]] + \
        [(e, 0.0) for e in splits["train"]["low"]]
    positives = sum(label for _, label in train_rows)
    pos_weight = torch.tensor((len(train_rows) - positives) / positives, device=dev)

    model = grader_module.AttentionGrader(len(placement_ids), outputs=1).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(train_rows)
        total, batches = 0.0, 0
        for start in range(0, len(train_rows), 512):
            chunk = train_rows[start:start + 512]
            batch = grader_module.tensors_for([e for e, _ in chunk], coords, placement_ids, dev)
            labels = torch.tensor([label for _, label in chunk], device=dev)
            loss = F.binary_cross_entropy_with_logits(model(*batch[:5]).squeeze(-1), labels, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        schedule.step()
        model.eval()
        test_auc = auc(logits_for(model, splits["test"]["pos"], coords, placement_ids, dev),
                       logits_for(model, splits["test"]["gen"], coords, placement_ids, dev))
        print(f"epoch {epoch + 1}: loss {total / batches:.3f}  test AUC (classic vs generated) {test_auc:.3f}")
    torch.save(model.state_dict(), args.checkpoint)

    # Calibrate on the calibration split with its true class mix.
    cal = splits["calibration"]
    cal_logits = np.concatenate([logits_for(model, cal[k], coords, placement_ids, dev) for k in ("pos", "gen", "low")])
    cal_labels = np.concatenate([np.ones(len(cal["pos"])), np.zeros(len(cal["gen"]) + len(cal["low"]))])
    best, temperature = float("inf"), 1.0
    for candidate in np.arange(0.5, 5.01, 0.05):
        nll = float(np.mean(np.logaddexp(0, -(2 * cal_labels - 1) * cal_logits / candidate)))
        if nll < best:
            best, temperature = nll, round(float(candidate), 2)
    print(f"temperature {temperature} (calibration NLL {best:.3f})")

    test = splits["test"]
    scores = {k: logits_for(model, test[k], coords, placement_ids, dev) for k in ("pos", "gen", "low")}
    rows = {k: [simple_features(e, xy, popularity) for e in v] for k, v in splits["train"].items()}
    baseline_train = rows["pos"] + rows["gen"] + rows["low"]
    baseline_labels = [1] * len(rows["pos"]) + [0] * (len(rows["gen"]) + len(rows["low"]))
    test_rows = {k: [simple_features(e, xy, popularity) for e in v] for k, v in test.items()}
    base_pos, base_gen, base_low = logistic_baseline(baseline_train, baseline_labels,
                                                     [test_rows["pos"], test_rows["gen"], test_rows["low"]])
    critic_gen, base_gen_auc = auc(scores["pos"], scores["gen"]), auc(base_pos, base_gen)
    critic_low, base_low_auc = auc(scores["pos"], scores["low"]), auc(base_pos, base_low)
    print("\nheld-out AUC        critic   baseline")
    print(f"classic vs generated {critic_gen:.3f}    {base_gen_auc:.3f}")
    print(f"classic vs low-rated {critic_low:.3f}    {base_low_auc:.3f}")
    gate = critic_gen - base_gen_auc >= 0.05
    print(f"gate (critic beats baseline by >= 0.05 on generated): {'PASS' if gate else 'FAIL'}")

    weights_ts = args.writer_weights or (args.writer.parent.parent / f"out/v3/{args.model}/model-weights.generated.ts")
    if not weights_ts.exists():
        raise SystemExit(f"need the writer's exported weights to pair the critic: {weights_ts}")
    import re
    blob = base64.b64decode(re.search(r"'([A-Za-z0-9+/=]+)'", weights_ts.read_text()).group(1))
    export(model, temperature, hashlib.sha256(blob).hexdigest(), args.out)
    summary = {"aucGenerated": critic_gen, "aucLowRated": critic_low, "baselineGenerated": base_gen_auc,
               "baselineLowRated": base_low_auc, "temperature": temperature, "gatePassed": gate}
    (args.out / "critic-report.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
