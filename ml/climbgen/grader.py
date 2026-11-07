"""Climb grader for the on-device generator: predicts a probability for each V-grade.

A small Deep Sets model (per-hold MLP -> mean/max/sum pool -> head), the same
family as ml/climb2vec, trained on the public Kilter Original snapshot.

- Hard grades are rare, so training weights each grade by 1/sqrt(frequency).
  That weighting is undone exactly at export (subtract log(weight) from each
  logit), so the shipped probabilities reflect real grade frequencies.
- A temperature fitted on a calibration split makes the probabilities honest,
  so an "80% range" really contains the true grade about 80% of the time.
- The acceptance report is computed on a third, untouched test split.

Usage (from ml/climbgen/):
  .venv/bin/python -I -u grader.py --climbs data/kilter1.db --catalog data/catalog.db --out out
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402

NUM_GRADES = train.MAX_V + 1
COVERAGE = 0.8
MAX_ANGLE = 70.0


def split_of(uuid: str) -> str:
    bucket = zlib.crc32(uuid.encode()) % 20
    return "test" if bucket == 0 else "calibration" if bucket == 1 else "train"


class Grader(nn.Module):
    def __init__(self, placements: int, placement_dim: int = 16, role_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.placement = nn.Embedding(placements, placement_dim)
        self.role = nn.Embedding(4, role_dim)
        node_in = placement_dim + role_dim + 3  # + x, y, angle
        self.node1 = nn.Linear(node_in, hidden)
        self.node2 = nn.Linear(hidden, hidden)
        self.head1 = nn.Linear(3 * hidden + 2, hidden)  # + hold count, angle
        self.head2 = nn.Linear(hidden, NUM_GRADES)

    def forward(self, placement, role, xy, angle, mask):
        # placement/role: [B, N]; xy: [B, N, 2] in 0..1; angle: [B] in 0..1; mask: [B, N] float
        batch, holds = placement.shape
        angle_per_hold = angle[:, None, None].expand(batch, holds, 1)
        nodes = torch.cat([self.placement(placement), self.role(role), xy, angle_per_hold], dim=-1)
        nodes = F.relu(self.node2(F.relu(self.node1(nodes))))
        weights = mask[:, :, None]
        count = weights.sum(dim=1).clamp(min=1)
        mean = (nodes * weights).sum(dim=1) / count
        maximum = (nodes * weights + (weights - 1) * 1e4).amax(dim=1)
        total = (nodes * weights).sum(dim=1) / 10.0
        pooled = torch.cat([mean, maximum, total, count / 30.0, angle[:, None]], dim=-1)
        return self.head2(F.relu(self.head1(pooled)))


class EncoderBlock(nn.Module):
    """Pre-LN self-attention block over a climb's holds; padding holds are masked out."""

    def __init__(self, dim: int, heads: int, ff: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ff)
        self.fc2 = nn.Linear(ff, dim)

    def forward(self, x, keep):
        batch, holds, dim = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(dim, dim=2)
        shape = (batch, holds, self.heads, dim // self.heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        x = x + self.proj(attn.transpose(1, 2).reshape(batch, holds, dim))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class AttentionGrader(nn.Module):
    """v3: every hold attends to every other before pooling, so moves between holds are visible.

    Same per-hold input as Grader; `outputs` is 16 grade logits, or 1 for the critic.
    """

    kind = "attention"

    def __init__(self, placements: int, placement_dim: int = 24, role_dim: int = 8, dim: int = 64,
                 layers: int = 2, heads: int = 4, outputs: int = NUM_GRADES):
        super().__init__()
        self.placement = nn.Embedding(placements, placement_dim)
        self.role = nn.Embedding(4, role_dim)
        self.inp = nn.Linear(placement_dim + role_dim + 3, dim)
        self.blocks = nn.ModuleList(EncoderBlock(dim, heads, dim * 2) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.head1 = nn.Linear(3 * dim + 2, dim)
        self.head2 = nn.Linear(dim, outputs)

    def forward(self, placement, role, xy, angle, mask):
        batch, holds = placement.shape
        angle_per_hold = angle[:, None, None].expand(batch, holds, 1)
        x = self.inp(torch.cat([self.placement(placement), self.role(role), xy, angle_per_hold], dim=-1))
        keep = mask > 0
        for block in self.blocks:
            x = block(x, keep)
        x = self.ln(x)
        weights = mask[:, :, None]
        count = weights.sum(dim=1).clamp(min=1)
        mean = (x * weights).sum(dim=1) / count
        maximum = (x * weights + (weights - 1) * 1e4).amax(dim=1)
        total = (x * weights).sum(dim=1) / 10.0
        pooled = torch.cat([mean, maximum, total, count / 30.0, angle[:, None]], dim=-1)
        return self.head2(F.relu(self.head1(pooled)))


def tensors_for(examples, coords, placement_ids, device):
    width = max(len(e.holds) for e in examples)
    placement = torch.zeros((len(examples), width), dtype=torch.long)
    role = torch.zeros((len(examples), width), dtype=torch.long)
    xy = torch.zeros((len(examples), width, 2))
    mask = torch.zeros((len(examples), width))
    for row, example in enumerate(examples):
        for column, (placement_index, role_index) in enumerate(example.holds):
            x, y = coords[placement_ids[placement_index]]
            placement[row, column] = placement_index
            role[row, column] = role_index
            edges = train.EDGES
            xy[row, column, 0] = (x - edges["left"]) / (edges["right"] - edges["left"])
            xy[row, column, 1] = (y - edges["bottom"]) / (edges["top"] - edges["bottom"])
            mask[row, column] = 1
    angle = torch.tensor([e.angle / MAX_ANGLE for e in examples])
    grade = torch.tensor([e.v_grade for e in examples])
    return [t.to(device) for t in (placement, role, xy, angle, mask, grade)]


def predict_logits(model, examples, coords, placement_ids, device, batch_size=4096):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            placement, role, xy, angle, mask, _ = tensors_for(examples[start : start + batch_size], coords, placement_ids, device)
            outputs.append(model(placement, role, xy, angle, mask).cpu())
    return torch.cat(outputs)


def shortest_range(probabilities: np.ndarray, coverage: float | None = None) -> tuple[int, int]:
    """Narrowest run of neighbouring grades holding at least `coverage` of the mass."""
    coverage = COVERAGE if coverage is None else coverage
    best = (0, len(probabilities) - 1)
    for low in range(len(probabilities)):
        total = 0.0
        for high in range(low, len(probabilities)):
            total += probabilities[high]
            if total >= coverage:
                if high - low < best[1] - best[0]:
                    best = (low, high)
                break
    return best


def median_grade(probabilities: np.ndarray) -> int:
    return int(np.searchsorted(np.cumsum(probabilities), 0.5))


def report(name, probabilities, grades):
    medians = np.array([median_grade(p) for p in probabilities])
    ranges = [shortest_range(p) for p in probabilities]
    inside = np.array([low <= g <= high for (low, high), g in zip(ranges, grades)])
    widths = np.array([high - low + 1 for low, high in ranges])
    print(f"\n[{name}] median-grade MAE {np.abs(medians - grades).mean():.2f} V | "
          f"range ({COVERAGE:.0%} mass) covers the true grade {inside.mean() * 100:.1f}% | mean range width {widths.mean():.2f} grades")
    print("true V | climbs | avg median guess | range covers | avg width")
    for grade in range(NUM_GRADES):
        rows = grades == grade
        if rows.sum() == 0:
            continue
        print(f"V{grade:<4} | {rows.sum():>6} | V{medians[rows].mean():>4.1f} | {inside[rows].mean() * 100:5.1f}% | {widths[rows].mean():.2f}")
    bands = {"V0-V4": (0, 4), "V5-V7": (5, 7), "V8-V11": (8, 11), "V12+": (12, 15)}
    print("band   | range covers")
    for band, (low, high) in bands.items():
        rows = (grades >= low) & (grades <= high)
        if rows.sum():
            print(f"{band:<6} | {inside[rows].mean() * 100:5.1f}%  (n={rows.sum()})")


def export(model, logit_correction, temperature, out: Path):
    tensors = {}
    for name, param in model.state_dict().items():
        array = param.detach().float().cpu().numpy().astype(np.float32)
        tensors[name] = {"shape": list(array.shape), "base64": base64.b64encode(array.tobytes()).decode()}
    payload = {
        "kind": getattr(model, "kind", "deepsets"),
        "numGrades": NUM_GRADES,
        "coverage": COVERAGE,
        "maxAngle": MAX_ANGLE,
        "key": train.MODEL_KEY,
        "edges": dict(train.EDGES),
        "temperature": temperature,
        "logitCorrection": [round(float(value), 6) for value in logit_correction],
        "tensors": tensors,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "grader.generated.json"
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"\nwrote {path} ({path.stat().st_size / 1e3:.0f} kB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(train.BOARD_MODELS), default="kilter-1")
    parser.add_argument("--climbs", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--prior", choices=["catalogue", "flat"], default="flat",
                        help="catalogue: grade frequencies as in the Kilter catalogue; flat: judge each climb on its "
                             "holds alone (right for generated climbs, which already aim at a grade)")
    parser.add_argument("--checkpoint", type=Path, default=Path("data/grader.pt"))
    parser.add_argument("--coverage", type=float, default=0.75,
                        help="probability mass the shown range holds; discrete grades overshoot it, "
                             "so 0.75 gives ranges that contain the true grade at least 80%% of the time per grade band")
    parser.add_argument("--eval-only", action="store_true", help="reuse --checkpoint instead of training")
    parser.add_argument("--arch", choices=["deepsets", "attention"], default="deepsets",
                        help="attention (v3): holds see each other before pooling")
    args = parser.parse_args()

    train.configure(args.model)
    random.seed(1)
    torch.manual_seed(1)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    placement_ids, coords = train.load_placements(args.catalog)
    examples = train.load_examples(args.climbs, placement_ids, coords)
    splits = collections.defaultdict(list)
    for example in examples:
        splits[split_of(example.climb_uuid)].append(example)
    print({name: len(rows) for name, rows in splits.items()})

    counts = np.bincount([e.v_grade for e in splits["train"]], minlength=NUM_GRADES).astype(np.float64)
    class_weight = np.where(counts > 0, (counts.max() / np.maximum(counts, 1)) ** 0.5, 1.0)
    class_weight = np.minimum(class_weight, 10.0)
    weight_tensor = torch.tensor(class_weight, dtype=torch.float32, device=device)
    print("class weights", np.round(class_weight, 2).tolist())

    global COVERAGE
    COVERAGE = args.coverage
    model = (AttentionGrader(len(placement_ids)) if args.arch == "attention" else Grader(len(placement_ids))).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    training = splits["train"]
    if args.eval_only:
        model.load_state_dict(torch.load(args.checkpoint))
    for epoch in range(0 if args.eval_only else args.epochs):
        model.train()
        random.shuffle(training)
        total, batches = 0.0, 0
        for start in range(0, len(training), 512):
            placement, role, xy, angle, mask, grade = tensors_for(training[start : start + 512], coords, placement_ids, device)
            loss = F.cross_entropy(model(placement, role, xy, angle, mask), grade, weight=weight_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        schedule.step()
        if epoch % 5 == 4 or epoch == args.epochs - 1:
            print(f"epoch {epoch + 1}: train loss {total / batches:.3f}")

    model.eval()
    if not args.eval_only:
        torch.save(model.state_dict(), args.checkpoint)
    # The weighted model learns p(v|x)·w_v. Subtracting log w_v gives the catalogue posterior p(v|x);
    # also subtracting log p(v) removes the grade frequencies, leaving how well the holds fit each grade.
    prior = np.maximum(counts / counts.sum(), 1e-6)
    corrections = {"catalogue": -np.log(class_weight), "flat": -np.log(class_weight) - np.log(prior)}

    calibration = splits["calibration"]
    calibration_raw = predict_logits(model, calibration, coords, placement_ids, device)
    calibration_grades = torch.tensor([e.v_grade for e in calibration])
    test = splits["test"]
    test_raw = predict_logits(model, test, coords, placement_ids, device)
    test_grades = np.array([e.v_grade for e in test])

    chosen = None
    for name, logit_correction in corrections.items():
        correction = torch.tensor(logit_correction, dtype=torch.float32)
        temperature, best = 1.0, float("inf")
        for candidate in np.arange(0.5, 3.01, 0.05):
            nll = F.cross_entropy((calibration_raw + correction) / candidate, calibration_grades).item()
            if nll < best:
                best, temperature = nll, round(float(candidate), 2)
        print(f"\n=== prior: {name} | temperature {temperature} (calibration NLL {best:.3f})")
        report(f"test, {name} prior", F.softmax((test_raw + correction) / temperature, dim=-1).numpy(), test_grades)
        if name == args.prior:
            chosen = (logit_correction, temperature)

    logit_correction, temperature = chosen
    export(model, logit_correction, temperature, args.out)


if __name__ == "__main__":
    main()
