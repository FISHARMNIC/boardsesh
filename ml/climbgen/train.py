"""Prototype: a tiny on-device Kilter climb generator.

Trains a small decoder-only transformer, from the public Boardsesh snapshots
(docs/board-snapshots-dataset.md), that writes a climb hold by hold,
conditioned on [angle, V-grade, keyword tags]. Hard grades are rare, so V8+
climbs are repeated during training (see hard_repeats). The grader that ranks
the samples on the phone is trained separately by grader.py.

It is exported into packages/mobile/src/lib/climb-generator/
(model-meta.generated.json + model-weights.generated.ts). The transformer
is written by hand (pre-LN, GELU, tied embeddings) so the TypeScript port in
packages/mobile/src/lib/climb-generator/transformer.ts can mirror it exactly.

Usage (from repo root):
  python3 -m venv .venv && .venv/bin/pip install torch numpy
  .venv/bin/python ml/climbgen/train.py --climbs kilter1.db --catalog catalog.db
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import re
import sqlite3
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BOARD = "kilter"
LAYOUT_ID = 1
# 12x12 with kickboard: the size most Kilter climbs are set on.
EDGES = {"left": 0, "right": 144, "bottom": 0, "top": 156}
ROLES = [12, 13, 14, 15]  # start, hand, finish, foot
ROLE_START, ROLE_HAND, ROLE_FINISH, ROLE_FOOT = ROLES
ANGLES = list(range(0, 75, 5))
# Kilter difficulty id -> V number, from board_difficulty_grades.
V_BY_DIFFICULTY = {
    10: 0, 11: 0, 12: 0, 13: 1, 14: 1, 15: 2, 16: 3, 17: 3, 18: 4, 19: 4, 20: 5, 21: 5,
    22: 6, 23: 7, 24: 8, 25: 8, 26: 9, 27: 10, 28: 11, 29: 12, 30: 13, 31: 14, 32: 15, 33: 16,
}
MAX_V = 15
MAX_HOLDS = 30

TEXT_TAGS = {
    "no-match": r"no match|no-match|nomatch",
    "campus": r"campus",
    "dyno": r"dyno|jump|deadpoint|lunge",
    "pinch": r"pinch",
    "crimp": r"crimp",
    "jug": r"\bjugs?\b",
    "power": r"power",
    "undercling": r"undercling|underc",
    "swing": r"swing|cut loose|cutloose",
    "endurance": r"endur|pump|stamina",
    "sloper": r"sloper|slopey|slopy",
    "heel": r"heel",
    "toe": r"\btoe",
}
GEOMETRY_TAGS = ["reachy", "traverse", "short", "long"]
TAGS = list(TEXT_TAGS) + GEOMETRY_TAGS

# Vocabulary layout. Exported in the meta JSON; generate.ts reads it from there.
PAD, BOS, SEP, EOS = 0, 1, 2, 3
ANGLE_BASE = 4
GRADE_BASE = ANGLE_BASE + len(ANGLES)
TAG_BASE = GRADE_BASE + MAX_V + 1
HOLD_BASE = TAG_BASE + len(TAGS)


@dataclass
class Example:
    climb_uuid: str
    angle: int
    v_grade: int
    difficulty: float
    tags: list[int]
    holds: list[tuple[int, int]]  # (placement index, role index)


def load_placements(catalog: Path) -> tuple[list[int], dict[int, tuple[int, int]]]:
    db = sqlite3.connect(catalog)
    rows = db.execute(
        """SELECT p.id, h.x, h.y FROM board_placements p
           JOIN board_holes h ON h.board_type = p.board_type AND h.id = p.hole_id
           WHERE p.board_type = ? AND p.layout_id = ? ORDER BY p.id""",
        (BOARD, LAYOUT_ID),
    ).fetchall()
    in_bounds = [
        (pid, x, y)
        for pid, x, y in rows
        if EDGES["left"] < x < EDGES["right"] and EDGES["bottom"] < y < EDGES["top"]
    ]
    return [pid for pid, _, _ in in_bounds], {pid: (x, y) for pid, x, y in in_bounds}


FRAME_TOKEN = re.compile(r"p(\d+)r(\d+)")


def load_examples(climbs_db: Path, placement_ids: list[int], coords: dict[int, tuple[int, int]]):
    index_by_placement = {pid: i for i, pid in enumerate(placement_ids)}
    db = sqlite3.connect(climbs_db)
    rows = db.execute(
        """SELECT c.uuid, c.name, c.description, c.characteristics, c.frames,
                  s.angle, s.display_difficulty
           FROM board_climbs c JOIN board_climb_stats s ON s.climb_uuid = c.uuid
           WHERE c.layout_id = ? AND c.is_listed = 1 AND s.ascensionist_count >= 3
             AND s.display_difficulty IS NOT NULL""",
        (LAYOUT_ID,),
    ).fetchall()

    parsed: dict[str, tuple[list[tuple[int, int]], set[str]] | None] = {}
    examples: list[Example] = []
    for uuid, name, description, characteristics, frames, angle, difficulty in rows:
        if angle not in ANGLES:
            continue
        if uuid not in parsed:
            parsed[uuid] = parse_climb(name, description, characteristics, frames, index_by_placement)
        climb = parsed[uuid]
        if climb is None:
            continue
        holds, text_tags = climb
        v_grade = V_BY_DIFFICULTY.get(round(difficulty))
        if v_grade is None or v_grade > MAX_V:
            continue
        examples.append(
            Example(uuid, angle, v_grade, float(difficulty), sorted(TAGS.index(t) for t in text_tags), holds)
        )

    add_geometry_tags(examples, placement_ids, coords)
    return examples


def parse_climb(name, description, characteristics, frames, index_by_placement):
    if not frames or "," in frames or "x" in frames:
        return None
    holds = []
    seen = set()
    for pid_text, role_text in FRAME_TOKEN.findall(frames):
        pid, role = int(pid_text), int(role_text)
        if role not in ROLES or pid not in index_by_placement or pid in seen:
            return None  # off-board for 12x12, alternate role codes, or duplicates
        seen.add(pid)
        holds.append((index_by_placement[pid], ROLES.index(role)))
    roles = [r for _, r in holds]
    if not (2 <= len(holds) <= MAX_HOLDS) or roles.count(0) not in (1, 2) or roles.count(2) not in (1, 2):
        return None
    text = f"{name or ''} {description or ''}".lower()
    tags = {tag for tag, pattern in TEXT_TAGS.items() if re.search(pattern, text)}
    if characteristics and "no_match" in characteristics:
        tags.add("no-match")
    return holds, tags


def add_geometry_tags(examples: list[Example], placement_ids, coords):
    def geometry(example: Example):
        points = [coords[placement_ids[p]] for p, role in example.holds if role != 3]
        points.sort(key=lambda point: point[1])
        reach = max(
            (math.dist(a, b) for a, b in zip(points, points[1:])),
            default=0.0,
        )
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        return reach, (max(xs) - min(xs)), (max(ys) - min(ys)), len(example.holds)

    stats = [geometry(e) for e in examples]
    reach_cut = np.percentile([s[0] for s in stats], 85)
    count_low = np.percentile([s[3] for s in stats], 15)
    count_high = np.percentile([s[3] for s in stats], 85)
    print(f"geometry cuts: reach>={reach_cut:.1f} holds<={count_low} holds>={count_high}")
    for example, (reach, width, height, count) in zip(examples, stats):
        extra = []
        if reach >= reach_cut:
            extra.append("reachy")
        if width > 1.3 * height:
            extra.append("traverse")
        if count <= count_low:
            extra.append("short")
        if count >= count_high:
            extra.append("long")
        example.tags = sorted(set(example.tags) | {TAGS.index(t) for t in extra})


def hold_order(coords, placement_ids):
    def key(hold):
        x, y = coords[placement_ids[hold[0]]]
        return (y, x)

    return key


def encode(example: Example, order_key, tag_keep: float) -> list[int]:
    tags = [t for t in example.tags if random.random() < tag_keep]
    tokens = [BOS, ANGLE_BASE + ANGLES.index(example.angle), GRADE_BASE + example.v_grade]
    tokens += [TAG_BASE + t for t in tags]
    tokens.append(SEP)
    tokens += [HOLD_BASE + p * 4 + r for p, r in sorted(example.holds, key=order_key)]
    tokens.append(EOS)
    return tokens


def hard_repeats(v_grade: int) -> int:
    """How many times a climb appears per epoch: hard grades are rare in the data."""
    if v_grade >= 12:
        return 4
    if v_grade >= 10:
        return 3
    if v_grade >= 8:
        return 2
    return 1


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, dim, heads, ff):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ff)
        self.fc2 = nn.Linear(ff, dim)

    def forward(self, x):
        batch, length, dim = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(dim, dim=2)
        shape = (batch, length, self.heads, dim // self.heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attn.transpose(1, 2).reshape(batch, length, dim))
        # tanh GELU: cheap and exact to port.
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x)), approximate="tanh"))


class ClimbGen(nn.Module):
    def __init__(self, vocab, dim, layers, heads, ff, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList(Block(dim, heads, ff) for _ in range(layers))
        self.ln_f = nn.LayerNorm(dim)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x) @ self.tok.weight.T


# ---------------------------------------------------------------- export
def quantize_rows(tensor: torch.Tensor):
    matrix = tensor.detach().float().cpu().reshape(tensor.shape[0], -1) if tensor.dim() > 1 else tensor.detach().float().cpu().reshape(1, -1)
    scales = matrix.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    quantized = torch.round(matrix / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized.numpy(), scales.numpy().astype(np.float32)


def export(model: ClimbGen, config: dict, placement_ids, coords, out: Path):
    tensors = []
    blob = bytearray()
    for name, param in model.state_dict().items():
        if param.dim() == 2:
            quantized, scales = quantize_rows(param)
            entry = {"name": name, "shape": list(param.shape), "dtype": "int8", "offset": len(blob)}
            blob += quantized.tobytes()
            pad = (-len(blob)) % 4
            blob += b"\0" * pad
            entry["scaleOffset"] = len(blob)
            blob += scales.tobytes()
        else:
            entry = {"name": name, "shape": list(param.shape), "dtype": "float32", "offset": len(blob)}
            blob += param.detach().float().cpu().numpy().astype(np.float32).tobytes()
        tensors.append(entry)

    meta = {
        **config,
        "board": BOARD,
        "layoutId": LAYOUT_ID,
        "roles": ROLES,
        "angles": ANGLES,
        "maxV": MAX_V,
        "tags": TAGS,
        "vocab": {"pad": PAD, "bos": BOS, "sep": SEP, "eos": EOS, "angleBase": ANGLE_BASE,
                  "gradeBase": GRADE_BASE, "tagBase": TAG_BASE, "holdBase": HOLD_BASE},
        "placements": [[pid, *coords[pid]] for pid in placement_ids],
        "tensors": tensors,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "model-meta.generated.json").write_text(json.dumps(meta, separators=(",", ":")) + "\n")
    weights_file = out / "model-weights.generated.ts"
    weights_file.write_text(
        "// Generated by ml/climbgen/train.py. Do not edit.\n"
        "// int8 per-row quantized transformer weights, base64.\n"
        f"export const CLIMB_GEN_WEIGHTS_BASE64 =\n  '{base64.b64encode(bytes(blob)).decode()}';\n"
    )
    print(f"wrote {out} (weights {len(blob) / 1e6:.2f} MB raw, {weights_file.stat().st_size / 1e6:.2f} MB base64)")


# ---------------------------------------------------------------- train
def batches(sequences, batch_size, device):
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        width = max(len(s) for s in chunk)
        tensor = torch.full((len(chunk), width), PAD, dtype=torch.long)
        for row, seq in enumerate(chunk):
            tensor[row, : len(seq)] = torch.tensor(seq)
        yield tensor.to(device)


def lm_loss(model, tokens):
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:].clone()
    # Only score the holds + EOS; the prefix is given.
    sep_pos = (tokens == SEP).float().argmax(dim=1)
    positions = torch.arange(targets.shape[1], device=tokens.device)[None, :]
    targets[positions < sep_pos[:, None]] = PAD
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--climbs", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("packages/mobile/src/lib/climb-generator"))
    parser.add_argument("--checkpoint", type=Path, default=Path("climbgen.pt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--hard-oversample", action=argparse.BooleanOptionalAction, default=True,
                        help="repeat V8+ climbs (x2 V8-9, x3 V10-11, x4 V12+) so hard grades are not drowned out")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    placement_ids, coords = load_placements(args.catalog)
    examples = load_examples(args.climbs, placement_ids, coords)
    tag_counts = {tag: sum(TAGS.index(tag) in e.tags for e in examples) for tag in TAGS}
    print(f"{len(placement_ids)} placements, {len(examples)} examples, "
          f"{len({e.climb_uuid for e in examples})} climbs\ntag counts {tag_counts}")

    val_uuids = {u for u in {e.climb_uuid for e in examples} if zlib.crc32(u.encode()) % 20 == 0}
    train = [e for e in examples if e.climb_uuid not in val_uuids]
    val = [e for e in examples if e.climb_uuid in val_uuids]
    order_key = hold_order(coords, placement_ids)

    vocab = HOLD_BASE + len(placement_ids) * 4
    max_len = 3 + len(TAGS) + 1 + MAX_HOLDS + 1
    config = {"dim": args.dim, "layers": args.layers, "heads": args.heads, "ff": args.dim * 3,
              "vocabSize": vocab, "maxLen": max_len}
    model = ClimbGen(vocab, args.dim, args.layers, args.heads, args.dim * 3, max_len).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    lm_train = [e for e in train for _ in range(hard_repeats(e.v_grade) if args.hard_oversample else 1)]
    print(f"generator training rows: {len(lm_train)} ({len(train)} before oversampling hard grades)")
    steps = args.epochs * math.ceil(len(lm_train) / 256)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=steps, pct_start=0.05)

    val_seqs = [encode(e, order_key, tag_keep=1.0) for e in val]
    for epoch in range(args.epochs):
        model.train()
        random.shuffle(lm_train)
        # Drop tags sometimes so "no keywords" is a valid prompt too.
        seqs = [encode(e, order_key, tag_keep=0.75) for e in lm_train]
        total = 0.0
        for count, tokens in enumerate(batches(seqs, 256, device), 1):
            loss = lm_loss(model, tokens)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([lm_loss(model, t).item() for t in batches(val_seqs, 512, device)])
        print(f"epoch {epoch + 1}: train {total / count:.3f} val {val_loss:.3f}")

    torch.save({"model": model.state_dict(), "config": config}, args.checkpoint)
    export(model, config, placement_ids, coords, args.out)


if __name__ == "__main__":
    main()
