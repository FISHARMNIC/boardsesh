"""The app's generation pipeline in Python, batched, for offline evaluation and
for making critic negatives. Mirrors packages/mobile/src/lib/climb-generator/
generate.ts: the same decoding rules (holds on the board, bottom-to-top order,
1-2 starts and finishes, 3-30 holds, roles the board has), plus top-p.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grader as grader_module  # noqa: E402
import train  # noqa: E402

MIN_HOLDS, MAX_HOLDS = 3, train.MAX_HOLDS


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_writer(checkpoint: Path, dev: str):
    """A trained writer; also switches train's vocabulary to match it."""
    state = torch.load(checkpoint, map_location="cpu")
    config = state["config"]
    train.set_quality_tokens(bool(config.get("qualityToken", False)))
    model = train.ClimbGen(config["vocabSize"], config["dim"], config["layers"], config["heads"], config["ff"],
                           config["maxLen"])
    model.load_state_dict(state["model"])
    return model.to(dev).eval(), config


def load_grader(checkpoint: Path, export_json: Path, placements: int, dev: str):
    exported = json.loads(export_json.read_text())
    kind = exported.get("kind", "deepsets")
    if kind == "attention":
        model = grader_module.AttentionGrader(placements)
    else:
        model = grader_module.Grader(placements)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    correction = torch.tensor(exported["logitCorrection"], dtype=torch.float32)
    return model.to(dev).eval(), correction, float(exported["temperature"])


def order_rank(placement_ids, coords) -> np.ndarray:
    order = sorted(range(len(placement_ids)), key=lambda i: (coords[placement_ids[i]][1], coords[placement_ids[i]][0]))
    rank = np.empty(len(placement_ids), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def prefix(angle: int, v_grade: int, quality: int | None, tags=()) -> list[int]:
    """`quality` is 1 (classic), 0 (ordinary) or None (no token / v2 writer)."""
    tokens = [train.BOS, train.ANGLE_BASE + train.ANGLES.index(angle), train.GRADE_BASE + v_grade]
    if quality is not None and train.QUALITY_BASE >= 0:
        tokens.append(train.QUALITY_BASE + quality)
    tokens += [train.TAG_BASE + train.TAGS.index(tag) for tag in tags]
    tokens.append(train.SEP)
    return tokens


@torch.no_grad()
def sample(model, prefixes: list[list[int]], rank: np.ndarray, allowed: np.ndarray, n_roles: int,
           temperature: float, top_p: float, dev: str, generator: torch.Generator | None = None):
    """One climb per prefix: list of (placement index, role index), or None if decoding got stuck."""
    count = len(prefixes)
    placements = len(rank)
    width = max(len(p) for p in prefixes)
    if any(len(p) != width for p in prefixes):
        raise ValueError("prefixes must have equal length")
    tokens = torch.tensor(prefixes, device=dev)
    rank_t = torch.tensor(rank, device=dev)
    allowed_t = torch.tensor(allowed, device=dev, dtype=torch.bool)
    last_rank = torch.full((count,), -1, device=dev)
    starts = torch.zeros(count, device=dev, dtype=torch.long)
    finishes = torch.zeros(count, device=dev, dtype=torch.long)
    holds_n = torch.zeros(count, device=dev, dtype=torch.long)
    done = torch.zeros(count, device=dev, dtype=torch.bool)
    stuck = torch.zeros(count, device=dev, dtype=torch.bool)
    picked: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    hold_base = train.HOLD_BASE

    for _ in range(MAX_HOLDS + 1):
        logits = model(tokens)[:, -1, :].float()
        vocab = logits.shape[1]
        rank_ok = (rank_t[None, :] > last_rank[:, None]) & allowed_t[None, :]
        role_ok = torch.stack([starts < 2, torch.ones_like(done), finishes < 2,
                               torch.full_like(done, n_roles > 3)], dim=1)
        hold_ok = (rank_ok[:, :, None] & role_ok[:, None, :]).reshape(count, placements * 4)
        hold_ok &= (holds_n < MAX_HOLDS)[:, None]
        mask = torch.zeros(count, vocab, dtype=torch.bool, device=dev)
        mask[:, hold_base:hold_base + placements * 4] = hold_ok
        mask[:, train.EOS] = (holds_n >= MIN_HOLDS) & (starts >= 1) & (finishes >= 1)
        logits = logits.masked_fill(~mask, float("-inf")) / temperature
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, order = probs.sort(dim=-1, descending=True)
            keep = sorted_probs.cumsum(dim=-1) - sorted_probs < top_p
            keep[:, 0] = True
            sorted_probs = sorted_probs * keep
            probs = torch.zeros_like(probs).scatter(1, order, sorted_probs)
        has_option = mask.any(dim=1)
        stuck |= ~has_option & ~done
        probs[~has_option] = 0
        probs[~has_option, train.EOS] = 1
        choice = torch.multinomial(probs / probs.sum(dim=1, keepdim=True), 1, generator=generator).squeeze(1)
        choice = torch.where(done, torch.full_like(choice, train.PAD), choice)
        is_hold = (choice >= hold_base) & ~done
        index = (choice - hold_base).clamp(min=0)
        placement, role = index // 4, index % 4
        for row in torch.nonzero(is_hold).flatten().tolist():
            picked[row].append((int(placement[row]), int(role[row])))
        last_rank = torch.where(is_hold, rank_t[placement.clamp(max=placements - 1)], last_rank)
        starts += (is_hold & (role == 0)).long()
        finishes += (is_hold & (role == 2)).long()
        holds_n += is_hold.long()
        done |= choice == train.EOS
        tokens = torch.cat([tokens, choice[:, None]], dim=1)
        if bool(done.all()):
            break
    return [None if (stuck[i] or not done[i]) else picked[i] for i in range(count)]


@torch.no_grad()
def grade_probabilities(model, correction, temperature, climbs, angles, coords, placement_ids, dev):
    examples = [train.Example("x", angle, 0, 0.0, [], holds) for holds, angle in zip(climbs, angles)]
    out = []
    for start in range(0, len(examples), 2048):
        batch = grader_module.tensors_for(examples[start:start + 2048], coords, placement_ids, dev)
        logits = model(*batch[:5]).cpu()
        out.append(F.softmax((logits + correction) / temperature, dim=-1))
    return torch.cat(out).numpy()


def median_grade(probabilities: np.ndarray) -> np.ndarray:
    return (probabilities.cumsum(axis=1) >= 0.5).argmax(axis=1)


class KnownClimbs:
    """Every real climb's hold set (placements only), for near-copy checks."""

    def __init__(self, hold_sets: list[frozenset[int]], placements: int):
        self.sets = hold_sets
        self.sizes = np.array([len(s) for s in hold_sets], dtype=np.int32)
        postings: list[list[int]] = [[] for _ in range(placements)]
        for climb, holds in enumerate(hold_sets):
            for placement in holds:
                postings[placement].append(climb)
        self.postings = [np.array(p, dtype=np.int32) for p in postings]

    def max_jaccard(self, holds) -> float:
        placements = {p for p, _ in holds}
        lists = [self.postings[p] for p in placements if len(self.postings[p])]
        if not lists:
            return 0.0
        overlap = np.bincount(np.concatenate(lists), minlength=len(self.sets))
        union = len(placements) + self.sizes - overlap
        return float((overlap / np.maximum(union, 1)).max())


def known_climbs(examples, placements: int) -> KnownClimbs:
    unique = {frozenset(p for p, _ in example.holds) for example in examples}
    return KnownClimbs(sorted(unique, key=sorted), placements)
