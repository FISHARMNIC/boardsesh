# climbgen: experimental on-device Kilter climb generator

A prototype. A small transformer (about 470k parameters, `train.py`) writes a Kilter Board Original climb hold by hold,
given an angle, a V-grade and optional keywords. A separate grader (about 27k parameters, `grader.py`) gives each
sample a probability per V-grade; the phone keeps the candidates most likely to be the requested grade and shows each
with a grade range. Both run in plain TypeScript on the phone (`packages/mobile/src/lib/climb-generator/`), with no
native module, so the feature ships over the air.

Open it from Settings › Developer › Climb Generator with a Kilter Original as the active board.

## Data

The public nightly snapshots (`docs/board-snapshots-dataset.md`): the `kilter` layout `1` climbs file and the board
catalog file. Training uses listed climbs with at least 3 ascents that fit a 12×12 board, one example per
(climb, angle): about 126k examples from 80k climbs.

Keywords come from two places:

- **Text:** regexes over each climb's name and description ("crimp", "dyno", "campus" and so on), plus the
  `no_match` characteristic. They are sparse (a few hundred to about 2,500 climbs each), so treat them as hints.
- **Geometry:** `reachy` (the biggest hand-to-hand gap is in the top 15%), `traverse` (wider than tall), `short` and
  `long` (hold count in the bottom or top 15%).

## Grader

A Deep Sets model, the same family as `ml/climb2vec`, trained on the same data with three splits by climb: train,
calibration (fits a temperature so the probabilities are honest) and test (the report below; never trained or tuned on).
It uses a **flat grade prior**: generated climbs already aim at a grade, so it judges each climb on its holds alone
instead of leaning toward common grades. With the catalogue prior it read real V10s as about V8.6.

Held-out test, flat prior: median-guess error 0.99 V. The average guess is within 0.5 V of the true grade for V1–V12
(V0 reads 0.7 high). The shown range (75% of the probability) contains the true grade 86% of the time at V0–V4,
84% at V5–V7, 80% at V8–V11 and 76% at V12+ (38 climbs, too few to tune on).

## Train and export

Everything lives in this folder; `.venv/`, `data/` and `out/` are gitignored.

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy
# download kilter layout 1 + catalog into data/ via the manifests in docs/board-snapshots-dataset.md
.venv/bin/python -I -u train.py --climbs data/kilter1.db --catalog data/catalog.db --out out --checkpoint data/climbgen.pt
.venv/bin/python -I -u grader.py --climbs data/kilter1.db --catalog data/catalog.db --out out
```

The generator takes about 6 minutes on an Apple Silicon GPU, the grader about 3. Check `out/` (the grader prints its
test report), then copy `model-meta.generated.json`, `model-weights.generated.ts` and `grader.generated.json` into
`packages/mobile/src/lib/climb-generator/`. `grader.py --eval-only` re-scores a saved grader without retraining.
