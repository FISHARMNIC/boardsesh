# climbgen: experimental on-device climb generator (Kilter and Tension)

A prototype. A small transformer (about 470k parameters, `train.py`) writes a Kilter Board Original climb hold by hold,
given an angle, a V-grade and optional keywords. A separate grader (about 27k parameters, `grader.py`) gives each
sample a probability per V-grade; the phone keeps the candidates most likely to be the requested grade and shows each
with a grade range. Both run in plain TypeScript on the phone (`packages/mobile/src/lib/climb-generator/`), with no
native module, so the feature ships over the air.

Open it from the climb editor: the wand button right of the heat toggle, on a new climb on one of the boards below.

## Boards

One generator and one grader per board layout (`--model <board>-<layout>` in both scripts; settings in `BOARD_MODELS`
in `train.py`). Each layout gets its own model because the models learn which hold sits where: TB2 Mirror and Spray
share 639 of 690 holes but bolt different holds into them.

| Model | Board | Training rows | Grader error | Range covers V0–V7 / V8–V11 |
|---|---|---|---|---|
| `kilter-1` | Kilter Original (12x12) | 126k | 0.99 V | 84–86% / 80% |
| `tension-9` | Tension Board 1 | 40k | 0.96 V | 82–86% / 79% |
| `tension-10` | Tension Board 2 Mirror | 24k | 1.05 V | 84–86% / 79% |
| `tension-11` | Tension Board 2 Spray | 19k | 1.08 V | 82% / 73% |
| `kilter-8` | Kilter Homewall (10x12) | 24k | 0.91 V | 85% / 78% |
| `moonboard-2` | MoonBoard 2016 | 50k | 1.09 V | 90% / 80% |
| `moonboard-3` | MoonBoard 2024 | 18k | 1.07 V | 90–91% / 84% |
| `moonboard-4` | MoonBoard Masters 2017 | 49k | 1.38 V | 88–89% / 90% |
| `moonboard-5` | MoonBoard Masters 2019 | 33k | 1.24 V | 88% / 81% |

The Tension generators use `--dim 64 --epochs 40` (smaller data) and keep the epoch with the best validation loss.
Keyword tags are sparse on Tension (a couple of hundred climbs each), so they steer less there.

MoonBoards number holds by grid cell on an 11x18 grid and have no foot-only role (`BOARD_MODELS` builds their
placements from the grid, not the catalogue). Their graders give wider ranges (about 4.5–6 grades on Masters 2017,
Masters 2019 and 2024) and read the easiest climbs up to a grade high, since most MoonBoard climbs sit at the bottom of
the scale. `out/train-more.sh` trains Homewall and the four MoonBoards in a row.

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

## Downloaded models

Every board downloads its model the first time its generator opens
(`packages/mobile/src/lib/climb-generator/remote-models.ts`): the app reads `manifest.json` from the models URL
(`models/remote-config.ts`, or `EXPO_PUBLIC_CLIMB_GENERATOR_MODELS_URL`), fetches that board's three files, checks each
against its SHA-256, and keeps them under the app's documents folder, so later opens work offline. A new version in
the manifest replaces the old files on the next open.

The models are hosted on Hugging Face for now. To publish:

```bash
.venv/bin/python -I package_models.py --version <date> tension-9 tension-10 tension-11   # writes out/hub/
.venv/bin/hf upload <user>/boardsesh-climbgen out/hub . --repo-type model
```

`package_models.py` keeps an existing `out/hub/manifest.json`, so packaging one board leaves the others listed.

## Train and export

Everything lives in this folder; `.venv/`, `data/` and `out/` are gitignored.

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy
# download kilter layout 1 + catalog into data/ via the manifests in docs/board-snapshots-dataset.md
.venv/bin/python -I -u train.py --climbs data/kilter1.db --catalog data/catalog.db --out out --checkpoint data/climbgen.pt
.venv/bin/python -I -u grader.py --climbs data/kilter1.db --catalog data/catalog.db --out out
```

The generator takes about 6 minutes on an Apple Silicon GPU, the grader about 3 (Tension boards: 2–4 each). Check
`out/` (the grader prints its test report), then copy `model-meta.generated.json`, `model-weights.generated.ts` and
`grader.generated.json` into `packages/mobile/src/lib/climb-generator/models/<model>/` and list the model in
`models/index.ts` and `models/supported.ts`. `out/train-tension.sh` trains all three Tension layouts in a row. `grader.py --eval-only` re-scores a saved grader without retraining.
