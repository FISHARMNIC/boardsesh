# climbgen: experimental on-device Kilter climb generator

A prototype. A small transformer (about 470k parameters) writes a Kilter Board Original climb hold by hold, given an
angle, a V-grade and optional keywords. A linear grade scorer trained alongside it reranks the samples, so the phone
shows the candidates whose predicted grade is closest to the one asked for. Both run in plain TypeScript on the phone
(`packages/mobile/src/lib/climb-generator/`), with no native module, so the feature ships over the air.

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

## Train and export

```bash
python3 -m venv .venv && .venv/bin/pip install torch numpy
# download kilter layout 1 + catalog via the manifests in docs/board-snapshots-dataset.md
.venv/bin/python ml/climbgen/train.py --climbs kilter1.db --catalog catalog.db
```

This takes about 5 minutes on an Apple Silicon GPU. It writes `model-meta.generated.json` (vocabulary, placements,
tensor index and grade scorer) and `model-weights.generated.ts` (int8 weights, base64) into the mobile folder.
