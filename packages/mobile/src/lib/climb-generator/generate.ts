/**
 * Experimental on-device Kilter climb generator.
 *
 * Samples candidate climbs from the tiny transformer (ml/climbgen/train.py),
 * then reranks them with a linear grade scorer trained alongside it, so the
 * climbs shown are the ones whose predicted grade is closest to what was asked.
 *
 * Decoding rules the model was trained on, enforced here as hard masks:
 * - holds come out ordered bottom-to-top (then left-to-right), so each new hold
 *   must sit after the previous one in that order (also rules out repeats);
 * - only placements present on the climber's board can be picked;
 * - 1–2 start holds, 1–2 finish holds, 3–30 holds in total.
 */

import { ClimbTransformer, decodeBase64, loadWeights, type ModelConfig, type TensorEntry } from './transformer';

type ClimbGenMeta = ModelConfig & {
  board: string;
  layoutId: number;
  roles: number[];
  angles: number[];
  maxV: number;
  tags: string[];
  vocab: {
    pad: number;
    bos: number;
    sep: number;
    eos: number;
    angleBase: number;
    gradeBase: number;
    tagBase: number;
    holdBase: number;
  };
  /** [placementId, x, y] in board units, indexed by model placement index. */
  placements: number[][];
  tensors: TensorEntry[];
  scorer: { hold: number[][]; angle: number[]; bias: number };
};

export type GenerateOptions = {
  angle: number;
  vGrade: number;
  tags: string[];
  /** Placement ids on the climber's board; holds outside it are never picked. */
  allowedPlacementIds: ReadonlySet<number>;
  candidates?: number;
  keep?: number;
  temperature?: number;
  random?: () => number;
};

export type GeneratedClimb = {
  frames: string;
  holdCount: number;
  predictedDifficulty: number;
  predictedV: number;
};

const MAX_HOLDS = 30;
const MIN_HOLDS = 3;
const ROLE_START = 0;
const ROLE_FINISH = 2;

// Kilter difficulty id -> V grade (board_difficulty_grades), same table as train.py.
const V_BY_DIFFICULTY: Record<number, number> = {
  10: 0,
  11: 0,
  12: 0,
  13: 1,
  14: 1,
  15: 2,
  16: 3,
  17: 3,
  18: 4,
  19: 4,
  20: 5,
  21: 5,
  22: 6,
  23: 7,
  24: 8,
  25: 8,
  26: 9,
  27: 10,
  28: 11,
  29: 12,
  30: 13,
  31: 14,
  32: 15,
  33: 16,
};

export function difficultyToV(difficulty: number): number {
  const rounded = Math.min(33, Math.max(10, Math.round(difficulty)));
  return V_BY_DIFFICULTY[rounded];
}

function targetDifficulty(vGrade: number): number {
  const ids = Object.entries(V_BY_DIFFICULTY)
    .filter(([, grade]) => grade === vGrade)
    .map(([id]) => Number(id));
  return ids.reduce((sum, id) => sum + id, 0) / ids.length;
}

type LoadedModel = {
  meta: ClimbGenMeta;
  transformer: ClimbTransformer;
  /** Model placement index -> its rank in bottom-to-top, left-to-right order. */
  orderRank: Int32Array;
};

let loaded: LoadedModel | null = null;

/** Lazily decodes the ~0.7 MB of weights the first time the screen asks. */
export function loadClimbGenerator(): LoadedModel {
  if (loaded) return loaded;
  // Required lazily so the weights string is only decoded on this screen.
  // oxlint-disable-next-line no-require-imports
  const meta = require('./model-meta.generated.json') as ClimbGenMeta;
  // oxlint-disable-next-line no-require-imports
  const { CLIMB_GEN_WEIGHTS_BASE64 } = require('./model-weights.generated') as { CLIMB_GEN_WEIGHTS_BASE64: string };
  const weights = loadWeights(decodeBase64(CLIMB_GEN_WEIGHTS_BASE64), meta.tensors);
  const order = meta.placements
    .map(([, x, y], index) => ({ index, x, y }))
    .sort((first, second) => first.y - second.y || first.x - second.x);
  const orderRank = new Int32Array(meta.placements.length);
  order.forEach((entry, rank) => {
    orderRank[entry.index] = rank;
  });
  loaded = { meta, transformer: new ClimbTransformer(meta, weights), orderRank };
  return loaded;
}

export function climbGeneratorTags(): string[] {
  return loadClimbGenerator().meta.tags;
}

function sampleFrom(
  logits: Float32Array,
  allowed: (token: number) => boolean,
  temperature: number,
  random: () => number,
) {
  let maxLogit = -Infinity;
  for (let token = 0; token < logits.length; token += 1) {
    if (allowed(token) && logits[token] > maxLogit) maxLogit = logits[token];
  }
  if (maxLogit === -Infinity) return -1;
  const probabilities = new Float64Array(logits.length);
  let total = 0;
  for (let token = 0; token < logits.length; token += 1) {
    if (!allowed(token)) continue;
    probabilities[token] = Math.exp((logits[token] - maxLogit) / temperature);
    total += probabilities[token];
  }
  let threshold = random() * total;
  for (let token = 0; token < logits.length; token += 1) {
    threshold -= probabilities[token];
    if (probabilities[token] > 0 && threshold <= 0) return token;
  }
  return -1;
}

type Hold = { placementIndex: number; roleIndex: number };

function sampleOne(
  model: LoadedModel,
  prefix: number[],
  allowedPlacement: Uint8Array,
  temperature: number,
  random: () => number,
): Hold[] | null {
  const { meta, transformer, orderRank } = model;
  const { holdBase, eos } = meta.vocab;
  const session = transformer.session();
  let logits: Float32Array = new Float32Array(0);
  for (const token of prefix) logits = session.feed(token);

  const holds: Hold[] = [];
  let lastRank = -1;
  let starts = 0;
  let finishes = 0;
  while (holds.length < MAX_HOLDS) {
    const canEnd = holds.length >= MIN_HOLDS && starts >= 1 && finishes >= 1;
    const token = sampleFrom(
      logits,
      (candidate) => {
        if (candidate === eos) return canEnd;
        if (candidate < holdBase) return false;
        const placementIndex = (candidate - holdBase) >> 2;
        const roleIndex = (candidate - holdBase) & 3;
        if (!allowedPlacement[placementIndex] || orderRank[placementIndex] <= lastRank) return false;
        if (roleIndex === ROLE_START && starts >= 2) return false;
        if (roleIndex === ROLE_FINISH && finishes >= 2) return false;
        return true;
      },
      temperature,
      random,
    );
    if (token === eos) return holds;
    if (token < 0) return null;
    const placementIndex = (token - holdBase) >> 2;
    const roleIndex = (token - holdBase) & 3;
    holds.push({ placementIndex, roleIndex });
    lastRank = orderRank[placementIndex];
    if (roleIndex === ROLE_START) starts += 1;
    if (roleIndex === ROLE_FINISH) finishes += 1;
    if (holds.length === MAX_HOLDS) break;
    logits = session.feed(token);
  }
  return starts >= 1 && finishes >= 1 ? holds : null;
}

function predictDifficulty(model: LoadedModel, holds: Hold[], angle: number): number {
  const { meta } = model;
  const angleIndex = Math.max(0, meta.angles.indexOf(angle));
  let flat = 0;
  let slope = 0;
  for (const { placementIndex, roleIndex } of holds) {
    const weights = meta.scorer.hold[meta.vocab.holdBase + placementIndex * 4 + roleIndex];
    flat += weights[0];
    slope += weights[1];
  }
  return meta.scorer.bias + meta.scorer.angle[angleIndex] + flat + ((angle - 40) / 10) * slope;
}

/** Yields to the event loop between candidates so the screen keeps painting. */
export async function generateClimbs(options: GenerateOptions): Promise<GeneratedClimb[]> {
  const model = loadClimbGenerator();
  const { meta } = model;
  const { candidates = 24, keep = 5, temperature = 0.9, random = Math.random } = options;

  const angle = meta.angles.reduce((best, candidate) =>
    Math.abs(candidate - options.angle) < Math.abs(best - options.angle) ? candidate : best,
  );
  const vGrade = Math.min(meta.maxV, Math.max(0, Math.round(options.vGrade)));
  const tagTokens = options.tags
    .map((tag) => meta.tags.indexOf(tag))
    .filter((index) => index >= 0)
    .sort((first, second) => first - second)
    .map((index) => meta.vocab.tagBase + index);
  const prefix = [
    meta.vocab.bos,
    meta.vocab.angleBase + meta.angles.indexOf(angle),
    meta.vocab.gradeBase + vGrade,
    ...tagTokens,
    meta.vocab.sep,
  ];

  const allowedPlacement = new Uint8Array(meta.placements.length);
  meta.placements.forEach(([placementId], index) => {
    allowedPlacement[index] = options.allowedPlacementIds.has(placementId) ? 1 : 0;
  });

  const target = targetDifficulty(vGrade);
  const seen = new Set<string>();
  const results: (GeneratedClimb & { distance: number })[] = [];
  for (let attempt = 0; attempt < candidates; attempt += 1) {
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    const holds = sampleOne(model, prefix, allowedPlacement, temperature, random);
    if (!holds) continue;
    const frames = holds
      .map(({ placementIndex, roleIndex }) => `p${meta.placements[placementIndex][0]}r${meta.roles[roleIndex]}`)
      .join('');
    if (seen.has(frames)) continue;
    seen.add(frames);
    const predictedDifficulty = predictDifficulty(model, holds, angle);
    results.push({
      frames,
      holdCount: holds.length,
      predictedDifficulty,
      predictedV: difficultyToV(predictedDifficulty),
      distance: Math.abs(predictedDifficulty - target),
    });
  }
  return results
    .sort((first, second) => first.distance - second.distance)
    .slice(0, keep)
    .map(({ distance: _distance, ...climb }) => climb);
}
