/**
 * Experimental on-device Kilter climb generator.
 *
 * Samples candidate climbs from the tiny transformer (ml/climbgen/train.py),
 * then grades each with the grader (ml/climbgen/grader.py) and keeps the ones
 * most likely to really be the requested grade, each with an 80% grade range.
 *
 * Decoding rules the model was trained on, enforced here as hard masks:
 * - holds come out ordered bottom-to-top (then left-to-right), so each new hold
 *   must sit after the previous one in that order (also rules out repeats);
 * - only placements present on the climber's board can be picked;
 * - 1–2 start holds, 1–2 finish holds, 3–30 holds in total.
 */

import { ClimbGrader, type GraderExport } from './grader';
import { CLIMB_GENERATOR_MODELS } from './models';
import type { ClimbGeneratorModelKey } from './models/supported';
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
};

export type GenerateOptions = {
  /** Which board's model to use; see models/supported.ts. */
  modelKey: ClimbGeneratorModelKey;
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
  /** The grader's median guess. */
  medianV: number;
  /** The grader's 80% range: the true grade should fall inside it 4 times in 5. */
  lowV: number;
  highV: number;
  /** How likely the grader thinks the requested grade is. */
  targetProbability: number;
};

const MAX_HOLDS = 30;
const MIN_HOLDS = 3;
const ROLE_START = 0;
const ROLE_FINISH = 2;

type LoadedModel = {
  meta: ClimbGenMeta;
  transformer: ClimbTransformer;
  grader: ClimbGrader;
  /** Model placement index -> its rank in bottom-to-top, left-to-right order. */
  orderRank: Int32Array;
};

const loaded = new Map<ClimbGeneratorModelKey, LoadedModel>();

/** Decodes a board's weights the first time the screen asks for it, not at app start. */
export function loadClimbGenerator(modelKey: ClimbGeneratorModelKey): LoadedModel {
  const cached = loaded.get(modelKey);
  if (cached) return cached;
  const files = CLIMB_GENERATOR_MODELS[modelKey];
  const meta = files.meta as ClimbGenMeta;
  const weights = loadWeights(decodeBase64(files.weightsBase64), meta.tensors);
  const order = meta.placements
    .map(([, x, y], index) => ({ index, x, y }))
    .sort((first, second) => first.y - second.y || first.x - second.x);
  const orderRank = new Int32Array(meta.placements.length);
  order.forEach((entry, rank) => {
    orderRank[entry.index] = rank;
  });
  const model: LoadedModel = {
    meta,
    transformer: new ClimbTransformer(meta, weights),
    grader: new ClimbGrader(files.grader as GraderExport, meta.placements),
    orderRank,
  };
  loaded.set(modelKey, model);
  return model;
}

export function climbGeneratorTags(modelKey: ClimbGeneratorModelKey): string[] {
  return loadClimbGenerator(modelKey).meta.tags;
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

/** Yields to the event loop between candidates so the screen keeps painting. */
export async function generateClimbs(options: GenerateOptions): Promise<GeneratedClimb[]> {
  const model = loadClimbGenerator(options.modelKey);
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

  const seen = new Set<string>();
  const results: GeneratedClimb[] = [];
  for (let attempt = 0; attempt < candidates; attempt += 1) {
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    const holds = sampleOne(model, prefix, allowedPlacement, temperature, random);
    if (!holds) continue;
    const frames = holds
      .map(({ placementIndex, roleIndex }) => `p${meta.placements[placementIndex][0]}r${meta.roles[roleIndex]}`)
      .join('');
    if (seen.has(frames)) continue;
    seen.add(frames);
    const estimate = model.grader.grade(holds, angle);
    results.push({
      frames,
      holdCount: holds.length,
      medianV: estimate.medianV,
      lowV: estimate.lowV,
      highV: estimate.highV,
      targetProbability: estimate.probabilities[vGrade],
    });
  }
  // Most likely to really be the requested grade first.
  return results.sort((first, second) => second.targetProbability - first.targetProbability).slice(0, keep);
}
