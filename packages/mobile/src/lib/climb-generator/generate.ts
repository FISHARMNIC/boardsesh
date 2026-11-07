/**
 * Experimental on-device climb generator.
 *
 * Samples candidate climbs from a small transformer (ml/climbgen/train.py),
 * grades each with the grader (ml/climbgen/grader.py) and keeps the ones most
 * likely to really be the requested grade, each with a grade range.
 *
 * Decoding rules the model was trained on, enforced here as hard masks:
 * - holds come out ordered bottom-to-top (then left-to-right), so each new hold
 *   must sit after the previous one in that order (also rules out repeats);
 * - only placements present on the climber's board can be picked;
 * - 1–2 start holds, 1–2 finish holds, 3–30 holds in total;
 * - only roles the board has (MoonBoard: no foot-only holds).
 *
 * Two versions, picked per call:
 * - v2: the models at the root of the models URL; ranks by grade alone.
 * - v3 (Kilter Original only, for now): a writer that can be asked for "classic"
 *   climbs, top-p sampling, then a geometry safety net, near-copy rejection
 *   against every known climb, and ranking that also weighs a realism critic
 *   (ml/climbgen/critic.py, ml/climbgen/V3.md).
 */

import { exceedsLimits, climbFeatures, type GeometryLimits } from './geometry';
import { ClimbGrader, type GraderExport } from './grader';
import { KnownClimbs } from './known-climbs';
import { CLIMB_GENERATOR_MODELS_URL } from './models/remote-config';
import type { ClimbGeneratorModelKey } from './models/supported';
import { expoRemoteModelStore, fetchBytes } from './remote-model-store';
import { RemoteModelError, decodeJson, ensureRemoteModel } from './remote-models';
import { AttentionSetEncoder, type SetEncoderExport } from './set-encoder';
import { sha256Hex } from './sha256';
import { ClimbTransformer, loadWeights, type ModelConfig, type TensorEntry } from './transformer';

export type ClimbGeneratorVersion = 'v2' | 'v3';

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
    /** v3 writers: the "ordinary" token; "classic" is the next one. Absent or -1 on v2. */
    qualityBase?: number;
    tagBase: number;
    holdBase: number;
  };
  /** [placementId, x, y] in board units, indexed by model placement index. */
  placements: number[][];
  tensors: TensorEntry[];
};

type CriticExport = SetEncoderExport & { kind: 'attention-critic'; writerSha256: string; temperature: number };

export type GenerateOptions = {
  /** Which board's model to use; see models/supported.ts. */
  modelKey: ClimbGeneratorModelKey;
  /** Defaults to v2. */
  version?: ClimbGeneratorVersion;
  angle: number;
  vGrade: number;
  tags: string[];
  /** Placement ids on the climber's board; holds outside it are never picked. */
  allowedPlacementIds: ReadonlySet<number>;
  candidates?: number;
  keep?: number;
  temperature?: number;
  random?: () => number;
  /** Clock for v3's time budget; tests pass a fake one. */
  now?: () => number;
};

export type GeneratedClimb = {
  frames: string;
  holdCount: number;
  /** The grader's median guess. */
  medianV: number;
  /** The grader's range: the true grade should fall inside it about 4 times in 5. */
  lowV: number;
  highV: number;
  /** How likely the grader thinks the requested grade is. */
  targetProbability: number;
};

/**
 * v3's settings, chosen offline with ml/climbgen/evaluate_v3.py (see V3.md):
 * sampling, how many candidates to try within a time budget, how much the
 * critic's realism counts against the grade, and the near-copy threshold.
 */
export const V3_PIPELINE = {
  temperature: 0.85,
  topP: 0.9,
  candidates: 48,
  timeBudgetMs: 3000,
  criticWeight: 0.5,
  criticLogFloor: -5,
  copyThreshold: 0.8,
  /** Ask for "classic" up to this grade; above it the classic data is too thin and copies rise. */
  classicUpToV: 15,
} as const;

const V3_FILES = ['meta', 'weights', 'grader', 'critic', 'geometry', 'known'] as const;
const MAX_HOLDS = 30;
const MIN_HOLDS = 3;
const ROLE_START = 0;
const ROLE_FINISH = 2;

type V3Parts = {
  critic: AttentionSetEncoder;
  criticTemperature: number;
  geometry: GeometryLimits;
  known: KnownClimbs;
};

type LoadedModel = {
  meta: ClimbGenMeta;
  transformer: ClimbTransformer;
  grader: ClimbGrader;
  /** Model placement index -> its rank in bottom-to-top, left-to-right order. */
  orderRank: Int32Array;
  v3: V3Parts | null;
};

const loaded = new Map<string, Promise<LoadedModel>>();

function buildModel(meta: ClimbGenMeta, weightBytes: Uint8Array, grader: GraderExport, v3: V3Parts | null) {
  const weights = loadWeights(weightBytes, meta.tensors);
  const order = meta.placements
    .map(([, x, y], index) => ({ index, x, y }))
    .sort((first, second) => first.y - second.y || first.x - second.x);
  const orderRank = new Int32Array(meta.placements.length);
  order.forEach((entry, rank) => {
    orderRank[entry.index] = rank;
  });
  return {
    meta,
    transformer: new ClimbTransformer(meta, weights),
    grader: new ClimbGrader(grader, meta.placements),
    orderRank,
    v3,
  };
}

async function fetchModel(modelKey: ClimbGeneratorModelKey, version: ClimbGeneratorVersion): Promise<LoadedModel> {
  if (!CLIMB_GENERATOR_MODELS_URL) {
    throw new RemoteModelError('This build has no address for downloaded climb generator models.', 'unavailable');
  }
  const deps = { baseUrl: CLIMB_GENERATOR_MODELS_URL, fetchBytes, store: expoRemoteModelStore };
  if (version === 'v2') {
    const { files } = await ensureRemoteModel(modelKey, deps);
    return buildModel(
      decodeJson(files.meta) as ClimbGenMeta,
      files.weights,
      decodeJson(files.grader) as GraderExport,
      null,
    );
  }
  const { files } = await ensureRemoteModel(modelKey, {
    ...deps,
    manifestPath: 'v3/manifest.json',
    requiredFiles: V3_FILES,
  });
  const meta = decodeJson(files.meta) as ClimbGenMeta;
  const critic = decodeJson(files.critic) as CriticExport;
  // A critic is a discriminator for one writer; a mismatched pair would rank nonsense.
  if (critic.writerSha256 !== sha256Hex(files.weights)) {
    throw new RemoteModelError('The v3 realism critic was trained for a different writer.', 'corrupt');
  }
  return buildModel(meta, files.weights, decodeJson(files.grader) as GraderExport, {
    critic: new AttentionSetEncoder(critic, meta.placements),
    criticTemperature: critic.temperature,
    geometry: decodeJson(files.geometry) as GeometryLimits,
    known: new KnownClimbs(files.known, meta.placements.length),
  });
}

/**
 * A board's generator, downloaded on first use and kept on the phone. Shared while
 * loading; a failed load is forgotten, so opening the screen again retries.
 */
export function loadClimbGenerator(
  modelKey: ClimbGeneratorModelKey,
  version: ClimbGeneratorVersion = 'v2',
): Promise<LoadedModel> {
  const cacheKey = `${version}:${modelKey}`;
  const pending = loaded.get(cacheKey);
  if (pending) return pending;
  const loading = fetchModel(modelKey, version);
  loaded.set(cacheKey, loading);
  loading.catch(() => loaded.delete(cacheKey));
  return loading;
}

export async function climbGeneratorTags(
  modelKey: ClimbGeneratorModelKey,
  version: ClimbGeneratorVersion = 'v2',
): Promise<string[]> {
  return (await loadClimbGenerator(modelKey, version)).meta.tags;
}

/**
 * Picks a token from `logits` among the allowed ones, after dividing by
 * `temperature`. With `topP` < 1, only the most likely tokens whose probabilities
 * add up to `topP` are kept (nucleus sampling), which drops the long tail of odd
 * holds.
 */
export function sampleFrom(
  logits: Float32Array,
  allowed: (token: number) => boolean,
  temperature: number,
  random: () => number,
  topP = 1,
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
  if (topP < 1) {
    const order: number[] = [];
    for (let token = 0; token < logits.length; token += 1) if (probabilities[token] > 0) order.push(token);
    order.sort((first, second) => probabilities[second] - probabilities[first]);
    let kept = 0;
    let cumulative = 0;
    for (const token of order) {
      if (cumulative >= topP * total) {
        probabilities[token] = 0;
        continue;
      }
      cumulative += probabilities[token];
      kept += probabilities[token];
    }
    total = kept;
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
  topP: number,
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
        // Four role slots per hold; a board with fewer roles (MoonBoard: no foot) never uses the rest.
        if (roleIndex >= meta.roles.length) return false;
        if (roleIndex === ROLE_START && starts >= 2) return false;
        if (roleIndex === ROLE_FINISH && finishes >= 2) return false;
        return true;
      },
      temperature,
      random,
      topP,
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

function logSigmoid(value: number): number {
  return value >= 0 ? -Math.log1p(Math.exp(-value)) : value - Math.log1p(Math.exp(value));
}

/** Yields to the event loop between candidates so the screen keeps painting. */
export async function generateClimbs(options: GenerateOptions): Promise<GeneratedClimb[]> {
  const version = options.version ?? 'v2';
  const model = await loadClimbGenerator(options.modelKey, version);
  const { meta, v3 } = model;
  const isV3 = version === 'v3' && v3 !== null;
  const {
    candidates = isV3 ? V3_PIPELINE.candidates : 24,
    keep = 5,
    temperature = isV3 ? V3_PIPELINE.temperature : 0.9,
    random = Math.random,
    now = () => Date.now(),
  } = options;
  const topP = isV3 ? V3_PIPELINE.topP : 1;

  const angle = meta.angles.reduce((best, candidate) =>
    Math.abs(candidate - options.angle) < Math.abs(best - options.angle) ? candidate : best,
  );
  const vGrade = Math.min(meta.maxV, Math.max(0, Math.round(options.vGrade)));
  const tagTokens = options.tags
    .map((tag) => meta.tags.indexOf(tag))
    .filter((index) => index >= 0)
    .sort((first, second) => first - second)
    .map((index) => meta.vocab.tagBase + index);
  const qualityBase = meta.vocab.qualityBase ?? -1;
  const qualityToken = isV3 && qualityBase >= 0 ? [qualityBase + (vGrade <= V3_PIPELINE.classicUpToV ? 1 : 0)] : [];
  const prefix = [
    meta.vocab.bos,
    meta.vocab.angleBase + meta.angles.indexOf(angle),
    meta.vocab.gradeBase + vGrade,
    ...qualityToken,
    ...tagTokens,
    meta.vocab.sep,
  ];

  const allowedPlacement = new Uint8Array(meta.placements.length);
  meta.placements.forEach(([placementId], index) => {
    allowedPlacement[index] = options.allowedPlacementIds.has(placementId) ? 1 : 0;
  });

  const deadline = now() + V3_PIPELINE.timeBudgetMs;
  const seen = new Set<string>();
  const results: (GeneratedClimb & { score: number })[] = [];
  for (let attempt = 0; attempt < candidates; attempt += 1) {
    // v3 tries more candidates; on a slow phone it stops at the time budget with what it has.
    if (isV3 && results.length >= keep && now() > deadline) break;
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
    const holds = sampleOne(model, prefix, allowedPlacement, temperature, topP, random);
    if (!holds) continue;
    const frames = holds
      .map(({ placementIndex, roleIndex }) => `p${meta.placements[placementIndex][0]}r${meta.roles[roleIndex]}`)
      .join('');
    if (seen.has(frames)) continue;
    seen.add(frames);

    let realism = 0;
    if (isV3 && v3) {
      if (exceedsLimits(climbFeatures(holds, meta.placements), v3.geometry, vGrade, angle)) continue;
      if (v3.known.maxJaccard(holds.map((hold) => hold.placementIndex)) >= V3_PIPELINE.copyThreshold) continue;
      const [criticLogit] = v3.critic.logits(holds, angle);
      realism = Math.max(logSigmoid(criticLogit / v3.criticTemperature), V3_PIPELINE.criticLogFloor);
    }
    const estimate = model.grader.grade(holds, angle);
    const targetProbability = estimate.probabilities[vGrade];
    results.push({
      frames,
      holdCount: holds.length,
      medianV: estimate.medianV,
      lowV: estimate.lowV,
      highV: estimate.highV,
      targetProbability,
      score: Math.log(targetProbability + 1e-9) + (isV3 ? V3_PIPELINE.criticWeight * realism : 0),
    });
  }
  // v2: most likely to really be the requested grade first. v3: that, weighed with realism.
  return results
    .sort((first, second) => second.score - first.score)
    .slice(0, keep)
    .map(({ score: _score, ...climb }) => climb);
}
