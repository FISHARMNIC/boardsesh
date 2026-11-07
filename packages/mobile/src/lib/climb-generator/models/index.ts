// One trained generator + grader per board layout (ml/climbgen, `--model <key>`).
import kilter1Grader from './kilter-1/grader.generated.json';
import kilter1Meta from './kilter-1/model-meta.generated.json';
import { CLIMB_GEN_WEIGHTS_BASE64 as kilter1Weights } from './kilter-1/model-weights.generated';
import type { ClimbGeneratorModelKey } from './supported';

export type ClimbGeneratorModelFiles = {
  meta: unknown;
  weightsBase64: string;
  grader: unknown;
};

export const CLIMB_GENERATOR_MODELS: Record<ClimbGeneratorModelKey, ClimbGeneratorModelFiles> = {
  'kilter-1': { meta: kilter1Meta, weightsBase64: kilter1Weights, grader: kilter1Grader },
};
