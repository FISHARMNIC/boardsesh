// One trained generator + grader per board layout (ml/climbgen, `--model <key>`).
import kilter1Grader from './kilter-1/grader.generated.json';
import kilter1Meta from './kilter-1/model-meta.generated.json';
import { CLIMB_GEN_WEIGHTS_BASE64 as kilter1Weights } from './kilter-1/model-weights.generated';
import tension9Grader from './tension-9/grader.generated.json';
import tension9Meta from './tension-9/model-meta.generated.json';
import { CLIMB_GEN_WEIGHTS_BASE64 as tension9Weights } from './tension-9/model-weights.generated';
import tension10Grader from './tension-10/grader.generated.json';
import tension10Meta from './tension-10/model-meta.generated.json';
import { CLIMB_GEN_WEIGHTS_BASE64 as tension10Weights } from './tension-10/model-weights.generated';
import tension11Grader from './tension-11/grader.generated.json';
import tension11Meta from './tension-11/model-meta.generated.json';
import { CLIMB_GEN_WEIGHTS_BASE64 as tension11Weights } from './tension-11/model-weights.generated';
import type { ClimbGeneratorModelKey } from './supported';

export type ClimbGeneratorModelFiles = {
  meta: unknown;
  weightsBase64: string;
  grader: unknown;
};

export const CLIMB_GENERATOR_MODELS: Record<ClimbGeneratorModelKey, ClimbGeneratorModelFiles> = {
  'kilter-1': { meta: kilter1Meta, weightsBase64: kilter1Weights, grader: kilter1Grader },
  'tension-9': { meta: tension9Meta, weightsBase64: tension9Weights, grader: tension9Grader },
  'tension-10': { meta: tension10Meta, weightsBase64: tension10Weights, grader: tension10Grader },
  'tension-11': { meta: tension11Meta, weightsBase64: tension11Weights, grader: tension11Grader },
};
