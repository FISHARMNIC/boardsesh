// Generators shipped inside the app. Every other supported board downloads its
// model on first use (remote-models.ts); see models/supported.ts for the list.
import kilter1Grader from './kilter-1/grader.generated.json';
import kilter1Meta from './kilter-1/model-meta.generated.json';
import { CLIMB_GEN_WEIGHTS_BASE64 as kilter1Weights } from './kilter-1/model-weights.generated';

export type BundledModelFiles = {
  meta: unknown;
  weightsBase64: string;
  grader: unknown;
};

export const BUNDLED_CLIMB_GENERATOR_MODELS: Partial<Record<string, BundledModelFiles>> = {
  'kilter-1': { meta: kilter1Meta, weightsBase64: kilter1Weights, grader: kilter1Grader },
};
