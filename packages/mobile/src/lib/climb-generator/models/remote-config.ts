/**
 * Where downloaded climb generator models live: a folder with `manifest.json`
 * at its root (see remote-models.ts). Overridable per build with
 * EXPO_PUBLIC_CLIMB_GENERATOR_MODELS_URL, so moving the models to Boardsesh's
 * own storage later is a one-line change.
 */
// Hugging Face model repo fisharmnic/boardsesh-climbgen, public; no login to download.
const DEFAULT_MODELS_URL = 'https://huggingface.co/fisharmnic/boardsesh-climbgen/resolve/main';

export const CLIMB_GENERATOR_MODELS_URL = (
  process.env.EXPO_PUBLIC_CLIMB_GENERATOR_MODELS_URL?.trim() || DEFAULT_MODELS_URL
).replace(/\/+$/, '');
