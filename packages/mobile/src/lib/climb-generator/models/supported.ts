/**
 * Which board layouts have a trained climb generator, keyed `<board>-<layoutId>`.
 * Kept free of the model files, so a screen can ask "is there a generator for
 * this board?" without pulling ~1 MB of weights into its import graph.
 */
export const CLIMB_GENERATOR_MODEL_KEYS = ['kilter-1'] as const;

export type ClimbGeneratorModelKey = (typeof CLIMB_GENERATOR_MODEL_KEYS)[number];

export function climbGeneratorModelKey(boardName: string, layoutId: number): ClimbGeneratorModelKey | null {
  const key = `${boardName}-${layoutId}`;
  return (CLIMB_GENERATOR_MODEL_KEYS as readonly string[]).includes(key) ? (key as ClimbGeneratorModelKey) : null;
}
