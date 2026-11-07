/**
 * Which board layouts have a trained climb generator, keyed `<board>-<layoutId>`.
 * Kilter Original ships in the app; the rest download on first use. Kept free of
 * the model files, so a screen can ask "is there a generator for this board?"
 * without pulling weights into its import graph.
 */
// Kilter Original, Tension Board 1, Tension Board 2 Mirror and Spray. The two TB2
// layouts share most holes but not the holds bolted into them, so each has its own.
export const CLIMB_GENERATOR_MODEL_KEYS = ['kilter-1', 'tension-9', 'tension-10', 'tension-11'] as const;

export type ClimbGeneratorModelKey = (typeof CLIMB_GENERATOR_MODEL_KEYS)[number];

export function climbGeneratorModelKey(boardName: string, layoutId: number): ClimbGeneratorModelKey | null {
  const key = `${boardName}-${layoutId}`;
  return (CLIMB_GENERATOR_MODEL_KEYS as readonly string[]).includes(key) ? (key as ClimbGeneratorModelKey) : null;
}
