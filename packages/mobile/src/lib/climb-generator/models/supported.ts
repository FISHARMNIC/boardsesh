/**
 * Which board layouts have a trained climb generator, keyed `<board>-<layoutId>`.
 * Every model downloads on first use (remote-models.ts). Kept free of model
 * code, so a screen can ask "is there a generator for this board?" cheaply.
 *
 * The two TB2 layouts share most holes but not the holds in them, and the four
 * MoonBoards are different hold sets on one grid, so each has its own model.
 */
export const CLIMB_GENERATOR_MODEL_KEYS = [
  'kilter-1',
  'kilter-8',
  'tension-9',
  'tension-10',
  'tension-11',
  'moonboard-2',
  'moonboard-3',
  'moonboard-4',
  'moonboard-5',
] as const;

export type ClimbGeneratorModelKey = (typeof CLIMB_GENERATOR_MODEL_KEYS)[number];

export function climbGeneratorModelKey(boardName: string, layoutId: number): ClimbGeneratorModelKey | null {
  const key = `${boardName}-${layoutId}`;
  return (CLIMB_GENERATOR_MODEL_KEYS as readonly string[]).includes(key) ? (key as ClimbGeneratorModelKey) : null;
}
