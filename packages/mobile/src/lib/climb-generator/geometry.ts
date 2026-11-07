/**
 * Geometry of a climb's hand moves (mirrors ml/climbgen/features.py), in board
 * units: inches (Kilter and Tension share a 4-inch grid). Used by v3 as a safety net: a candidate is
 * dropped only if it goes beyond the 99th percentile of real classic climbs in its
 * grade band and angle band.
 */

export type GeometryHold = { placementIndex: number; roleIndex: number };

export type GeometryLimits = {
  percentile: number;
  gradeBands: number[][];
  angleBands: number[][];
  /** `<grade band>-<angle band>` -> [max step, max vertical gap, unsupported moves]. */
  bands: Record<string, number[]>;
};

const FOOT_BELOW_MIN = 20;
const FOOT_BELOW_MAX = 70;
const FOOT_SIDEWAYS = 30;
const FOOT_ROLE = 3;
const START_ROLE = 0;

/** [max step, max vertical gap, unsupported hand moves]. `placements` rows are [placementId, x, y]. */
export function climbFeatures(holds: GeometryHold[], placements: number[][]): [number, number, number] {
  const points = holds.map(({ placementIndex, roleIndex }) => ({
    x: placements[placementIndex][1],
    y: placements[placementIndex][2],
    role: roleIndex,
  }));
  const hands = points
    .filter((point) => point.role !== FOOT_ROLE)
    .sort((first, second) => first.y - second.y || first.x - second.x);
  let maxStep = 0;
  let maxGap = 0;
  for (let index = 1; index < hands.length; index += 1) {
    const step = Math.hypot(hands[index].x - hands[index - 1].x, hands[index].y - hands[index - 1].y);
    if (step > maxStep) maxStep = step;
    const gap = hands[index].y - hands[index - 1].y;
    if (gap > maxGap) maxGap = gap;
  }
  let unsupported = 0;
  for (const hand of hands) {
    if (hand.role === START_ROLE) continue;
    const supported = points.some(
      (other) =>
        (other.x !== hand.x || other.y !== hand.y) &&
        hand.y - other.y >= FOOT_BELOW_MIN &&
        hand.y - other.y <= FOOT_BELOW_MAX &&
        Math.abs(hand.x - other.x) <= FOOT_SIDEWAYS,
    );
    if (!supported) unsupported += 1;
  }
  return [maxStep, maxGap, unsupported];
}

function bandIndex(bands: number[][], value: number): number {
  return bands.findIndex(([low, high]) => low <= value && value <= high);
}

export function exceedsLimits(
  features: [number, number, number],
  limits: GeometryLimits,
  vGrade: number,
  angle: number,
): boolean {
  const band = limits.bands[`${bandIndex(limits.gradeBands, vGrade)}-${bandIndex(limits.angleBands, angle)}`];
  return band !== undefined && features.some((value, index) => value > band[index]);
}
