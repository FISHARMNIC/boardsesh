import { describe, expect, it } from 'vitest';
import meta from '../model-meta.generated.json';
import { generateClimbs } from '../generate';

function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

const allPlacementIds = new Set(meta.placements.map(([placementId]) => placementId));

describe('generateClimbs', () => {
  it('only writes valid Kilter climbs on the holds it was allowed', async () => {
    // Half the board, so the mask is doing real work.
    const allowed = new Set([...allPlacementIds].filter((_, index) => index % 2 === 0));
    const climbs = await generateClimbs({
      angle: 40,
      vGrade: 5,
      tags: ['crimp'],
      allowedPlacementIds: allowed,
      candidates: 6,
      keep: 6,
      random: seededRandom(7),
    });
    expect(climbs.length).toBeGreaterThan(0);
    for (const climb of climbs) {
      const holds = [...climb.frames.matchAll(/p(\d+)r(\d+)/g)].map(([, placement, role]) => [
        Number(placement),
        Number(role),
      ]);
      expect(holds.length).toBe(climb.holdCount);
      expect(holds.length).toBeGreaterThanOrEqual(3);
      expect(new Set(holds.map(([placement]) => placement)).size).toBe(holds.length);
      for (const [placement, role] of holds) {
        expect(allowed.has(placement)).toBe(true);
        expect([12, 13, 14, 15]).toContain(role);
      }
      const starts = holds.filter(([, role]) => role === 12).length;
      const finishes = holds.filter(([, role]) => role === 14).length;
      expect(starts).toBeGreaterThanOrEqual(1);
      expect(starts).toBeLessThanOrEqual(2);
      expect(finishes).toBeGreaterThanOrEqual(1);
      expect(finishes).toBeLessThanOrEqual(2);
      expect(climb.lowV).toBeLessThanOrEqual(climb.medianV);
      expect(climb.medianV).toBeLessThanOrEqual(climb.highV);
      expect(climb.targetProbability).toBeGreaterThanOrEqual(0);
      expect(climb.targetProbability).toBeLessThanOrEqual(1);
    }
  });
});
