import { describe, expect, it } from 'vitest';
import { generateClimbs } from '../generate';
import { BUNDLED_CLIMB_GENERATOR_MODELS } from '../models';
import { climbGeneratorModelKey, type ClimbGeneratorModelKey } from '../models/supported';

// Only bundled models can run here; downloaded ones are covered in remote-models.test.ts.
const BUNDLED_KEYS = Object.keys(BUNDLED_CLIMB_GENERATOR_MODELS) as ClimbGeneratorModelKey[];

function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

type TestMeta = { placements: number[][]; roles: number[] };

describe('generateClimbs', () => {
  it.each(BUNDLED_KEYS)('only writes valid %s climbs on the holds it was allowed', async (modelKey) => {
    const meta = BUNDLED_CLIMB_GENERATOR_MODELS[modelKey]?.meta as TestMeta;
    const [startRole, , finishRole] = meta.roles;
    // Half the board, so the mask is doing real work.
    const allowed = new Set(meta.placements.map(([placementId]) => placementId).filter((_, index) => index % 2 === 0));
    const climbs = await generateClimbs({
      modelKey,
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
        expect(meta.roles).toContain(role);
      }
      const starts = holds.filter(([, role]) => role === startRole).length;
      const finishes = holds.filter(([, role]) => role === finishRole).length;
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

describe('climbGeneratorModelKey', () => {
  it('names the model for a supported board layout and nothing for the rest', () => {
    expect(climbGeneratorModelKey('kilter', 1)).toBe('kilter-1');
    expect(climbGeneratorModelKey('tension', 9)).toBe('tension-9');
    expect(climbGeneratorModelKey('tension', 10)).toBe('tension-10');
    expect(climbGeneratorModelKey('tension', 11)).toBe('tension-11');
    expect(climbGeneratorModelKey('kilter', 8)).toBeNull();
    expect(climbGeneratorModelKey('moonboard', 1)).toBeNull();
  });

  it('ships Kilter Original in the app', () => {
    expect(BUNDLED_KEYS).toEqual(['kilter-1']);
  });
});
