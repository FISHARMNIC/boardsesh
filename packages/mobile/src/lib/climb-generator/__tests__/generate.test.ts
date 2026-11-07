import { describe, expect, it, vi } from 'vitest';
import { generateClimbs } from '../generate';
import { readFileSync } from 'node:fs';
import { climbGeneratorModelKey } from '../models/supported';

// Every model downloads, so the test runs a tiny random one (ml/climbgen/make_test_fixture.py)
// in place of the download. The decoding rules checked below hold whatever the weights are.
const FIXTURE = new URL('./fixtures/tiny-model/', import.meta.url);
const fixtureMeta = JSON.parse(readFileSync(new URL('model-meta.json', FIXTURE), 'utf8'));
vi.mock('../models/remote-config', () => ({ CLIMB_GENERATOR_MODELS_URL: 'https://models.test' }));
vi.mock('../remote-models', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../remote-models')>()),
  ensureRemoteModel: async (modelKey: string) => ({
    version: 'fixture',
    files: {
      // MoonBoards have start, hand and finish only; the same tiny model stands in for one.
      meta: new TextEncoder().encode(
        JSON.stringify(modelKey.startsWith('moonboard') ? { ...fixtureMeta, roles: [42, 43, 44] } : fixtureMeta),
      ),
      weights: new Uint8Array(readFileSync(new URL('weights.bin', FIXTURE))),
      grader: new Uint8Array(readFileSync(new URL('grader.json', FIXTURE))),
    },
  }),
}));

function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
}

type TestMeta = { placements: number[][]; roles: number[] };

describe('generateClimbs', () => {
  it('only writes valid climbs on the holds it was allowed', async () => {
    const modelKey = 'kilter-1';
    const meta = fixtureMeta as TestMeta;
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

describe('a board with three roles (MoonBoard)', () => {
  it('never writes a foot hold, even from random weights', async () => {
    const allowed = new Set<number>(fixtureMeta.placements.map(([placementId]: number[]) => placementId));
    for (const seed of [1, 2, 3]) {
      const climbs = await generateClimbs({
        modelKey: 'moonboard-2',
        angle: 40,
        vGrade: 6,
        tags: [],
        allowedPlacementIds: allowed,
        candidates: 8,
        keep: 8,
        random: seededRandom(seed),
      });
      for (const climb of climbs) {
        for (const [, role] of climb.frames.matchAll(/p\d+r(\d+)/g)) expect([42, 43, 44]).toContain(Number(role));
      }
    }
  });
});

describe('climbGeneratorModelKey', () => {
  it('names the model for a supported board layout and nothing for the rest', () => {
    expect(climbGeneratorModelKey('kilter', 1)).toBe('kilter-1');
    expect(climbGeneratorModelKey('tension', 9)).toBe('tension-9');
    expect(climbGeneratorModelKey('tension', 10)).toBe('tension-10');
    expect(climbGeneratorModelKey('tension', 11)).toBe('tension-11');
    expect(climbGeneratorModelKey('kilter', 8)).toBe('kilter-8');
    expect(climbGeneratorModelKey('moonboard', 2)).toBe('moonboard-2');
    expect(climbGeneratorModelKey('kilter', 5)).toBeNull();
    expect(climbGeneratorModelKey('moonboard', 1)).toBeNull();
    expect(climbGeneratorModelKey('moonboard', 6)).toBeNull();
  });
});
