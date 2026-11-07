import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { sampleFrom } from '../generate';
import { climbFeatures, exceedsLimits, type GeometryLimits } from '../geometry';
import { KnownClimbs } from '../known-climbs';
import { AttentionSetEncoder, type SetEncoderExport } from '../set-encoder';

// Reference outputs from PyTorch (the critic) and ml/climbgen/features.py (geometry),
// written by ml/climbgen/make_test_fixture.py for the same tiny model.
const FIXTURE = new URL('./fixtures/tiny-model/', import.meta.url);
const meta = JSON.parse(readFileSync(new URL('model-meta.json', FIXTURE), 'utf8')) as { placements: number[][] };
const critic = JSON.parse(readFileSync(new URL('critic.json', FIXTURE), 'utf8')) as SetEncoderExport;
const parity = JSON.parse(readFileSync(new URL('parity.json', FIXTURE), 'utf8')) as {
  cases: { holds: number[][]; angle: number; criticLogit: number; features: number[] }[];
};
const toHolds = (holds: number[][]) => holds.map(([placementIndex, roleIndex]) => ({ placementIndex, roleIndex }));

describe('AttentionSetEncoder (the v3 critic and pairwise grader)', () => {
  it('matches PyTorch on every reference climb', () => {
    const encoder = new AttentionSetEncoder(critic, meta.placements);
    for (const testCase of parity.cases) {
      const [logit] = encoder.logits(toHolds(testCase.holds), testCase.angle);
      expect(logit).toBeCloseTo(testCase.criticLogit, 4);
    }
  });
});

describe('climbFeatures', () => {
  it('matches ml/climbgen/features.py on every reference climb', () => {
    for (const testCase of parity.cases) {
      const [step, gap, unsupported] = climbFeatures(toHolds(testCase.holds), meta.placements);
      expect(step).toBeCloseTo(testCase.features[0], 5);
      expect(gap).toBeCloseTo(testCase.features[1], 5);
      expect(unsupported).toBe(testCase.features[2]);
    }
  });

  it('rejects only climbs beyond their grade and angle band', () => {
    const limits: GeometryLimits = {
      percentile: 99,
      gradeBands: [
        [0, 4],
        [5, 7],
      ],
      angleBands: [
        [0, 30],
        [35, 45],
      ],
      bands: { '1-1': [90, 40, 3] },
    };
    expect(exceedsLimits([80, 40, 3], limits, 6, 40)).toBe(false);
    expect(exceedsLimits([95, 40, 3], limits, 6, 40)).toBe(true);
    expect(exceedsLimits([80, 40, 4], limits, 6, 40)).toBe(true);
    // No band measured there (too few real climbs): nothing is rejected.
    expect(exceedsLimits([500, 500, 9], limits, 2, 40)).toBe(false);
  });
});

describe('KnownClimbs', () => {
  function index(records: number[][]): KnownClimbs {
    const words = records.flatMap((holds) => [holds.length, ...holds]);
    const bytes = new Uint8Array(words.length * 2);
    const view = new DataView(bytes.buffer);
    words.forEach((word, position) => view.setUint16(position * 2, word, true));
    return new KnownClimbs(bytes, 20);
  }

  it('finds the closest known climb by shared holds', () => {
    const known = index([
      [1, 2, 3, 4, 5],
      [10, 11, 12],
    ]);
    expect(known.count).toBe(2);
    expect(known.maxJaccard([1, 2, 3, 4, 5])).toBe(1);
    expect(known.maxJaccard([1, 2, 3, 4, 6])).toBeCloseTo(4 / 6);
    expect(known.maxJaccard([15, 16])).toBe(0);
  });

  it('gives the same answer when asked twice (its scratch counts reset)', () => {
    const known = index([[1, 2, 3]]);
    expect(known.maxJaccard([1, 2])).toBeCloseTo(2 / 3);
    expect(known.maxJaccard([1, 2])).toBeCloseTo(2 / 3);
  });
});

describe('sampleFrom', () => {
  const logits = new Float32Array([3, 2, 1, 0]);
  const allowAll = () => true;

  it('with top-p keeps only the likeliest tokens', () => {
    // Token 0 holds about 64% of the probability; a top-p of 0.5 keeps only it.
    for (const draw of [0.01, 0.5, 0.99]) {
      expect(sampleFrom(logits, allowAll, 1, () => draw, 0.5)).toBe(0);
    }
  });

  it('without top-p can pick any allowed token', () => {
    expect(sampleFrom(logits, allowAll, 1, () => 0.999, 1)).toBe(3);
    expect(
      sampleFrom(
        logits,
        (token) => token !== 3,
        1,
        () => 0.999,
        1,
      ),
    ).toBe(2);
  });
});
