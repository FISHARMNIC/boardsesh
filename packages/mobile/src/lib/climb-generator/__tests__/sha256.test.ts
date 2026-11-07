import { createHash } from 'node:crypto';
import { describe, expect, it } from 'vitest';
import { sha256Hex } from '../sha256';

describe('sha256Hex', () => {
  it.each([0, 1, 55, 56, 63, 64, 65, 1000, 70_000])('matches node for %i bytes', (length) => {
    const bytes = new Uint8Array(length).map((_, index) => (index * 31 + 7) & 0xff);
    expect(sha256Hex(bytes)).toBe(createHash('sha256').update(bytes).digest('hex'));
  });
});
