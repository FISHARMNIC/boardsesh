/**
 * Every real climb's hold set, so v3 can drop a generated climb that is a
 * near-copy of an existing one (Jaccard >= 0.8 on the holds used).
 *
 * File format (`known-climbs.bin`, from ml/climbgen/package_models.py): little-
 * endian uint16s, one record per climb: the hold count, then that many model
 * placement indices. About 2 MB for Kilter Original.
 */

export class KnownClimbs {
  private readonly sizes: Uint8Array;
  /** Per placement index, the climbs that use it. */
  private readonly postings: Int32Array[];
  private readonly overlap: Uint16Array;

  constructor(bytes: Uint8Array, placementCount: number) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const words = bytes.byteLength >> 1;
    const climbHolds: number[][] = [];
    for (let offset = 0; offset < words;) {
      const count = view.getUint16(offset * 2, true);
      const holds: number[] = [];
      for (let index = 1; index <= count; index += 1) holds.push(view.getUint16((offset + index) * 2, true));
      climbHolds.push(holds);
      offset += count + 1;
    }
    this.sizes = Uint8Array.from(climbHolds, (holds) => holds.length);
    const lengths = new Int32Array(placementCount);
    for (const holds of climbHolds) for (const placement of holds) lengths[placement] += 1;
    this.postings = Array.from(lengths, (length) => new Int32Array(length));
    const fill = new Int32Array(placementCount);
    climbHolds.forEach((holds, climb) => {
      for (const placement of holds) this.postings[placement][fill[placement]++] = climb;
    });
    this.overlap = new Uint16Array(climbHolds.length);
  }

  get count(): number {
    return this.sizes.length;
  }

  /** The highest Jaccard similarity between `placementIndices` and any known climb. */
  maxJaccard(placementIndices: readonly number[]): number {
    const unique = [...new Set(placementIndices)];
    const touched: number[] = [];
    for (const placement of unique) {
      const list = this.postings[placement];
      if (!list) continue;
      for (let index = 0; index < list.length; index += 1) {
        const climb = list[index];
        if (this.overlap[climb] === 0) touched.push(climb);
        this.overlap[climb] += 1;
      }
    }
    let best = 0;
    for (const climb of touched) {
      const shared = this.overlap[climb];
      const score = shared / (unique.length + this.sizes[climb] - shared);
      if (score > best) best = score;
      this.overlap[climb] = 0;
    }
    return best;
  }
}
