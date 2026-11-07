/**
 * Pure-TS forward pass for the climb grader trained by ml/climbgen/grader.py.
 * Mirrors `Grader` there: per-hold MLP over [placement embedding, role
 * embedding, x, y, angle], mean/max/sum pooling, then a head that scores each
 * V-grade. The exported logit correction undoes the training class weights,
 * and the temperature calibrates the probabilities.
 */

import { decodeBase64 } from './transformer';

type GraderTensor = { shape: number[]; base64: string };

export type GraderExport = {
  numGrades: number;
  coverage: number;
  maxAngle: number;
  /** The wall the grader was trained on; `left`/`bottom` absent on older exports mean 0. */
  edges: { left?: number; right: number; bottom?: number; top: number };
  temperature: number;
  logitCorrection: number[];
  tensors: Record<string, GraderTensor>;
};

export type GradeEstimate = {
  /** Probability for each V-grade, V0 first. */
  probabilities: Float64Array;
  /** Median grade: half the probability is at or below it. */
  medianV: number;
  /** Narrowest run of neighbouring grades holding `coverage` of the probability. */
  lowV: number;
  highV: number;
};

export type GraderHold = { placementIndex: number; roleIndex: number };

function tensorFrom(entry: GraderTensor): Float32Array {
  const bytes = decodeBase64(entry.base64);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function dense(input: Float32Array, weight: Float32Array, bias: Float32Array, relu: boolean): Float32Array {
  const outFeatures = bias.length;
  const inFeatures = input.length;
  const out = new Float32Array(outFeatures);
  for (let row = 0; row < outFeatures; row += 1) {
    let sum = bias[row];
    const base = row * inFeatures;
    for (let column = 0; column < inFeatures; column += 1) sum += weight[base + column] * input[column];
    out[row] = relu && sum < 0 ? 0 : sum;
  }
  return out;
}

export class ClimbGrader {
  private readonly placementEmbedding: Float32Array;
  private readonly placementDim: number;
  private readonly roleEmbedding: Float32Array;
  private readonly roleDim: number;
  private readonly weights: Record<'node1' | 'node2' | 'head1' | 'head2', { weight: Float32Array; bias: Float32Array }>;

  constructor(
    private readonly exported: GraderExport,
    /** [placementId, x, y] per model placement index, from the generator's meta. */
    private readonly placements: number[][],
  ) {
    const { tensors } = exported;
    this.placementEmbedding = tensorFrom(tensors['placement.weight']);
    this.placementDim = tensors['placement.weight'].shape[1];
    this.roleEmbedding = tensorFrom(tensors['role.weight']);
    this.roleDim = tensors['role.weight'].shape[1];
    const layer = (name: string) => ({
      weight: tensorFrom(tensors[`${name}.weight`]),
      bias: tensorFrom(tensors[`${name}.bias`]),
    });
    this.weights = { node1: layer('node1'), node2: layer('node2'), head1: layer('head1'), head2: layer('head2') };
  }

  grade(holds: GraderHold[], angle: number): GradeEstimate {
    const { maxAngle, edges, temperature, logitCorrection, numGrades, coverage } = this.exported;
    const angleFeature = angle / maxAngle;
    const hidden = this.weights.node2.bias.length;
    const mean = new Float32Array(hidden);
    const maximum = new Float32Array(hidden).fill(-Infinity);
    for (const { placementIndex, roleIndex } of holds) {
      const [, x, y] = this.placements[placementIndex];
      const input = new Float32Array(this.placementDim + this.roleDim + 3);
      input.set(
        this.placementEmbedding.subarray(placementIndex * this.placementDim, (placementIndex + 1) * this.placementDim),
      );
      input.set(
        this.roleEmbedding.subarray(roleIndex * this.roleDim, (roleIndex + 1) * this.roleDim),
        this.placementDim,
      );
      const left = edges.left ?? 0;
      const bottom = edges.bottom ?? 0;
      input[this.placementDim + this.roleDim] = (x - left) / (edges.right - left);
      input[this.placementDim + this.roleDim + 1] = (y - bottom) / (edges.top - bottom);
      input[this.placementDim + this.roleDim + 2] = angleFeature;
      const first = dense(input, this.weights.node1.weight, this.weights.node1.bias, true);
      const node = dense(first, this.weights.node2.weight, this.weights.node2.bias, true);
      for (let index = 0; index < hidden; index += 1) {
        mean[index] += node[index];
        if (node[index] > maximum[index]) maximum[index] = node[index];
      }
    }
    const count = Math.max(1, holds.length);
    const pooled = new Float32Array(3 * hidden + 2);
    for (let index = 0; index < hidden; index += 1) {
      pooled[index] = mean[index] / count;
      pooled[hidden + index] = maximum[index];
      pooled[2 * hidden + index] = mean[index] / 10;
    }
    pooled[3 * hidden] = holds.length / 30;
    pooled[3 * hidden + 1] = angleFeature;
    const head = dense(pooled, this.weights.head1.weight, this.weights.head1.bias, true);
    const logits = dense(head, this.weights.head2.weight, this.weights.head2.bias, false);

    const probabilities = new Float64Array(numGrades);
    let maxLogit = -Infinity;
    for (let grade = 0; grade < numGrades; grade += 1) {
      logits[grade] = (logits[grade] + logitCorrection[grade]) / temperature;
      if (logits[grade] > maxLogit) maxLogit = logits[grade];
    }
    let total = 0;
    for (let grade = 0; grade < numGrades; grade += 1) {
      probabilities[grade] = Math.exp(logits[grade] - maxLogit);
      total += probabilities[grade];
    }
    for (let grade = 0; grade < numGrades; grade += 1) probabilities[grade] /= total;

    let cumulative = 0;
    let medianV = numGrades - 1;
    for (let grade = 0; grade < numGrades; grade += 1) {
      cumulative += probabilities[grade];
      if (cumulative >= 0.5) {
        medianV = grade;
        break;
      }
    }
    let lowV = 0;
    let highV = numGrades - 1;
    for (let low = 0; low < numGrades; low += 1) {
      let mass = 0;
      for (let high = low; high < numGrades; high += 1) {
        mass += probabilities[high];
        if (mass >= coverage) {
          if (high - low < highV - lowV) {
            lowV = low;
            highV = high;
          }
          break;
        }
      }
    }
    return { probabilities, medianV, lowV, highV };
  }
}
