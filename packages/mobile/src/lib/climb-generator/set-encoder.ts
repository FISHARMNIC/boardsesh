/**
 * Pure-TS forward pass for `AttentionGrader` in ml/climbgen/grader.py (v3). It is
 * the pairwise grader (16 grade logits) and the realism critic (1 logit).
 *
 * Per hold: [placement embedding, role embedding, x, y, angle] -> linear -> a few
 * pre-LN self-attention blocks in which every hold attends to every other ->
 * final LayerNorm -> mean / max / sum pooling (+ hold count, angle) -> two-layer
 * head. Mirrors the PyTorch module exactly, including tanh-GELU and LayerNorm eps.
 */

import { decodeBase64 } from './transformer';

type ExportedTensor = { shape: number[]; base64: string };

export type SetEncoderExport = {
  maxAngle: number;
  edges: { left?: number; right: number; bottom?: number; top: number };
  tensors: Record<string, ExportedTensor>;
};

export type EncoderHold = { placementIndex: number; roleIndex: number };

type Linear = { weight: Float32Array; bias: Float32Array; outFeatures: number };
type Norm = { gain: Float32Array; bias: Float32Array };
type Block = { ln1: Norm; qkv: Linear; proj: Linear; ln2: Norm; fc1: Linear; fc2: Linear };

const GELU_COEFFICIENT = Math.sqrt(2 / Math.PI);
const HEADS = 4;

function tensor(entry: ExportedTensor | undefined, name: string): Float32Array {
  if (!entry) throw new Error(`climb generator: missing tensor ${name}`);
  const bytes = decodeBase64(entry.base64);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

function linearAt(tensors: Record<string, ExportedTensor>, name: string): Linear {
  const bias = tensor(tensors[`${name}.bias`], `${name}.bias`);
  return { weight: tensor(tensors[`${name}.weight`], `${name}.weight`), bias, outFeatures: bias.length };
}

function normAt(tensors: Record<string, ExportedTensor>, name: string): Norm {
  return { gain: tensor(tensors[`${name}.weight`], `${name}.weight`), bias: tensor(tensors[`${name}.bias`], name) };
}

function apply(layer: Linear, input: Float32Array): Float32Array {
  const inFeatures = input.length;
  const out = new Float32Array(layer.outFeatures);
  for (let row = 0; row < layer.outFeatures; row += 1) {
    let sum = layer.bias[row];
    const base = row * inFeatures;
    for (let column = 0; column < inFeatures; column += 1) sum += layer.weight[base + column] * input[column];
    out[row] = sum;
  }
  return out;
}

function layerNorm(input: Float32Array, norm: Norm): Float32Array {
  let mean = 0;
  for (let index = 0; index < input.length; index += 1) mean += input[index];
  mean /= input.length;
  let variance = 0;
  for (let index = 0; index < input.length; index += 1) variance += (input[index] - mean) ** 2;
  variance /= input.length;
  const inverse = 1 / Math.sqrt(variance + 1e-5);
  const out = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    out[index] = (input[index] - mean) * inverse * norm.gain[index] + norm.bias[index];
  }
  return out;
}

export class AttentionSetEncoder {
  private readonly placementEmbedding: Float32Array;
  private readonly placementDim: number;
  private readonly roleEmbedding: Float32Array;
  private readonly roleDim: number;
  private readonly input: Linear;
  private readonly blocks: Block[];
  private readonly finalNorm: Norm;
  private readonly head1: Linear;
  private readonly head2: Linear;

  constructor(
    private readonly exported: SetEncoderExport,
    /** [placementId, x, y] per model placement index, from the generator's meta. */
    private readonly placements: number[][],
  ) {
    const { tensors } = exported;
    this.placementEmbedding = tensor(tensors['placement.weight'], 'placement.weight');
    this.placementDim = tensors['placement.weight'].shape[1];
    this.roleEmbedding = tensor(tensors['role.weight'], 'role.weight');
    this.roleDim = tensors['role.weight'].shape[1];
    this.input = linearAt(tensors, 'inp');
    this.blocks = [];
    for (let layer = 0; `blocks.${layer}.qkv.weight` in tensors; layer += 1) {
      const prefix = `blocks.${layer}.`;
      this.blocks.push({
        ln1: normAt(tensors, `${prefix}ln1`),
        qkv: linearAt(tensors, `${prefix}qkv`),
        proj: linearAt(tensors, `${prefix}proj`),
        ln2: normAt(tensors, `${prefix}ln2`),
        fc1: linearAt(tensors, `${prefix}fc1`),
        fc2: linearAt(tensors, `${prefix}fc2`),
      });
    }
    this.finalNorm = normAt(tensors, 'ln');
    this.head1 = linearAt(tensors, 'head1');
    this.head2 = linearAt(tensors, 'head2');
  }

  /** The raw output logits for one climb (16 for the grader, 1 for the critic). */
  logits(holds: EncoderHold[], angle: number): Float32Array {
    const { maxAngle, edges } = this.exported;
    const left = edges.left ?? 0;
    const bottom = edges.bottom ?? 0;
    const angleFeature = angle / maxAngle;
    const dim = this.input.outFeatures;

    let states = holds.map(({ placementIndex, roleIndex }) => {
      const [, x, y] = this.placements[placementIndex];
      const features = new Float32Array(this.placementDim + this.roleDim + 3);
      features.set(
        this.placementEmbedding.subarray(placementIndex * this.placementDim, (placementIndex + 1) * this.placementDim),
      );
      features.set(
        this.roleEmbedding.subarray(roleIndex * this.roleDim, (roleIndex + 1) * this.roleDim),
        this.placementDim,
      );
      features[this.placementDim + this.roleDim] = (x - left) / (edges.right - left);
      features[this.placementDim + this.roleDim + 1] = (y - bottom) / (edges.top - bottom);
      features[this.placementDim + this.roleDim + 2] = angleFeature;
      return apply(this.input, features);
    });

    const headDim = dim / HEADS;
    const scale = 1 / Math.sqrt(headDim);
    for (const block of this.blocks) {
      const qkv = states.map((state) => apply(block.qkv, layerNorm(state, block.ln1)));
      states = states.map((state, self) => {
        const attended = new Float32Array(dim);
        const scores = new Float64Array(holds.length);
        for (let head = 0; head < HEADS; head += 1) {
          const offset = head * headDim;
          let maxScore = -Infinity;
          for (let other = 0; other < holds.length; other += 1) {
            let score = 0;
            for (let index = 0; index < headDim; index += 1) {
              score += qkv[self][offset + index] * qkv[other][dim + offset + index];
            }
            scores[other] = score * scale;
            if (scores[other] > maxScore) maxScore = scores[other];
          }
          let total = 0;
          for (let other = 0; other < holds.length; other += 1) {
            scores[other] = Math.exp(scores[other] - maxScore);
            total += scores[other];
          }
          for (let other = 0; other < holds.length; other += 1) {
            const weight = scores[other] / total;
            for (let index = 0; index < headDim; index += 1) {
              attended[offset + index] += weight * qkv[other][2 * dim + offset + index];
            }
          }
        }
        const projected = apply(block.proj, attended);
        const afterAttention = new Float32Array(dim);
        for (let index = 0; index < dim; index += 1) afterAttention[index] = state[index] + projected[index];
        const expanded = apply(block.fc1, layerNorm(afterAttention, block.ln2));
        for (let index = 0; index < expanded.length; index += 1) {
          const value = expanded[index];
          expanded[index] = 0.5 * value * (1 + Math.tanh(GELU_COEFFICIENT * (value + 0.044715 * value ** 3)));
        }
        const contracted = apply(block.fc2, expanded);
        const next = new Float32Array(dim);
        for (let index = 0; index < dim; index += 1) next[index] = afterAttention[index] + contracted[index];
        return next;
      });
    }

    const pooled = new Float32Array(3 * dim + 2);
    const maximum = new Float32Array(dim).fill(-Infinity);
    for (const state of states) {
      const normalized = layerNorm(state, this.finalNorm);
      for (let index = 0; index < dim; index += 1) {
        pooled[index] += normalized[index];
        if (normalized[index] > maximum[index]) maximum[index] = normalized[index];
      }
    }
    const count = Math.max(1, holds.length);
    for (let index = 0; index < dim; index += 1) {
      const sum = pooled[index];
      pooled[index] = sum / count;
      pooled[dim + index] = maximum[index];
      pooled[2 * dim + index] = sum / 10;
    }
    pooled[3 * dim] = holds.length / 30;
    pooled[3 * dim + 1] = angleFeature;
    const hidden = apply(this.head1, pooled);
    for (let index = 0; index < hidden.length; index += 1) if (hidden[index] < 0) hidden[index] = 0;
    return apply(this.head2, hidden);
  }
}
