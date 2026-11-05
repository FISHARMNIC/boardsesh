/**
 * Pure-TS forward pass for the tiny climb generator trained by
 * ml/climbgen/train.py. Mirrors `ClimbGen` there exactly: pre-LN blocks,
 * causal multi-head attention, tanh-GELU MLP, final LayerNorm, logits tied to
 * the token embedding. Runs one token at a time with a KV cache.
 */

export type ModelConfig = {
  dim: number;
  layers: number;
  heads: number;
  ff: number;
  vocabSize: number;
  maxLen: number;
};

export type TensorEntry = {
  name: string;
  shape: readonly number[];
  dtype: 'int8' | 'float32';
  offset: number;
  scaleOffset?: number;
};

type Weights = Map<string, Float32Array>;

const BASE64_ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

export function decodeBase64(encoded: string): Uint8Array {
  const lookup = new Uint8Array(128);
  for (let index = 0; index < BASE64_ALPHABET.length; index += 1) lookup[BASE64_ALPHABET.charCodeAt(index)] = index;
  const padding = encoded.endsWith('==') ? 2 : encoded.endsWith('=') ? 1 : 0;
  const bytes = new Uint8Array((encoded.length * 3) / 4 - padding);
  let byteIndex = 0;
  for (let charIndex = 0; charIndex < encoded.length; charIndex += 4) {
    const chunk =
      (lookup[encoded.charCodeAt(charIndex)] << 18) |
      (lookup[encoded.charCodeAt(charIndex + 1)] << 12) |
      (lookup[encoded.charCodeAt(charIndex + 2)] << 6) |
      lookup[encoded.charCodeAt(charIndex + 3)];
    if (byteIndex < bytes.length) bytes[byteIndex++] = (chunk >> 16) & 0xff;
    if (byteIndex < bytes.length) bytes[byteIndex++] = (chunk >> 8) & 0xff;
    if (byteIndex < bytes.length) bytes[byteIndex++] = chunk & 0xff;
  }
  return bytes;
}

/** Dequantizes the int8-per-row blob into Float32 tensors. */
export function loadWeights(blob: Uint8Array, tensors: readonly TensorEntry[]): Weights {
  const weights: Weights = new Map();
  const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  for (const tensor of tensors) {
    const size = tensor.shape.reduce((product, dimension) => product * dimension, 1);
    const out = new Float32Array(size);
    if (tensor.dtype === 'float32') {
      for (let index = 0; index < size; index += 1) out[index] = view.getFloat32(tensor.offset + index * 4, true);
    } else {
      const rows = tensor.shape[0];
      const columns = size / rows;
      const scaleOffset = tensor.scaleOffset ?? 0;
      for (let row = 0; row < rows; row += 1) {
        const scale = view.getFloat32(scaleOffset + row * 4, true);
        for (let column = 0; column < columns; column += 1) {
          const flat = row * columns + column;
          out[flat] = view.getInt8(tensor.offset + flat) * scale;
        }
      }
    }
    weights.set(tensor.name, out);
  }
  return weights;
}

function requireTensor(weights: Weights, name: string): Float32Array {
  const tensor = weights.get(name);
  if (!tensor) throw new Error(`climb generator: missing tensor ${name}`);
  return tensor;
}

/** out = W·input + bias, with W stored [outFeatures, inFeatures]. */
function linear(input: Float32Array, weight: Float32Array, bias: Float32Array, outFeatures: number): Float32Array {
  const inFeatures = input.length;
  const out = new Float32Array(outFeatures);
  for (let row = 0; row < outFeatures; row += 1) {
    let sum = bias[row];
    const base = row * inFeatures;
    for (let column = 0; column < inFeatures; column += 1) sum += weight[base + column] * input[column];
    out[row] = sum;
  }
  return out;
}

function layerNorm(input: Float32Array, gain: Float32Array, bias: Float32Array): Float32Array {
  let mean = 0;
  for (let index = 0; index < input.length; index += 1) mean += input[index];
  mean /= input.length;
  let variance = 0;
  for (let index = 0; index < input.length; index += 1) variance += (input[index] - mean) ** 2;
  variance /= input.length;
  const inverse = 1 / Math.sqrt(variance + 1e-5);
  const out = new Float32Array(input.length);
  for (let index = 0; index < input.length; index += 1) {
    out[index] = (input[index] - mean) * inverse * gain[index] + bias[index];
  }
  return out;
}

const GELU_COEFFICIENT = Math.sqrt(2 / Math.PI);

function gelu(input: number): number {
  return 0.5 * input * (1 + Math.tanh(GELU_COEFFICIENT * (input + 0.044715 * input ** 3)));
}

type LayerWeights = {
  ln1Weight: Float32Array;
  ln1Bias: Float32Array;
  qkvWeight: Float32Array;
  qkvBias: Float32Array;
  projWeight: Float32Array;
  projBias: Float32Array;
  ln2Weight: Float32Array;
  ln2Bias: Float32Array;
  fc1Weight: Float32Array;
  fc1Bias: Float32Array;
  fc2Weight: Float32Array;
  fc2Bias: Float32Array;
};

export class ClimbTransformer {
  private readonly tokenEmbedding: Float32Array;
  private readonly positionEmbedding: Float32Array;
  private readonly layerWeights: LayerWeights[];
  private readonly finalGain: Float32Array;
  private readonly finalBias: Float32Array;

  constructor(
    private readonly config: ModelConfig,
    weights: Weights,
  ) {
    this.tokenEmbedding = requireTensor(weights, 'tok.weight');
    this.positionEmbedding = requireTensor(weights, 'pos.weight');
    this.finalGain = requireTensor(weights, 'ln_f.weight');
    this.finalBias = requireTensor(weights, 'ln_f.bias');
    this.layerWeights = Array.from({ length: config.layers }, (_, layer) => {
      const prefix = `blocks.${layer}.`;
      const tensor = (name: string) => requireTensor(weights, prefix + name);
      return {
        ln1Weight: tensor('ln1.weight'),
        ln1Bias: tensor('ln1.bias'),
        qkvWeight: tensor('qkv.weight'),
        qkvBias: tensor('qkv.bias'),
        projWeight: tensor('proj.weight'),
        projBias: tensor('proj.bias'),
        ln2Weight: tensor('ln2.weight'),
        ln2Bias: tensor('ln2.bias'),
        fc1Weight: tensor('fc1.weight'),
        fc1Bias: tensor('fc1.bias'),
        fc2Weight: tensor('fc2.weight'),
        fc2Bias: tensor('fc2.bias'),
      };
    });
  }

  /** A fresh decoding session with its own KV cache. */
  session(): DecodeSession {
    return new DecodeSession(this.config, this);
  }

  /** Internal: one step for `token` at `position`, appending to the cache. */
  step(token: number, position: number, keys: Float32Array[][], values: Float32Array[][]): Float32Array {
    const { dim, heads, ff, vocabSize } = this.config;
    const headDim = dim / heads;
    let hidden = new Float32Array(dim);
    for (let index = 0; index < dim; index += 1) {
      hidden[index] = this.tokenEmbedding[token * dim + index] + this.positionEmbedding[position * dim + index];
    }

    this.layerWeights.forEach((layer, layerIndex) => {
      const qkv = linear(layerNorm(hidden, layer.ln1Weight, layer.ln1Bias), layer.qkvWeight, layer.qkvBias, 3 * dim);
      const query = qkv.subarray(0, dim);
      keys[layerIndex].push(qkv.slice(dim, 2 * dim));
      values[layerIndex].push(qkv.slice(2 * dim, 3 * dim));
      const cachedKeys = keys[layerIndex];
      const cachedValues = values[layerIndex];

      const attended = new Float32Array(dim);
      const scores = new Float32Array(cachedKeys.length);
      const scale = 1 / Math.sqrt(headDim);
      for (let head = 0; head < heads; head += 1) {
        const offset = head * headDim;
        let maxScore = -Infinity;
        for (let past = 0; past < cachedKeys.length; past += 1) {
          let score = 0;
          for (let index = 0; index < headDim; index += 1) {
            score += query[offset + index] * cachedKeys[past][offset + index];
          }
          scores[past] = score * scale;
          if (scores[past] > maxScore) maxScore = scores[past];
        }
        let total = 0;
        for (let past = 0; past < cachedKeys.length; past += 1) {
          scores[past] = Math.exp(scores[past] - maxScore);
          total += scores[past];
        }
        for (let past = 0; past < cachedKeys.length; past += 1) {
          const weight = scores[past] / total;
          for (let index = 0; index < headDim; index += 1) {
            attended[offset + index] += weight * cachedValues[past][offset + index];
          }
        }
      }
      const projected = linear(attended, layer.projWeight, layer.projBias, dim);
      const afterAttention = new Float32Array(dim);
      for (let index = 0; index < dim; index += 1) afterAttention[index] = hidden[index] + projected[index];

      const normalizedAfterAttention = layerNorm(afterAttention, layer.ln2Weight, layer.ln2Bias);
      const expanded = linear(normalizedAfterAttention, layer.fc1Weight, layer.fc1Bias, ff);
      for (let index = 0; index < ff; index += 1) expanded[index] = gelu(expanded[index]);
      const contracted = linear(expanded, layer.fc2Weight, layer.fc2Bias, dim);
      const next = new Float32Array(dim);
      for (let index = 0; index < dim; index += 1) next[index] = afterAttention[index] + contracted[index];
      hidden = next;
    });

    const normalized = layerNorm(hidden, this.finalGain, this.finalBias);
    const logits = new Float32Array(vocabSize);
    for (let token = 0; token < vocabSize; token += 1) {
      let sum = 0;
      const base = token * dim;
      for (let index = 0; index < dim; index += 1) sum += this.tokenEmbedding[base + index] * normalized[index];
      logits[token] = sum;
    }
    return logits;
  }
}

export class DecodeSession {
  private position = 0;
  private readonly keys: Float32Array[][];
  private readonly values: Float32Array[][];

  constructor(
    private readonly config: ModelConfig,
    private readonly model: ClimbTransformer,
  ) {
    this.keys = Array.from({ length: config.layers }, () => []);
    this.values = Array.from({ length: config.layers }, () => []);
  }

  get length(): number {
    return this.position;
  }

  /** Feeds one token and returns the logits for the next one. */
  feed(token: number): Float32Array {
    if (this.position >= this.config.maxLen) throw new Error('climb generator: sequence too long');
    const logits = this.model.step(token, this.position, this.keys, this.values);
    this.position += 1;
    return logits;
  }
}
