/**
 * Climb generator models that are downloaded instead of bundled.
 *
 * One `manifest.json` at the models URL lists every board's current version and
 * its three files with their SHA-256. A board's files are fetched the first time
 * its generator opens, checked against the manifest, and kept on the phone; after
 * that the generator works offline. The last manifest seen is kept too, so an
 * offline launch can still find the cached files.
 *
 * Network and disk are passed in (`RemoteModelDeps`) so this logic is testable;
 * remote-model-store.ts is the on-phone implementation.
 */

import { sha256Hex } from './sha256';

export type RemoteModelFileName = 'meta' | 'weights' | 'grader';

type ManifestFile = { path: string; sha256: string; bytes: number };

export type RemoteModelEntry = {
  version: string;
  files: Record<RemoteModelFileName, ManifestFile>;
};

export type ClimbGeneratorManifest = {
  formatVersion: 1;
  models: Record<string, RemoteModelEntry>;
};

export type RemoteModelFiles = {
  version: string;
  meta: unknown;
  weights: Uint8Array;
  grader: unknown;
};

export interface RemoteModelStore {
  readManifest(): Promise<string | null>;
  writeManifest(json: string): Promise<void>;
  readFile(modelKey: string, version: string, name: RemoteModelFileName): Promise<Uint8Array | null>;
  /** Must be atomic: a reader never sees a half-written file. */
  writeFile(modelKey: string, version: string, name: RemoteModelFileName, bytes: Uint8Array): Promise<void>;
  /** Drop every cached version of `modelKey` except `keepVersion`. */
  pruneVersions(modelKey: string, keepVersion: string): Promise<void>;
}

export type RemoteModelDeps = {
  baseUrl: string;
  fetchBytes: (url: string) => Promise<Uint8Array>;
  store: RemoteModelStore;
};

const FILE_NAMES: RemoteModelFileName[] = ['meta', 'weights', 'grader'];

export class RemoteModelError extends Error {
  constructor(
    message: string,
    readonly reason: 'unavailable' | 'offline' | 'corrupt',
  ) {
    super(message);
    this.name = 'RemoteModelError';
  }
}

function isManifestFile(value: unknown): value is ManifestFile {
  if (typeof value !== 'object' || value === null) return false;
  const file = value as Record<string, unknown>;
  return (
    typeof file.path === 'string' &&
    !file.path.includes('..') &&
    typeof file.sha256 === 'string' &&
    /^[0-9a-f]{64}$/.test(file.sha256) &&
    typeof file.bytes === 'number'
  );
}

/** Parses a manifest, throwing on anything that does not match the format. */
export function parseManifest(json: string): ClimbGeneratorManifest {
  const parsed: unknown = JSON.parse(json);
  if (typeof parsed !== 'object' || parsed === null) throw new Error('manifest is not an object');
  const { formatVersion, models } = parsed as Record<string, unknown>;
  if (formatVersion !== 1) throw new Error(`unsupported manifest formatVersion ${String(formatVersion)}`);
  if (typeof models !== 'object' || models === null) throw new Error('manifest has no models');
  for (const [key, entry] of Object.entries(models as Record<string, unknown>)) {
    const { version, files } = (entry ?? {}) as Record<string, unknown>;
    if (typeof version !== 'string' || !/^[\w.-]+$/.test(version)) throw new Error(`model ${key} has a bad version`);
    if (typeof files !== 'object' || files === null) throw new Error(`model ${key} has no files`);
    for (const name of FILE_NAMES) {
      if (!isManifestFile((files as Record<string, unknown>)[name])) throw new Error(`model ${key} lacks ${name}`);
    }
  }
  return parsed as ClimbGeneratorManifest;
}

const textDecoder = new TextDecoder();

function decodeFiles(version: string, bytes: Record<RemoteModelFileName, Uint8Array>): RemoteModelFiles {
  return {
    version,
    meta: JSON.parse(textDecoder.decode(bytes.meta)),
    weights: bytes.weights,
    grader: JSON.parse(textDecoder.decode(bytes.grader)),
  };
}

async function readCachedFiles(
  store: RemoteModelStore,
  modelKey: string,
  entry: RemoteModelEntry,
): Promise<Record<RemoteModelFileName, Uint8Array> | null> {
  const result: Partial<Record<RemoteModelFileName, Uint8Array>> = {};
  for (const name of FILE_NAMES) {
    const bytes = await store.readFile(modelKey, entry.version, name);
    if (!bytes || sha256Hex(bytes) !== entry.files[name].sha256) return null;
    result[name] = bytes;
  }
  return result as Record<RemoteModelFileName, Uint8Array>;
}

/**
 * The newest manifest we can get: the network's if it answers (and then saved),
 * else the one saved last time, else null.
 */
async function currentManifest(deps: RemoteModelDeps): Promise<ClimbGeneratorManifest | null> {
  try {
    const bytes = await deps.fetchBytes(`${deps.baseUrl}/manifest.json`);
    const json = textDecoder.decode(bytes);
    const manifest = parseManifest(json);
    await deps.store.writeManifest(json);
    return manifest;
  } catch {
    const saved = await deps.store.readManifest();
    if (!saved) return null;
    try {
      return parseManifest(saved);
    } catch {
      return null;
    }
  }
}

/**
 * The files for `modelKey`: from the phone if the current version is already
 * there and intact, else downloaded, checked and saved. Throws a
 * `RemoteModelError` saying whether the model does not exist, the phone is
 * offline with nothing cached, or a download did not match its checksum.
 */
export async function ensureRemoteModel(modelKey: string, deps: RemoteModelDeps): Promise<RemoteModelFiles> {
  const manifest = await currentManifest(deps);
  if (!manifest) throw new RemoteModelError('Could not reach the climb generator models.', 'offline');
  const entry = manifest.models[modelKey];
  if (!entry) throw new RemoteModelError(`No climb generator model for ${modelKey}.`, 'unavailable');

  const cached = await readCachedFiles(deps.store, modelKey, entry);
  if (cached) return decodeFiles(entry.version, cached);

  const downloaded: Partial<Record<RemoteModelFileName, Uint8Array>> = {};
  for (const name of FILE_NAMES) {
    const file = entry.files[name];
    let bytes: Uint8Array;
    try {
      bytes = await deps.fetchBytes(`${deps.baseUrl}/${file.path}`);
    } catch {
      throw new RemoteModelError('Could not download the climb generator model.', 'offline');
    }
    if (sha256Hex(bytes) !== file.sha256) {
      throw new RemoteModelError(`The downloaded ${name} file did not match its checksum.`, 'corrupt');
    }
    downloaded[name] = bytes;
  }
  for (const name of FILE_NAMES) {
    await deps.store.writeFile(modelKey, entry.version, name, downloaded[name] as Uint8Array);
  }
  await deps.store.pruneVersions(modelKey, entry.version);
  return decodeFiles(entry.version, downloaded as Record<RemoteModelFileName, Uint8Array>);
}
