/**
 * Climb generator models are downloaded, not bundled.
 *
 * Each model generation has its own folder at the models URL (`v2/`, `v3/`), whose
 * `manifest.json` lists every board's current version and its files with their SHA-256. A board's
 * files are fetched the first time its generator opens, checked against the
 * manifest, and kept on the phone; after that the generator works offline. The last
 * manifest seen is kept too, so an offline launch can still find the cached files.
 *
 * Format 1 (v2 models) has exactly `meta`, `weights` and `grader`. Format 2 (v3)
 * may add more named files (a critic, geometry limits, a known-climbs index); the
 * caller says which files it needs.
 *
 * Network and disk are passed in (`RemoteModelDeps`) so this logic is testable;
 * remote-model-store.ts is the on-phone implementation.
 */

import { sha256Hex } from './sha256';

type ManifestFile = { path: string; sha256: string; bytes: number };

export type RemoteModelEntry = {
  version: string;
  files: Record<string, ManifestFile>;
};

export type ClimbGeneratorManifest = {
  formatVersion: 1 | 2;
  models: Record<string, RemoteModelEntry>;
};

export type RemoteModelFiles = {
  version: string;
  files: Record<string, Uint8Array>;
};

export interface RemoteModelStore {
  readManifest(manifestPath: string): Promise<string | null>;
  writeManifest(manifestPath: string, json: string): Promise<void>;
  readFile(storeKey: string, version: string, name: string): Promise<Uint8Array | null>;
  /** Must be atomic: a reader never sees a half-written file. */
  writeFile(storeKey: string, version: string, name: string, bytes: Uint8Array): Promise<void>;
  /** Drop every cached version under `storeKey` except `keepVersion`. */
  pruneVersions(storeKey: string, keepVersion: string): Promise<void>;
}

export type RemoteModelDeps = {
  baseUrl: string;
  fetchBytes: (url: string) => Promise<Uint8Array>;
  store: RemoteModelStore;
  /** Relative to `baseUrl`, e.g. `v2/manifest.json`. Its folder namespaces the phone's copies. */
  manifestPath: string;
  /** Files this caller needs; each must be in the model's manifest entry. */
  requiredFiles?: readonly string[];
};

export const V2_MODEL_FILES = ['meta', 'weights', 'grader'] as const;
const FILE_NAME = /^[a-z][a-zA-Z0-9-]*$/;

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
    !file.path.startsWith('/') &&
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
  if (formatVersion !== 1 && formatVersion !== 2) {
    throw new Error(`unsupported manifest formatVersion ${String(formatVersion)}`);
  }
  if (typeof models !== 'object' || models === null) throw new Error('manifest has no models');
  for (const [key, entry] of Object.entries(models as Record<string, unknown>)) {
    const { version, files } = (entry ?? {}) as Record<string, unknown>;
    if (typeof version !== 'string' || !/^[\w.-]+$/.test(version)) throw new Error(`model ${key} has a bad version`);
    if (typeof files !== 'object' || files === null) throw new Error(`model ${key} has no files`);
    for (const [name, file] of Object.entries(files as Record<string, unknown>)) {
      if (!FILE_NAME.test(name) || !isManifestFile(file)) throw new Error(`model ${key} has a bad ${name} file`);
    }
    if (formatVersion === 1) {
      for (const name of V2_MODEL_FILES) {
        if (!(name in (files as Record<string, unknown>))) throw new Error(`model ${key} lacks ${name}`);
      }
    }
  }
  return parsed as ClimbGeneratorManifest;
}

const textDecoder = new TextDecoder();

export function decodeJson(bytes: Uint8Array): unknown {
  return JSON.parse(textDecoder.decode(bytes));
}

/** Where a manifest's models live on the phone, so v2 and v3 copies never prune each other. */
function storeKeyFor(manifestPath: string, modelKey: string): string {
  const folder = manifestPath.includes('/') ? manifestPath.slice(0, manifestPath.lastIndexOf('/')) : '';
  return folder ? `${folder}/${modelKey}` : modelKey;
}

async function readCachedFiles(
  store: RemoteModelStore,
  storeKey: string,
  entry: RemoteModelEntry,
  names: readonly string[],
): Promise<Record<string, Uint8Array> | null> {
  const result: Record<string, Uint8Array> = {};
  for (const name of names) {
    const bytes = await store.readFile(storeKey, entry.version, name);
    if (!bytes || sha256Hex(bytes) !== entry.files[name].sha256) return null;
    result[name] = bytes;
  }
  return result;
}

/**
 * The newest manifest we can get: the network's if it answers (and then saved),
 * else the one saved last time, else null.
 */
async function currentManifest(deps: RemoteModelDeps, manifestPath: string): Promise<ClimbGeneratorManifest | null> {
  try {
    const bytes = await deps.fetchBytes(`${deps.baseUrl}/${manifestPath}`);
    const json = textDecoder.decode(bytes);
    const manifest = parseManifest(json);
    await deps.store.writeManifest(manifestPath, json);
    return manifest;
  } catch {
    const saved = await deps.store.readManifest(manifestPath);
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
  const { manifestPath } = deps;
  const names = deps.requiredFiles ?? V2_MODEL_FILES;
  const manifest = await currentManifest(deps, manifestPath);
  if (!manifest) throw new RemoteModelError('Could not reach the climb generator models.', 'offline');
  const entry = manifest.models[modelKey];
  if (!entry || names.some((name) => !(name in entry.files))) {
    throw new RemoteModelError(`No climb generator model for ${modelKey}.`, 'unavailable');
  }
  const storeKey = storeKeyFor(manifestPath, modelKey);

  const cached = await readCachedFiles(deps.store, storeKey, entry, names);
  if (cached) return { version: entry.version, files: cached };

  const downloaded: Record<string, Uint8Array> = {};
  for (const name of names) {
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
  for (const name of names) {
    await deps.store.writeFile(storeKey, entry.version, name, downloaded[name]);
  }
  await deps.store.pruneVersions(storeKey, entry.version);
  return { version: entry.version, files: downloaded };
}
