import { createHash } from 'node:crypto';
import { describe, expect, it, vi } from 'vitest';
import {
  ensureRemoteModel,
  parseManifest,
  RemoteModelError,
  type RemoteModelFileName,
  type RemoteModelStore,
} from '../remote-models';

const encoder = new TextEncoder();
const BASE = 'https://models.example/climbgen';

function sha(bytes: Uint8Array) {
  return createHash('sha256').update(bytes).digest('hex');
}

function modelFiles(version: string) {
  return {
    meta: encoder.encode(JSON.stringify({ tags: ['crimp'], version })),
    weights: new Uint8Array([1, 2, 3, version.length]),
    grader: encoder.encode(JSON.stringify({ temperature: 1.15 })),
  };
}

function serve(version: string, override: Partial<Record<RemoteModelFileName, Uint8Array>> = {}) {
  const files = modelFiles(version);
  const entry = (name: RemoteModelFileName, path: string) => ({ path, sha256: sha(files[name]), bytes: 0 });
  const manifest = {
    formatVersion: 1,
    models: {
      'tension-10': {
        version,
        files: {
          meta: entry('meta', `tension-10/${version}/model-meta.json`),
          weights: entry('weights', `tension-10/${version}/weights.bin`),
          grader: entry('grader', `tension-10/${version}/grader.json`),
        },
      },
    },
  };
  const byUrl: Record<string, Uint8Array> = {
    [`${BASE}/manifest.json`]: encoder.encode(JSON.stringify(manifest)),
    [`${BASE}/tension-10/${version}/model-meta.json`]: override.meta ?? files.meta,
    [`${BASE}/tension-10/${version}/weights.bin`]: override.weights ?? files.weights,
    [`${BASE}/tension-10/${version}/grader.json`]: override.grader ?? files.grader,
  };
  return vi.fn(async (url: string) => {
    const body = byUrl[url];
    if (!body) throw new Error(`404 ${url}`);
    return body;
  });
}

function offline() {
  return vi.fn(async (): Promise<Uint8Array> => {
    throw new Error('offline');
  });
}

function memoryStore(): RemoteModelStore & { files: Map<string, Uint8Array> } {
  let manifest: string | null = null;
  const files = new Map<string, Uint8Array>();
  return {
    files,
    readManifest: async () => manifest,
    writeManifest: async (json) => {
      manifest = json;
    },
    readFile: async (key, version, name) => files.get(`${key}/${version}/${name}`) ?? null,
    writeFile: async (key, version, name, bytes) => {
      files.set(`${key}/${version}/${name}`, bytes);
    },
    pruneVersions: async (key, keep) => {
      for (const path of [...files.keys()]) {
        if (path.startsWith(`${key}/`) && !path.startsWith(`${key}/${keep}/`)) files.delete(path);
      }
    },
  };
}

describe('ensureRemoteModel', () => {
  it('downloads, checks and saves a model the first time, then reads it from the phone', async () => {
    const store = memoryStore();
    const fetchBytes = serve('v1');
    const first = await ensureRemoteModel('tension-10', { baseUrl: BASE, fetchBytes, store });
    expect(first.meta).toEqual({ tags: ['crimp'], version: 'v1' });
    expect([...first.weights]).toEqual([1, 2, 3, 2]);
    expect(store.files.size).toBe(3);

    fetchBytes.mockClear();
    await ensureRemoteModel('tension-10', { baseUrl: BASE, fetchBytes, store });
    // Only the manifest is fetched; the files come from the phone.
    expect(fetchBytes.mock.calls.map(([url]) => url)).toEqual([`${BASE}/manifest.json`]);
  });

  it('works offline from the saved manifest and files', async () => {
    const store = memoryStore();
    await ensureRemoteModel('tension-10', { baseUrl: BASE, fetchBytes: serve('v1'), store });
    const files = await ensureRemoteModel('tension-10', { baseUrl: BASE, fetchBytes: offline(), store });
    expect(files.version).toBe('v1');
  });

  it('says offline when there is no network and nothing saved', async () => {
    await expect(
      ensureRemoteModel('tension-10', { baseUrl: BASE, fetchBytes: offline(), store: memoryStore() }),
    ).rejects.toMatchObject({ reason: 'offline' });
  });

  it('refuses a download that does not match its checksum, and saves nothing', async () => {
    const store = memoryStore();
    const fetchBytes = serve('v1', { weights: new Uint8Array([9, 9, 9]) });
    const loading = ensureRemoteModel('tension-10', { baseUrl: BASE, fetchBytes, store });
    await expect(loading).rejects.toBeInstanceOf(RemoteModelError);
    await expect(loading).rejects.toMatchObject({ reason: 'corrupt' });
    expect(store.files.size).toBe(0);
  });

  it('says unavailable for a board the manifest does not list', async () => {
    await expect(
      ensureRemoteModel('moonboard-2', { baseUrl: BASE, fetchBytes: serve('v1'), store: memoryStore() }),
    ).rejects.toMatchObject({ reason: 'unavailable' });
  });

  it('replaces an old version with a new one and drops the old files', async () => {
    const store = memoryStore();
    await ensureRemoteModel('tension-10', { baseUrl: BASE, fetchBytes: serve('v1'), store });
    const files = await ensureRemoteModel('tension-10', { baseUrl: BASE, fetchBytes: serve('v2'), store });
    expect(files.version).toBe('v2');
    expect([...store.files.keys()].every((path) => path.startsWith('tension-10/v2/'))).toBe(true);
  });
});

describe('parseManifest', () => {
  it('rejects a path that climbs out of the models folder', () => {
    const bad = {
      formatVersion: 1,
      models: {
        x: {
          version: 'v1',
          files: {
            meta: { path: '../escape', sha256: 'a'.repeat(64), bytes: 1 },
            weights: { path: 'w', sha256: 'a'.repeat(64), bytes: 1 },
            grader: { path: 'g', sha256: 'a'.repeat(64), bytes: 1 },
          },
        },
      },
    };
    expect(() => parseManifest(JSON.stringify(bad))).toThrow();
  });

  it('rejects an unknown format version', () => {
    expect(() => parseManifest(JSON.stringify({ formatVersion: 2, models: {} }))).toThrow();
  });
});
