import { createHash } from 'node:crypto';
import { describe, expect, it, vi } from 'vitest';
import {
  decodeJson,
  ensureRemoteModel,
  parseManifest,
  RemoteModelError,
  type RemoteModelStore,
} from '../remote-models';

type FileName = 'meta' | 'weights' | 'grader';

const encoder = new TextEncoder();
const BASE = 'https://models.example/climbgen';
const V2 = 'v2/manifest.json';

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

function serve(version: string, override: Partial<Record<FileName, Uint8Array>> = {}) {
  const files = modelFiles(version);
  const entry = (name: FileName, path: string) => ({ path, sha256: sha(files[name]), bytes: 0 });
  const manifest = {
    formatVersion: 1,
    models: {
      'tension-10': {
        version,
        files: {
          meta: entry('meta', `v2/tension-10/${version}/model-meta.json`),
          weights: entry('weights', `v2/tension-10/${version}/weights.bin`),
          grader: entry('grader', `v2/tension-10/${version}/grader.json`),
        },
      },
    },
  };
  const byUrl: Record<string, Uint8Array> = {
    [`${BASE}/v2/manifest.json`]: encoder.encode(JSON.stringify(manifest)),
    [`${BASE}/v2/tension-10/${version}/model-meta.json`]: override.meta ?? files.meta,
    [`${BASE}/v2/tension-10/${version}/weights.bin`]: override.weights ?? files.weights,
    [`${BASE}/v2/tension-10/${version}/grader.json`]: override.grader ?? files.grader,
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
  const manifests = new Map<string, string>();
  const files = new Map<string, Uint8Array>();
  return {
    files,
    readManifest: async (path) => manifests.get(path) ?? null,
    writeManifest: async (path, json) => {
      manifests.set(path, json);
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
    const first = await ensureRemoteModel('tension-10', { baseUrl: BASE, manifestPath: V2, fetchBytes, store });
    expect(decodeJson(first.files.meta)).toEqual({ tags: ['crimp'], version: 'v1' });
    expect([...first.files.weights]).toEqual([1, 2, 3, 2]);
    expect(store.files.size).toBe(3);

    fetchBytes.mockClear();
    await ensureRemoteModel('tension-10', { baseUrl: BASE, manifestPath: V2, fetchBytes, store });
    // Only the manifest is fetched; the files come from the phone.
    expect(fetchBytes.mock.calls.map(([url]) => url)).toEqual([`${BASE}/v2/manifest.json`]);
  });

  it('works offline from the saved manifest and files', async () => {
    const store = memoryStore();
    await ensureRemoteModel('tension-10', { baseUrl: BASE, manifestPath: V2, fetchBytes: serve('v1'), store });
    const files = await ensureRemoteModel('tension-10', {
      baseUrl: BASE,
      manifestPath: V2,
      fetchBytes: offline(),
      store,
    });
    expect(files.version).toBe('v1');
  });

  it('says offline when there is no network and nothing saved', async () => {
    await expect(
      ensureRemoteModel('tension-10', { baseUrl: BASE, manifestPath: V2, fetchBytes: offline(), store: memoryStore() }),
    ).rejects.toMatchObject({ reason: 'offline' });
  });

  it('refuses a download that does not match its checksum, and saves nothing', async () => {
    const store = memoryStore();
    const fetchBytes = serve('v1', { weights: new Uint8Array([9, 9, 9]) });
    const loading = ensureRemoteModel('tension-10', { baseUrl: BASE, manifestPath: V2, fetchBytes, store });
    await expect(loading).rejects.toBeInstanceOf(RemoteModelError);
    await expect(loading).rejects.toMatchObject({ reason: 'corrupt' });
    expect(store.files.size).toBe(0);
  });

  it('says unavailable for a board the manifest does not list', async () => {
    await expect(
      ensureRemoteModel('moonboard-2', {
        baseUrl: BASE,
        manifestPath: V2,
        fetchBytes: serve('v1'),
        store: memoryStore(),
      }),
    ).rejects.toMatchObject({ reason: 'unavailable' });
  });

  it('replaces an old version with a new one and drops the old files', async () => {
    const store = memoryStore();
    await ensureRemoteModel('tension-10', { baseUrl: BASE, manifestPath: V2, fetchBytes: serve('v1'), store });
    const files = await ensureRemoteModel('tension-10', {
      baseUrl: BASE,
      manifestPath: V2,
      fetchBytes: serve('v2'),
      store,
    });
    expect(files.version).toBe('v2');
    expect([...store.files.keys()].every((path) => path.startsWith('v2/tension-10/v2/'))).toBe(true);
  });
});

describe('ensureRemoteModel with a v3 manifest', () => {
  it('reads v3/manifest.json, fetches the extra files it needs, and keeps v2 and v3 copies apart', async () => {
    const extra = encoder.encode('critic-bytes');
    const v3Manifest = {
      formatVersion: 2,
      models: {
        'tension-10': {
          version: 'v3a',
          files: {
            meta: { path: 'v3/tension-10/v3a/model-meta.json', sha256: sha(modelFiles('v1').meta), bytes: 0 },
            weights: { path: 'v3/tension-10/v3a/weights.bin', sha256: sha(modelFiles('v1').weights), bytes: 0 },
            grader: { path: 'v3/tension-10/v3a/grader.json', sha256: sha(modelFiles('v1').grader), bytes: 0 },
            critic: { path: 'v3/tension-10/v3a/critic.json', sha256: sha(extra), bytes: 0 },
          },
        },
      },
    };
    const v2Fetch = serve('v1');
    const files = modelFiles('v1');
    const byUrl: Record<string, Uint8Array> = {
      [`${BASE}/v3/manifest.json`]: encoder.encode(JSON.stringify(v3Manifest)),
      [`${BASE}/v3/tension-10/v3a/model-meta.json`]: files.meta,
      [`${BASE}/v3/tension-10/v3a/weights.bin`]: files.weights,
      [`${BASE}/v3/tension-10/v3a/grader.json`]: files.grader,
      [`${BASE}/v3/tension-10/v3a/critic.json`]: extra,
    };
    const fetchBytes = vi.fn(async (url: string) => byUrl[url] ?? v2Fetch(url));
    const store = memoryStore();
    await ensureRemoteModel('tension-10', { baseUrl: BASE, manifestPath: V2, fetchBytes, store });
    const v3 = await ensureRemoteModel('tension-10', {
      baseUrl: BASE,
      fetchBytes,
      store,
      manifestPath: 'v3/manifest.json',
      requiredFiles: ['meta', 'weights', 'grader', 'critic'],
    });
    expect(new TextDecoder().decode(v3.files.critic)).toBe('critic-bytes');
    // Both versions stay on the phone: switching between them never re-downloads.
    expect([...store.files.keys()].filter((path) => path.startsWith('v2/tension-10/'))).toHaveLength(3);
    expect([...store.files.keys()].filter((path) => path.startsWith('v3/tension-10/'))).toHaveLength(4);
  });

  it('says unavailable when the model lacks a file the caller needs', async () => {
    await expect(
      ensureRemoteModel('tension-10', {
        baseUrl: BASE,
        manifestPath: V2,
        fetchBytes: serve('v1'),
        store: memoryStore(),
        requiredFiles: ['meta', 'weights', 'grader', 'critic'],
      }),
    ).rejects.toMatchObject({ reason: 'unavailable' });
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
    expect(() => parseManifest(JSON.stringify({ formatVersion: 3, models: {} }))).toThrow();
  });
});
