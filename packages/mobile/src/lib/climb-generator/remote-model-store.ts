/**
 * Where downloaded climb generator models live on the phone:
 * `<documents>/climb-generator/<manifest folder>/<model>/<version>/<file>` plus
 * each manifest last seen. Documents, not caches, so the OS never reclaims a model
 * and the generator keeps working offline once a board has been opened.
 */

import { Directory, File, Paths } from 'expo-file-system';
import type { RemoteModelStore } from './remote-models';

const ROOT_NAME = 'climb-generator';
// v2 file names kept as they were, so models already on a phone stay valid.
const KNOWN_FILE_NAMES: Record<string, string> = {
  meta: 'model-meta.json',
  weights: 'weights.bin',
  grader: 'grader.json',
};

function fileName(name: string): string {
  return KNOWN_FILE_NAMES[name] ?? `${name}.bin`;
}

function root(): Directory {
  return new Directory(Paths.document, ROOT_NAME);
}

function nested(...parts: string[]): Directory {
  return new Directory(root(), ...parts.flatMap((part) => part.split('/')));
}

function manifestFile(manifestPath: string): File {
  const parts = manifestPath.split('/');
  const name = parts.pop() as string;
  return new File(parts.length ? nested(...parts) : root(), name);
}

function deleteQuietly(entry: File | Directory) {
  try {
    if (entry.exists) entry.delete();
  } catch {
    // A file we could not delete is retried on the next prune.
  }
}

async function writeAtomically(directory: Directory, name: string, content: string | Uint8Array) {
  directory.create({ intermediates: true, idempotent: true });
  // Write beside, then rename into place, so a reader never sees half a file.
  const partial = new File(directory, `${name}.part`);
  await partial.write(content);
  await partial.move(new File(directory, name), { overwrite: true });
}

const MANIFEST_FOLDERS = new Set(['v2', 'v3']);

/**
 * Before the v2/ and v3/ folders, v2 models sat at the root of the store. Delete
 * anything there that is not a current manifest folder, so those copies don't
 * linger. Safe to call on every load.
 */
export function removeLegacyRootModels() {
  const directory = root();
  if (!directory.exists) return;
  for (const entry of directory.list()) {
    if (!(entry instanceof Directory && MANIFEST_FOLDERS.has(entry.name))) deleteQuietly(entry);
  }
}

export const expoRemoteModelStore: RemoteModelStore = {
  async readManifest(manifestPath) {
    const file = manifestFile(manifestPath);
    return file.exists ? file.text() : null;
  },

  async writeManifest(manifestPath, json) {
    const parts = manifestPath.split('/');
    const name = parts.pop() as string;
    await writeAtomically(parts.length ? nested(...parts) : root(), name, json);
  },

  async readFile(storeKey, version, name) {
    const file = new File(nested(storeKey, version), fileName(name));
    return file.exists ? file.bytes() : null;
  },

  async writeFile(storeKey, version, name, bytes) {
    await writeAtomically(nested(storeKey, version), fileName(name), bytes);
  },

  async pruneVersions(storeKey, keepVersion) {
    const modelDirectory = nested(storeKey);
    if (!modelDirectory.exists) return;
    for (const entry of modelDirectory.list()) {
      if (entry instanceof Directory && entry.name !== keepVersion) deleteQuietly(entry);
    }
  },
};

/** The whole body of `url`, or a throw on a network error or a non-2xx status. */
export async function fetchBytes(url: string): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} for ${url}`);
  return new Uint8Array(await response.arrayBuffer());
}
