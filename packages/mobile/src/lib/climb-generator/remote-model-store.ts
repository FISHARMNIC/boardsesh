/**
 * Where downloaded climb generator models live on the phone:
 * `<documents>/climb-generator/<model>/<version>/<file>` plus the last
 * `manifest.json`. Documents, not caches, so the OS never reclaims a model and
 * the generator keeps working offline once a board has been opened.
 */

import { Directory, File, Paths } from 'expo-file-system';
import type { RemoteModelFileName, RemoteModelStore } from './remote-models';

const ROOT_NAME = 'climb-generator';
const FILE_NAMES: Record<RemoteModelFileName, string> = {
  meta: 'model-meta.json',
  weights: 'weights.bin',
  grader: 'grader.json',
};

function root(): Directory {
  return new Directory(Paths.document, ROOT_NAME);
}

function versionDirectory(modelKey: string, version: string): Directory {
  return new Directory(root(), modelKey, version);
}

function deleteQuietly(entry: File | Directory) {
  try {
    if (entry.exists) entry.delete();
  } catch {
    // A file we could not delete is retried on the next prune.
  }
}

export const expoRemoteModelStore: RemoteModelStore = {
  async readManifest() {
    const file = new File(root(), 'manifest.json');
    return file.exists ? file.text() : null;
  },

  async writeManifest(json) {
    root().create({ intermediates: true, idempotent: true });
    const partial = new File(root(), 'manifest.json.part');
    await partial.write(json);
    await partial.move(new File(root(), 'manifest.json'), { overwrite: true });
  },

  async readFile(modelKey, version, name) {
    const file = new File(versionDirectory(modelKey, version), FILE_NAMES[name]);
    return file.exists ? file.bytes() : null;
  },

  async writeFile(modelKey, version, name, bytes) {
    const directory = versionDirectory(modelKey, version);
    directory.create({ intermediates: true, idempotent: true });
    // Write beside, then rename into place, so a reader never sees half a file.
    const partial = new File(directory, `${FILE_NAMES[name]}.part`);
    await partial.write(bytes);
    await partial.move(new File(directory, FILE_NAMES[name]), { overwrite: true });
  },

  async pruneVersions(modelKey, keepVersion) {
    const modelDirectory = new Directory(root(), modelKey);
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
