import type { DsmSourceDescriptor, TerrainSourceSelection } from '@/terrain/types';
import { sha256 } from '@noble/hashes/sha256';
import { bytesToHex } from '@noble/hashes/utils';

declare global {
  // Test-only override used by tsx/node tests outside the Vite runtime.
  var __TERRAIN_BACKEND_URL_FOR_TESTS__: string | undefined;
}

export interface DsmTerrainBackendDataset {
  datasetId: string | null;
  descriptor: DsmSourceDescriptor | null;
  processingStatus: 'ready' | null;
  reusedExisting: boolean;
  terrainTileUrlTemplate: string | null;
}

interface DsmPrepareUploadRequest {
  sha256: string;
  fileSizeBytes: number;
  originalName: string;
  contentType: string | null;
}

interface DsmUploadTarget {
  url: string;
  method: 'PUT';
  headers: Record<string, string>;
  expiresAtIso: string;
}

interface DsmPrepareUploadExistingResponse {
  status: 'existing';
  dataset: DsmTerrainBackendDataset;
}

interface DsmPrepareUploadRequiredResponse {
  status: 'upload-required';
  uploadId: string;
  uploadTarget: DsmUploadTarget;
}

type DsmPrepareUploadResponse = DsmPrepareUploadExistingResponse | DsmPrepareUploadRequiredResponse;

function configuredBackendBaseUrl(): string | undefined {
  const fromImportMeta = (import.meta as ImportMeta & { env?: Record<string, string | undefined> }).env?.VITE_TERRAIN_PARTITION_BACKEND_URL;
  const fromGlobal = globalThis.__TERRAIN_BACKEND_URL_FOR_TESTS__;
  const fromProcess = typeof process !== 'undefined' ? process.env.VITE_TERRAIN_PARTITION_BACKEND_URL : undefined;
  return [fromImportMeta, fromGlobal, fromProcess].find(
    (value): value is string => typeof value === 'string' && value.trim().length > 0
  );
}

function backendBase(): string {
  const backendBaseUrl = configuredBackendBaseUrl();
  if (typeof backendBaseUrl !== 'string' || backendBaseUrl.trim().length === 0) {
    throw new Error('Terrain backend is not configured.');
  }
  return backendBaseUrl.replace(/\/$/, '');
}

async function readErrorText(response: Response): Promise<string> {
  try {
    const text = await response.text();
    return text;
  } catch {
    return '';
  }
}

async function computeFileSha256(file: File): Promise<string> {
  const digest = sha256.create();
  const chunkSize = 4 * 1024 * 1024;
  for (let offset = 0; offset < file.size; offset += chunkSize) {
    const chunk = file.slice(offset, Math.min(file.size, offset + chunkSize));
    const bytes = new Uint8Array(await chunk.arrayBuffer());
    digest.update(bytes);
  }
  return bytesToHex(digest.digest());
}

export function isDsmTerrainBackendEnabled(): boolean {
  const backendBaseUrl = configuredBackendBaseUrl();
  return typeof backendBaseUrl === 'string' && backendBaseUrl.trim().length > 0;
}

export function getTerrainTileUrlTemplateForSource(source: TerrainSourceSelection): string | null {
  if (!isDsmTerrainBackendEnabled()) return null;
  const params = new URLSearchParams({ mode: source.mode });
  if (source.mode === 'blended' && source.datasetId) params.set('datasetId', source.datasetId);
  return `${backendBase()}/v1/terrain-rgb/{z}/{x}/{y}.png?${params.toString()}`;
}

export function getTerrainTileUrlForSource(source: TerrainSourceSelection, z: number, x: number, y: number): string | null {
  const template = getTerrainTileUrlTemplateForSource(source);
  if (!template) return null;
  return template
    .replace('{z}', String(z))
    .replace('{x}', String(x))
    .replace('{y}', String(y));
}

export async function getDsmDatasetFromTerrainBackend(datasetId: string): Promise<DsmTerrainBackendDataset> {
  const found = await findDsmDatasetFromTerrainBackend(datasetId);
  if (found) return found;
  throw new Error(`DSM dataset ${datasetId} was not found.`);
}

export async function findDsmDatasetFromTerrainBackend(datasetId: string): Promise<DsmTerrainBackendDataset | null> {
  const response = await fetch(`${backendBase()}/v1/dsm/datasets/${encodeURIComponent(datasetId)}`);
  if (response.status === 404) {
    return null;
  }
  if (!response.ok) {
    const text = await readErrorText(response);
    throw new Error(text || `DSM dataset lookup failed with status ${response.status}`);
  }
  return await response.json() as DsmTerrainBackendDataset;
}

export async function uploadDsmToTerrainBackend(file: File): Promise<DsmTerrainBackendDataset> {
  const fileSha256 = await computeFileSha256(file);
  const prepareRequest: DsmPrepareUploadRequest = {
    sha256: fileSha256,
    fileSizeBytes: file.size,
    originalName: file.name,
    contentType: file.type?.trim().length ? file.type : null,
  };
  const prepareResponse = await fetch(`${backendBase()}/v1/dsm/prepare-upload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(prepareRequest),
  });
  if (!prepareResponse.ok) {
    const text = await readErrorText(prepareResponse);
    throw new Error(text || `DSM upload preparation failed with status ${prepareResponse.status}`);
  }
  const prepared = await prepareResponse.json() as DsmPrepareUploadResponse;
  if (prepared.status === 'existing') {
    return prepared.dataset;
  }

  const uploadResponse = await fetch(prepared.uploadTarget.url, {
    method: prepared.uploadTarget.method,
    headers: prepared.uploadTarget.headers,
    body: file,
  });
  if (!uploadResponse.ok) {
    const text = await readErrorText(uploadResponse);
    throw new Error(text || `DSM staged upload failed with status ${uploadResponse.status}`);
  }

  const finalizeResponse = await fetch(`${backendBase()}/v1/dsm/finalize-upload`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ uploadId: prepared.uploadId }),
  });
  if (!finalizeResponse.ok) {
    const text = await readErrorText(finalizeResponse);
    throw new Error(text || `DSM upload finalization failed with status ${finalizeResponse.status}`);
  }
  return await finalizeResponse.json() as DsmTerrainBackendDataset;
}
