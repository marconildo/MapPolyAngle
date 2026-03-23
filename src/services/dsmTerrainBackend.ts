import type { DsmSourceDescriptor, TerrainSourceSelection } from '@/terrain/types';

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

interface DsmTerrainDatasetListResponse {
  datasets: DsmTerrainBackendDataset[];
}

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

export async function listDsmDatasetsFromTerrainBackend(): Promise<DsmTerrainBackendDataset[]> {
  if (!isDsmTerrainBackendEnabled()) return [];
  const response = await fetch(`${backendBase()}/v1/dsm/datasets`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `DSM dataset list failed with status ${response.status}`);
  }
  const payload = await response.json() as DsmTerrainDatasetListResponse;
  return Array.isArray(payload.datasets) ? payload.datasets : [];
}

export async function getDsmDatasetFromTerrainBackend(datasetId: string): Promise<DsmTerrainBackendDataset> {
  const response = await fetch(`${backendBase()}/v1/dsm/datasets/${encodeURIComponent(datasetId)}`);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `DSM dataset lookup failed with status ${response.status}`);
  }
  return await response.json() as DsmTerrainBackendDataset;
}

export async function uploadDsmToTerrainBackend(file: File): Promise<DsmTerrainBackendDataset> {
  const formData = new FormData();
  formData.set('file', file, file.name);
  const response = await fetch(`${backendBase()}/v1/dsm/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `DSM upload failed with status ${response.status}`);
  }
  return await response.json() as DsmTerrainBackendDataset;
}
