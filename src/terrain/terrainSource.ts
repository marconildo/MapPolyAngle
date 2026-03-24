import { clearActiveDsm, loadDsmFromFile, validateDsmGeoTiffFile } from './dsmSource';
import type { DsmSourceDescriptor, TerrainSourceMode, TerrainSourceSelection, TerrainSourceState } from './types';
import {
  findDsmDatasetFromTerrainBackend,
  getDsmDatasetFromTerrainBackend,
  getTerrainTileUrlForSource,
  getTerrainTileUrlTemplateForSource,
  isDsmTerrainBackendEnabled,
  uploadDsmToTerrainBackend,
} from '@/services/dsmTerrainBackend';

const listeners = new Set<() => void>();
const STORAGE_KEY = 'terrain-source-selection-v1';

interface PersistedTerrainSelection {
  mode: TerrainSourceMode;
  selectedDatasetId: string | null;
  rememberedDescriptor: DsmSourceDescriptor | null;
}

let terrainSourceState: TerrainSourceState = initialTerrainSourceState();

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function readPersistedDescriptor(raw: unknown): DsmSourceDescriptor | null {
  if (!raw || typeof raw !== 'object') return null;
  const candidate = raw as Partial<DsmSourceDescriptor>;
  if (typeof candidate.id !== 'string' || candidate.id.trim().length === 0) return null;
  if (typeof candidate.name !== 'string' || candidate.name.trim().length === 0) return null;
  if (!isFiniteNumber(candidate.fileSizeBytes) || !isFiniteNumber(candidate.width) || !isFiniteNumber(candidate.height)) return null;
  if (typeof candidate.sourceCrsLabel !== 'string') return null;
  if (!candidate.footprintLngLat || typeof candidate.footprintLngLat !== 'object') return null;
  if (
    !isFiniteNumber(candidate.footprintLngLat.minLng) ||
    !isFiniteNumber(candidate.footprintLngLat.minLat) ||
    !isFiniteNumber(candidate.footprintLngLat.maxLng) ||
    !isFiniteNumber(candidate.footprintLngLat.maxLat)
  ) {
    return null;
  }
  if (!Array.isArray(candidate.footprintRingLngLat)) return null;
  return candidate as DsmSourceDescriptor;
}

function readPersistedTerrainSelection(): PersistedTerrainSelection | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<PersistedTerrainSelection>;
    return {
      mode: parsed.mode === 'blended' ? 'blended' : 'mapbox',
      selectedDatasetId:
        typeof parsed.selectedDatasetId === 'string' && parsed.selectedDatasetId.trim().length > 0
          ? parsed.selectedDatasetId
          : null,
      rememberedDescriptor: readPersistedDescriptor((parsed as { rememberedDescriptor?: unknown }).rememberedDescriptor),
    };
  } catch {
    return null;
  }
}

function persistTerrainSelection() {
  if (typeof window === 'undefined') return;
  try {
    const rememberedDescriptor = terrainSourceState.descriptor ?? terrainSourceState.rememberedDescriptor;
    const payload: PersistedTerrainSelection = {
      mode: terrainSourceState.source.mode,
      selectedDatasetId: rememberedDescriptor?.id ?? null,
      rememberedDescriptor,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {
    // Ignore localStorage failures.
  }
}

function emit(options?: { persist?: boolean }) {
  terrainSourceState = {
    ...terrainSourceState,
    backendEnabled: isDsmTerrainBackendEnabled(),
  };
  if (options?.persist !== false) {
    persistTerrainSelection();
  }
  for (const listener of listeners) listener();
}

function initialTerrainSourceState(): TerrainSourceState {
  return {
    source: { mode: 'mapbox', datasetId: null },
    descriptor: null,
    rememberedDescriptor: null,
    isLoading: false,
    error: null,
    backendEnabled: isDsmTerrainBackendEnabled(),
  };
}

export function subscribeTerrainSource(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getTerrainSourceState(): TerrainSourceState {
  return terrainSourceState;
}

export function getCurrentTerrainSource(): TerrainSourceSelection {
  return terrainSourceState.source;
}

export function getCurrentTerrainDescriptor(): DsmSourceDescriptor | null {
  return terrainSourceState.descriptor;
}

export function getTerrainDemUrlTemplateForCurrentSource(): string | null {
  return getTerrainTileUrlTemplateForSource(terrainSourceState.source);
}

export function getTerrainTileUrlForCurrentSource(z: number, x: number, y: number): string | null {
  return getTerrainTileUrlForSource(terrainSourceState.source, z, x, y);
}

export function setTerrainSourceMode(mode: TerrainSourceMode) {
  const nextMode: TerrainSourceMode = mode === 'blended' && terrainSourceState.descriptor ? 'blended' : 'mapbox';
  terrainSourceState = {
    ...terrainSourceState,
    source: {
      mode: nextMode,
      datasetId: nextMode === 'blended' ? (terrainSourceState.descriptor?.id ?? null) : null,
    },
    error: nextMode === 'blended' || terrainSourceState.error == null ? terrainSourceState.error : null,
  };
  emit();
}

export function clearTerrainSourceSelection() {
  clearActiveDsm();
  terrainSourceState = {
    ...terrainSourceState,
    source: { mode: 'mapbox', datasetId: null },
    descriptor: null,
    rememberedDescriptor: null,
    isLoading: false,
    error: null,
  };
  emit();
}

export function __resetTerrainSourceForTests(options?: { clearStorage?: boolean }) {
  clearActiveDsm();
  listeners.clear();
  terrainSourceState = initialTerrainSourceState();
  if (options?.clearStorage !== false && typeof window !== 'undefined') {
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      // Ignore storage cleanup failures in tests.
    }
  }
}

export async function initializeTerrainSourceState(): Promise<void> {
  const persisted = readPersistedTerrainSelection();
  terrainSourceState = {
    ...terrainSourceState,
    source: { mode: 'mapbox', datasetId: null },
    descriptor: null,
    rememberedDescriptor: persisted?.rememberedDescriptor ?? null,
    isLoading: false,
    error: null,
  };
  emit({ persist: false });
}

export async function selectTerrainSourceDataset(datasetId: string, mode: TerrainSourceMode = 'blended'): Promise<DsmSourceDescriptor> {
  const normalizedDatasetId = datasetId.trim();
  if (!normalizedDatasetId) {
    throw new Error('A DSM dataset id is required.');
  }
  if (!isDsmTerrainBackendEnabled()) {
    throw new Error('Selecting an existing DSM dataset requires a configured terrain backend.');
  }

  terrainSourceState = {
    ...terrainSourceState,
    isLoading: true,
    error: null,
  };
  emit();

  try {
    const fetched = (await getDsmDatasetFromTerrainBackend(normalizedDatasetId)).descriptor;
    if (!fetched) {
      throw new Error(`DSM dataset ${normalizedDatasetId} is not available.`);
    }
    terrainSourceState = {
      ...terrainSourceState,
      descriptor: fetched,
      rememberedDescriptor: fetched,
      source: {
        mode: mode === 'blended' ? 'blended' : 'mapbox',
        datasetId: mode === 'blended' ? fetched.id : null,
      },
      isLoading: false,
      error: null,
    };
    emit();
    return fetched;
  } catch (error) {
    terrainSourceState = {
      ...terrainSourceState,
      isLoading: false,
      error: error instanceof Error ? error.message : String(error),
    };
    emit();
    throw error;
  }
}

export async function activateRememberedTerrainSource(): Promise<DsmSourceDescriptor> {
  const rememberedDescriptor = terrainSourceState.rememberedDescriptor;
  if (!rememberedDescriptor) {
    throw new Error('There is no saved DSM to load.');
  }
  if (!isDsmTerrainBackendEnabled()) {
    throw new Error('Loading a saved DSM requires a configured terrain backend.');
  }

  terrainSourceState = {
    ...terrainSourceState,
    isLoading: true,
    error: null,
  };
  emit();

  try {
    const fetched = await findDsmDatasetFromTerrainBackend(rememberedDescriptor.id);
    if (!fetched?.descriptor) {
      terrainSourceState = {
        ...terrainSourceState,
        source: { mode: 'mapbox', datasetId: null },
        descriptor: null,
        rememberedDescriptor: null,
        isLoading: false,
        error: `Saved DSM ${rememberedDescriptor.name} is no longer available.`,
      };
      emit();
      throw new Error(`Saved DSM ${rememberedDescriptor.name} is no longer available.`);
    }
    terrainSourceState = {
      ...terrainSourceState,
      source: { mode: 'blended', datasetId: fetched.descriptor.id },
      descriptor: fetched.descriptor,
      rememberedDescriptor: fetched.descriptor,
      isLoading: false,
      error: null,
    };
    emit();
    return fetched.descriptor;
  } catch (error) {
    if (terrainSourceState.isLoading) {
      terrainSourceState = {
        ...terrainSourceState,
        isLoading: false,
        error: error instanceof Error ? error.message : String(error),
      };
      emit();
    }
    throw error;
  }
}

export async function loadTerrainSourceFromFile(
  file: File,
  options?: { onProgressPhase?: (phase: 'validating' | 'uploading') => void },
): Promise<DsmSourceDescriptor> {
  terrainSourceState = {
    ...terrainSourceState,
    isLoading: true,
    error: null,
  };
  emit();

  try {
    if (isDsmTerrainBackendEnabled()) {
      try {
        options?.onProgressPhase?.('validating');
        await validateDsmGeoTiffFile(file);
        options?.onProgressPhase?.('uploading');
        const uploaded = await uploadDsmToTerrainBackend(file);
        clearActiveDsm();
        if (!uploaded.descriptor || !uploaded.datasetId) {
          throw new Error('Terrain backend did not return a usable DSM descriptor.');
        }
        terrainSourceState = {
          ...terrainSourceState,
          source: { mode: 'blended', datasetId: uploaded.datasetId },
          descriptor: uploaded.descriptor,
          rememberedDescriptor: uploaded.descriptor,
          isLoading: false,
          error: null,
        };
        emit();
        return uploaded.descriptor;
      } catch (error) {
        clearActiveDsm();
        throw error;
      }
    }

    const localDescriptor = await loadDsmFromFile(file);
    terrainSourceState = {
      ...terrainSourceState,
      source: { mode: 'blended', datasetId: localDescriptor.id },
      descriptor: localDescriptor,
      rememberedDescriptor: localDescriptor,
      isLoading: false,
      error: null,
    };
    emit();
    return localDescriptor;
  } catch (error) {
    clearActiveDsm();
    terrainSourceState = {
      ...terrainSourceState,
      source: { mode: 'mapbox', datasetId: null },
      descriptor: null,
      isLoading: false,
      error: error instanceof Error ? error.message : String(error),
    };
    emit();
    throw error;
  }
}
