import { fromArrayBuffer, fromBlob } from "geotiff";
import { convertCoordinates, toProj4 } from "geotiff-geokeys-to-proj4";
import proj4 from "proj4";
import { tileMetersBounds } from "@/overlap/mercator";
import type { Bounds, DsmSourceDescriptor, LngLatBounds } from "./types";

export interface DsmSourceState {
  descriptor: DsmSourceDescriptor | null;
  isLoading: boolean;
  error: string | null;
}

type LoadedDsmSource = {
  file: File;
  descriptor: DsmSourceDescriptor;
  image: any;
  sourceBounds: Bounds;
  sourceProj4: string;
  noDataValue: number | null;
  verticalScaleToMeters: number;
};

const listeners = new Set<() => void>();

let activeDsm: LoadedDsmSource | null = null;
let dsmState: DsmSourceState = {
  descriptor: null,
  isLoading: false,
  error: null,
};
let loadGeneration = 0;

const EXTRA_SAMPLE_ASSOC_ALPHA = 1;
const EXTRA_SAMPLE_UNASS_ALPHA = 2;
const INVALID_DSM_LAYOUT_MESSAGE =
  "This file is not a DSM. Upload a single-band elevation GeoTIFF; RGB/RGBA ortho imagery is not supported.";

function emit() {
  for (const listener of listeners) listener();
}

export function subscribeDsmSource(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getDsmSourceState(): DsmSourceState {
  return dsmState;
}

export function getActiveDsmDescriptor(): DsmSourceDescriptor | null {
  return activeDsm?.descriptor ?? null;
}

export function getActiveDsmFile(): File | null {
  return activeDsm?.file ?? null;
}

export function clearActiveDsm() {
  activeDsm = null;
  dsmState = {
    descriptor: null,
    isLoading: false,
    error: null,
  };
  emit();
}

function intersectBounds(a: Bounds, b: Bounds) {
  return !(a.maxX <= b.minX || a.minX >= b.maxX || a.maxY <= b.minY || a.minY >= b.maxY);
}

function terrainRgbEncodeMeters(elevationM: number): [number, number, number] {
  const encoded = Math.max(0, Math.min(256 * 256 * 256 - 1, Math.round((elevationM + 10000) * 10)));
  return [(encoded >> 16) & 255, (encoded >> 8) & 255, encoded & 255];
}

function boundsFromPoints(points: Array<[number, number]>): Bounds {
  return {
    minX: Math.min(...points.map((point) => point[0])),
    minY: Math.min(...points.map((point) => point[1])),
    maxX: Math.max(...points.map((point) => point[0])),
    maxY: Math.max(...points.map((point) => point[1])),
  };
}

function lngLatBoundsFromPoints(points: Array<[number, number]>): LngLatBounds {
  return {
    minLng: Math.min(...points.map((point) => point[0])),
    minLat: Math.min(...points.map((point) => point[1])),
    maxLng: Math.max(...points.map((point) => point[0])),
    maxLat: Math.max(...points.map((point) => point[1])),
  };
}

function inferSourceCrsCode(geoKeys: Record<string, any>): string | null {
  if (Number.isFinite(geoKeys.ProjectedCSTypeGeoKey)) return `EPSG:${geoKeys.ProjectedCSTypeGeoKey}`;
  if (Number.isFinite(geoKeys.GeographicTypeGeoKey)) return `EPSG:${geoKeys.GeographicTypeGeoKey}`;
  return null;
}

function generateLocalDsmId(): string {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `local-dsm-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function readExtraSamples(image: any): number[] {
  const value = image.fileDirectory.getValue("ExtraSamples");
  if (Array.isArray(value)) {
    return value.map((sample) => Number(sample)).filter((sample) => Number.isFinite(sample));
  }
  if (Number.isFinite(value)) {
    return [Number(value)];
  }
  return [];
}

function validateDsmSampleLayout(image: any): void {
  const samplesPerPixel = Number(image.getSamplesPerPixel?.() ?? 1);
  if (!Number.isFinite(samplesPerPixel) || samplesPerPixel <= 0) {
    throw new Error("GeoTIFF sample layout is invalid.");
  }
  const extraSamples = readExtraSamples(image);
  const alphaOrMaskExtraCount = extraSamples.filter(
    (sample) => sample === EXTRA_SAMPLE_ASSOC_ALPHA || sample === EXTRA_SAMPLE_UNASS_ALPHA,
  ).length;
  const unsupportedExtraCount = extraSamples.length - alphaOrMaskExtraCount;
  const baseSampleCount = samplesPerPixel - alphaOrMaskExtraCount;
  if (baseSampleCount !== 1 || unsupportedExtraCount > 0) {
    throw new Error(INVALID_DSM_LAYOUT_MESSAGE);
  }
}

async function inspectDsmGeoTiffFile(file: File): Promise<Omit<LoadedDsmSource, "file">> {
  const tiff = typeof FileReader === "function"
    ? await fromBlob(file)
    : await fromArrayBuffer(await file.arrayBuffer());
  const image = await tiff.getImage();
  validateDsmSampleLayout(image);

  const geoKeys = image.getGeoKeys() as Record<string, any>;
  const projObj = toProj4(geoKeys as any);
  const sourceProj4 = String(projObj.proj4 || "").trim();
  if (!sourceProj4) {
    throw new Error("Could not determine a usable source projection from the GeoTIFF geokeys.");
  }

  const width = image.getWidth();
  const height = image.getHeight();
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error("GeoTIFF dimensions are invalid.");
  }

  const [minX, minY, maxX, maxY] = image.getBoundingBox();
  const sourceBounds = { minX, minY, maxX, maxY };
  const footprint3857 = toProjectedBounds3857(sourceProj4, sourceBounds);
  const footprintLngLat = toFootprintLngLat(sourceProj4, sourceBounds);
  const noDataRaw = image.getGDALNoData();
  const noDataValue = noDataRaw == null ? null : Number.parseFloat(String(noDataRaw));
  const descriptor: DsmSourceDescriptor = {
    id: generateLocalDsmId(),
    name: file.name,
    fileSizeBytes: file.size,
    width,
    height,
    sourceBounds,
    footprint3857,
    footprintLngLat: footprintLngLat.bounds,
    footprintRingLngLat: footprintLngLat.ring,
    sourceCrsCode: inferSourceCrsCode(geoKeys),
    sourceCrsLabel: String(geoKeys.GTCitationGeoKey || inferSourceCrsCode(geoKeys) || "GeoTIFF CRS"),
    sourceProj4,
    horizontalUnits: typeof projObj.coordinatesUnits === "string" ? projObj.coordinatesUnits : null,
    verticalScaleToMeters:
      projObj.coordinatesConversionParameters &&
      Number.isFinite(projObj.coordinatesConversionParameters.z)
        ? projObj.coordinatesConversionParameters.z
        : 1,
    noDataValue: Number.isFinite(noDataValue) ? noDataValue : null,
    loadedAtIso: new Date().toISOString(),
  };

  return {
    descriptor,
    image,
    sourceBounds,
    sourceProj4,
    noDataValue: descriptor.noDataValue,
    verticalScaleToMeters: descriptor.verticalScaleToMeters,
  };
}

export async function validateDsmGeoTiffFile(file: File): Promise<void> {
  await inspectDsmGeoTiffFile(file);
}

function sourcePointToPixel(dsm: LoadedDsmSource, sourceX: number, sourceY: number) {
  const { minX, minY, maxX, maxY } = dsm.sourceBounds;
  const pixelSizeX = (maxX - minX) / dsm.descriptor.width;
  const pixelSizeY = (maxY - minY) / dsm.descriptor.height;
  return {
    col: (sourceX - minX) / pixelSizeX,
    row: (maxY - sourceY) / pixelSizeY,
  };
}

function toProjectedBounds3857(sourceProj4: string, sourceBounds: Bounds): Bounds {
  const sourceCorners: Array<[number, number]> = [
    [sourceBounds.minX, sourceBounds.maxY],
    [sourceBounds.maxX, sourceBounds.maxY],
    [sourceBounds.maxX, sourceBounds.minY],
    [sourceBounds.minX, sourceBounds.minY],
  ];
  const projected = sourceCorners.map(
    ([x, y]) => proj4(sourceProj4, "EPSG:3857", [x, y]) as [number, number]
  );
  return boundsFromPoints(projected);
}

function toFootprintLngLat(sourceProj4: string, sourceBounds: Bounds) {
  const sourceCorners: Array<[number, number]> = [
    [sourceBounds.minX, sourceBounds.maxY],
    [sourceBounds.maxX, sourceBounds.maxY],
    [sourceBounds.maxX, sourceBounds.minY],
    [sourceBounds.minX, sourceBounds.minY],
  ];
  const ring = sourceCorners.map(
    ([x, y]) => proj4(sourceProj4, "EPSG:4326", [x, y]) as [number, number]
  );
  ring.push(ring[0]);
  return {
    ring,
    bounds: lngLatBoundsFromPoints(ring.slice(0, 4)),
  };
}

export async function loadDsmFromFile(file: File): Promise<DsmSourceDescriptor> {
  const generation = ++loadGeneration;
  dsmState = {
    descriptor: dsmState.descriptor,
    isLoading: true,
    error: null,
  };
  emit();

  try {
    const loaded = await inspectDsmGeoTiffFile(file);
    const descriptor = loaded.descriptor;

    if (generation !== loadGeneration) return descriptor;

    activeDsm = {
      file,
      ...loaded,
    };
    dsmState = {
      descriptor,
      isLoading: false,
      error: null,
    };
    emit();
    return descriptor;
  } catch (error) {
    const message = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    if (generation === loadGeneration) {
      activeDsm = null;
      dsmState = {
        descriptor: null,
        isLoading: false,
        error: message,
      };
      emit();
    }
    throw error;
  }
}

export async function applyActiveDsmToTerrainRgbTile(
  z: number,
  x: number,
  y: number,
  size: number,
  data: Uint8ClampedArray
): Promise<boolean> {
  const dsm = activeDsm;
  if (!dsm) return false;

  const tileBounds = tileMetersBounds(z, x, y);
  const tileFootprint = {
    minX: tileBounds.minX,
    minY: tileBounds.minY,
    maxX: tileBounds.maxX,
    maxY: tileBounds.maxY,
  };
  if (!intersectBounds(tileFootprint, dsm.descriptor.footprint3857)) return false;

  const corners3857 = [
    [tileBounds.minX, tileBounds.maxY],
    [tileBounds.maxX, tileBounds.maxY],
    [tileBounds.minX, tileBounds.minY],
    [tileBounds.maxX, tileBounds.minY],
  ] as const;
  const sourceCorners = corners3857.map(
    ([mx, my]) => proj4("EPSG:3857", dsm.sourceProj4, [mx, my]) as [number, number]
  );

  const minSourceX = Math.min(...sourceCorners.map((corner) => corner[0]));
  const minSourceY = Math.min(...sourceCorners.map((corner) => corner[1]));
  const maxSourceX = Math.max(...sourceCorners.map((corner) => corner[0]));
  const maxSourceY = Math.max(...sourceCorners.map((corner) => corner[1]));
  const topLeftPx = sourcePointToPixel(dsm, minSourceX, maxSourceY);
  const bottomRightPx = sourcePointToPixel(dsm, maxSourceX, minSourceY);

  const window = [
    Math.max(0, Math.floor(topLeftPx.col) - 1),
    Math.max(0, Math.floor(topLeftPx.row) - 1),
    Math.min(dsm.descriptor.width, Math.ceil(bottomRightPx.col) + 1),
    Math.min(dsm.descriptor.height, Math.ceil(bottomRightPx.row) + 1),
  ] as [number, number, number, number];

  if (window[2] - window[0] < 2 || window[3] - window[1] < 2) return false;

  const raster = (await dsm.image.readRasters({
    window,
    width: size,
    height: size,
    interleave: true,
    fillValue: Number.NaN,
    resampleMethod: "bilinear",
  })) as Float32Array;

  let changed = false;
  for (let index = 0; index < raster.length; index++) {
    const value = raster[index];
    if (!Number.isFinite(value)) continue;
    if (dsm.noDataValue != null && Math.abs(value - dsm.noDataValue) < 1e-6) continue;

    const elevationM = convertCoordinates(0, 0, value, {
      x: 1,
      y: 1,
      z: dsm.verticalScaleToMeters,
    }).z;
    const [r, g, b] = terrainRgbEncodeMeters(elevationM);
    const out = index * 4;
    data[out] = r;
    data[out + 1] = g;
    data[out + 2] = b;
    data[out + 3] = 255;
    changed = true;
  }

  return changed;
}
