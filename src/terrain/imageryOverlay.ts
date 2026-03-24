import { fromArrayBuffer, fromBlob } from "geotiff";
import { toProj4 } from "geotiff-geokeys-to-proj4";
import proj4 from "proj4";

import type { Bounds, GeoTiffSourceDescriptor, ImageryOverlayDescriptor, ImageryOverlayState, LngLatBounds } from "./types";

type ActiveImageryOverlay = {
  file: File;
  descriptor: ImageryOverlayDescriptor;
  imageUrl: string;
  coordinates: [[number, number], [number, number], [number, number], [number, number]];
};

const MAX_OVERLAY_RENDER_DIMENSION = 4096;

const listeners = new Set<() => void>();

let activeImageryOverlay: ActiveImageryOverlay | null = null;
let imageryOverlayState: ImageryOverlayState = {
  descriptor: null,
  isLoading: false,
  error: null,
};

function emit() {
  for (const listener of listeners) listener();
}

export function subscribeImageryOverlay(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getImageryOverlayState(): ImageryOverlayState {
  return imageryOverlayState;
}

export function getActiveImageryOverlay(): ActiveImageryOverlay | null {
  return activeImageryOverlay;
}

export function clearActiveImageryOverlay() {
  if (activeImageryOverlay?.imageUrl) {
    URL.revokeObjectURL(activeImageryOverlay.imageUrl);
  }
  activeImageryOverlay = null;
  imageryOverlayState = {
    descriptor: null,
    isLoading: false,
    error: null,
  };
  emit();
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

function generateLocalImageryId(): string {
  if (globalThis.crypto?.randomUUID) {
    return globalThis.crypto.randomUUID();
  }
  return `imagery-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

async function openGeoTiff(file: File) {
  if (typeof FileReader === "function") {
    return fromBlob(file);
  }
  return fromArrayBuffer(await file.arrayBuffer());
}

function descriptorFromGeoTiffDescriptor(file: File, shared: Omit<GeoTiffSourceDescriptor, "id" | "name" | "fileSizeBytes" | "loadedAtIso">): ImageryOverlayDescriptor {
  return {
    id: generateLocalImageryId(),
    name: file.name,
    fileSizeBytes: file.size,
    loadedAtIso: new Date().toISOString(),
    ...shared,
  };
}

async function renderImageToObjectUrl(image: any): Promise<string> {
  const sourceWidth = image.getWidth();
  const sourceHeight = image.getHeight();
  const scale = Math.min(1, MAX_OVERLAY_RENDER_DIMENSION / Math.max(sourceWidth, sourceHeight, 1));
  const renderWidth = Math.max(1, Math.round(sourceWidth * scale));
  const renderHeight = Math.max(1, Math.round(sourceHeight * scale));
  const raster = await image.readRGB({
    interleave: true,
    enableAlpha: true,
    width: renderWidth,
    height: renderHeight,
    resampleMethod: "bilinear",
  });

  const channelCount = Math.max(1, Math.round(raster.length / Math.max(renderWidth * renderHeight, 1)));
  if (channelCount < 3 || channelCount > 4) {
    throw new Error("Could not render imagery overlay from this GeoTIFF.");
  }

  const canvas = document.createElement("canvas");
  canvas.width = renderWidth;
  canvas.height = renderHeight;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Could not create a canvas for the imagery overlay.");
  }

  const imageData = context.createImageData(renderWidth, renderHeight);
  const rgba = imageData.data;
  const hasAlpha = channelCount >= 4;
  for (let pixelIndex = 0; pixelIndex < renderWidth * renderHeight; pixelIndex += 1) {
    const sourceOffset = pixelIndex * channelCount;
    const destOffset = pixelIndex * 4;
    rgba[destOffset] = raster[sourceOffset] ?? 0;
    rgba[destOffset + 1] = raster[sourceOffset + 1] ?? raster[sourceOffset] ?? 0;
    rgba[destOffset + 2] = raster[sourceOffset + 2] ?? raster[sourceOffset] ?? 0;
    rgba[destOffset + 3] = hasAlpha ? (raster[sourceOffset + 3] ?? 255) : 255;
  }
  context.putImageData(imageData, 0, 0);

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((nextBlob) => {
      if (nextBlob) {
        resolve(nextBlob);
        return;
      }
      reject(new Error("Could not encode the imagery overlay."));
    }, "image/png");
  });
  return URL.createObjectURL(blob);
}

export async function loadImageryOverlayFromFile(file: File): Promise<ImageryOverlayDescriptor> {
  imageryOverlayState = {
    ...imageryOverlayState,
    isLoading: true,
    error: null,
  };
  emit();

  try {
    const tiff = await openGeoTiff(file);
    const image = await tiff.getImage();
    const geoKeys = image.getGeoKeys() as Record<string, any>;
    const projObj = toProj4(geoKeys as any);
    const sourceProj4 = String(projObj.proj4 || "").trim();
    if (!sourceProj4) {
      throw new Error("Could not determine a usable source projection from the GeoTIFF geokeys.");
    }

    const [minX, minY, maxX, maxY] = image.getBoundingBox();
    const sourceBounds = { minX, minY, maxX, maxY };
    const sourceCorners: [[number, number], [number, number], [number, number], [number, number]] = [
      [sourceBounds.minX, sourceBounds.maxY],
      [sourceBounds.maxX, sourceBounds.maxY],
      [sourceBounds.maxX, sourceBounds.minY],
      [sourceBounds.minX, sourceBounds.minY],
    ];
    const projected3857 = sourceCorners.map(
      ([x, y]) => proj4(sourceProj4, "EPSG:3857", [x, y]) as [number, number],
    );
    const projected4326 = sourceCorners.map(
      ([x, y]) => proj4(sourceProj4, "EPSG:4326", [x, y]) as [number, number],
    );

    const descriptor = descriptorFromGeoTiffDescriptor(file, {
      width: image.getWidth(),
      height: image.getHeight(),
      sourceBounds,
      footprint3857: boundsFromPoints(projected3857),
      footprintLngLat: lngLatBoundsFromPoints(projected4326),
      footprintRingLngLat: [...projected4326, projected4326[0]],
      sourceCrsCode: inferSourceCrsCode(geoKeys),
      sourceCrsLabel: String(geoKeys.GTCitationGeoKey || inferSourceCrsCode(geoKeys) || "GeoTIFF CRS"),
      sourceProj4,
      horizontalUnits: typeof projObj.coordinatesUnits === "string" ? projObj.coordinatesUnits : null,
    });
    const imageUrl = await renderImageToObjectUrl(image);

    if (activeImageryOverlay?.imageUrl) {
      URL.revokeObjectURL(activeImageryOverlay.imageUrl);
    }
    activeImageryOverlay = {
      file,
      descriptor,
      imageUrl,
      coordinates: [
        projected4326[0],
        projected4326[1],
        projected4326[2],
        projected4326[3],
      ],
    };
    imageryOverlayState = {
      descriptor,
      isLoading: false,
      error: null,
    };
    emit();
    return descriptor;
  } catch (error) {
    if (activeImageryOverlay?.imageUrl) {
      URL.revokeObjectURL(activeImageryOverlay.imageUrl);
    }
    activeImageryOverlay = null;
    imageryOverlayState = {
      descriptor: null,
      isLoading: false,
      error: error instanceof Error ? error.message : String(error),
    };
    emit();
    throw error;
  }
}
