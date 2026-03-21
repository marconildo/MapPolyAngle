import type mapboxgl from "mapbox-gl";
import type { TileResult } from "./types";
import { tileCornersForImageSource } from "./controller";

/**
 * Convert a normalized value (0-1) to a heatmap color (RGB)
 * Uses a blue -> cyan -> green -> yellow -> red color scale
 */
function heatmapColor(t: number): [number, number, number] {
  // Clamp t to [0, 1]
  t = Math.max(0, Math.min(1, t));

  if (t < 0.25) {
    // Blue to Cyan (0 -> 0.25)
    const s = t / 0.25;
    return [0, Math.round(255 * s), 255];
  } else if (t < 0.5) {
    // Cyan to Green (0.25 -> 0.5)
    const s = (t - 0.25) / 0.25;
    return [0, 255, Math.round(255 * (1 - s))];
  } else if (t < 0.75) {
    // Green to Yellow (0.5 -> 0.75)
    const s = (t - 0.5) / 0.25;
    return [Math.round(255 * s), 255, 0];
  } else {
    // Yellow to Red (0.75 -> 1.0)
    const s = (t - 0.75) / 0.25;
    return [255, Math.round(255 * (1 - s)), 0];
  }
}

/**
 * Convert overlap values to a heatmap visualization
 * Higher overlap = warmer colors (red), lower overlap = cooler colors (blue)
 */
function encodeOverlapToImage(overlap: Uint16Array, size: number, maxValue: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const img = ctx.createImageData(size, size);
  const effectiveMax = Math.max(1, maxValue); // Consistent across tiles

  for (let i = 0, j = 0; i < overlap.length; i++, j += 4) {
    const v = overlap[i];
    if (v === 0) {
      // Transparent for no overlap
      img.data[j] = 0; img.data[j + 1] = 0; img.data[j + 2] = 0; img.data[j + 3] = 0;
      continue;
    }
    // Normalize to 0-1 range (consistent)
    const t = effectiveMax > 0 ? Math.min(1, v / effectiveMax) : 0;
    const [r, g, b] = heatmapColor(t);

    img.data[j] = r;
    img.data[j + 1] = g;
    img.data[j + 2] = b;
    img.data[j + 3] = 200; // Good opacity for overlay
  }

  ctx.putImageData(img, 0, 0);
  return canvas;
}

/**
 * Convert GSD values to a heatmap visualization
 * Higher GSD = worse resolution = warmer colors (red), Lower GSD = better resolution = cooler colors (blue/green)
 */
function valueRange(
  values: Float32Array,
  isValid: (value: number) => boolean
): { min: number; max: number } | null {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (let index = 0; index < values.length; index++) {
    const value = values[index];
    if (!isValid(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }

  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  return { min, max: Math.max(min + 1e-6, max) };
}

function encodeGsdToImage(gsd: Float32Array, size: number, gsdMin?: number, gsdMax?: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = size; canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const img = ctx.createImageData(size, size);
  const autoRange = valueRange(gsd, (value) => Number.isFinite(value));
  const lo = Math.max(0, gsdMin ?? autoRange?.min ?? 0.005);
  const hi = Math.max(lo + 1e-6, gsdMax ?? autoRange?.max ?? 0.06);

  for (let i = 0, j = 0; i < gsd.length; i++, j += 4) {
    const g = gsd[i];
    if (!Number.isFinite(g)) {
      // Transparent for invalid GSD
      img.data[j] = 0; img.data[j + 1] = 0; img.data[j + 2] = 0; img.data[j + 3] = 0;
      continue;
    }
    // Normalize within [gsdMin, gsdMax]: higher GSD (worse) → 1 (warmer); lower (better) → 0 (cooler)
    const t = Math.max(0, Math.min(1, (g - lo) / (hi - lo)));
    const [r, g_color, b] = heatmapColor(t);

    img.data[j] = r;
    img.data[j + 1] = g_color;
    img.data[j + 2] = b;
    img.data[j + 3] = 200; // Good opacity for overlay
  }

  ctx.putImageData(img, 0, 0);
  return canvas;
}

function smoothDensityForDisplay(density: Float32Array, size: number): Float32Array {
  const smoothed = new Float32Array(density.length);
  const kernel = [
    1, 2, 1,
    2, 4, 2,
    1, 2, 1,
  ];

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const index = y * size + x;
      const value = density[index];
      if (!(value > 0) || !Number.isFinite(value)) continue;

      // Keep tile borders raw. The blur is applied tile-by-tile, so smoothing the
      // outermost pixels creates visible seams where adjacent tiles cannot share
      // neighborhood samples.
      if (x === 0 || y === 0 || x === size - 1 || y === size - 1) {
        smoothed[index] = value;
        continue;
      }

      let weightedSum = 0;
      let totalWeight = 0;

      for (let ky = -1; ky <= 1; ky++) {
        const ny = y + ky;
        if (ny < 0 || ny >= size) continue;
        for (let kx = -1; kx <= 1; kx++) {
          const nx = x + kx;
          if (nx < 0 || nx >= size) continue;
          const neighborIndex = ny * size + nx;
          const neighbor = density[neighborIndex];
          if (!(neighbor > 0) || !Number.isFinite(neighbor)) continue;
          const weight = kernel[(ky + 1) * 3 + (kx + 1)];
          weightedSum += neighbor * weight;
          totalWeight += weight;
        }
      }

      smoothed[index] = totalWeight > 0 ? weightedSum / totalWeight : value;
    }
  }

  return smoothed;
}

function encodeDensityToImage(density: Float32Array, size: number, densityMin?: number, densityMax?: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const img = ctx.createImageData(size, size);
  const autoRange = valueRange(density, (value) => value > 0 && Number.isFinite(value));
  const lo = Math.max(0, densityMin ?? autoRange?.min ?? 10);
  const hi = Math.max(lo + 1e-6, densityMax ?? autoRange?.max ?? 100);
  const displayDensity = smoothDensityForDisplay(density, size);

  for (let i = 0, j = 0; i < density.length; i++, j += 4) {
    const rawValue = density[i];
    if (!(rawValue > 0) || !Number.isFinite(rawValue)) {
      img.data[j] = 0;
      img.data[j + 1] = 0;
      img.data[j + 2] = 0;
      img.data[j + 3] = 0;
      continue;
    }
    const value = displayDensity[i];
    // Invert density coloring so low density is worse/red and high density is better/blue.
    const t = 1 - Math.max(0, Math.min(1, (value - lo) / (hi - lo)));
    const [r, g, b] = heatmapColor(t);
    img.data[j] = r;
    img.data[j + 1] = g;
    img.data[j + 2] = b;
    img.data[j + 3] = 200;
  }

  ctx.putImageData(img, 0, 0);
  return canvas;
}

export function addOrUpdateTileOverlay(
  map: mapboxgl.Map,
  result: TileResult,
  opts: {
    kind: "overlap" | "pass" | "gsd" | "density";
    runId: string;
    opacity?: number;
    gsdMax?: number;
    gsdMin?: number;
    densityMax?: number;
    densityMin?: number;
  }
) {
  const idBase = `ogsd-${opts.runId}-${opts.kind}-${result.z}-${result.x}-${result.y}`;
  const sourceId = idBase;
  const layerId = idBase;

  const corners = tileCornersForImageSource(result.z, result.x, result.y);

  let canvas: HTMLCanvasElement;
  if (opts.kind === "overlap" || opts.kind === "pass") {
    canvas = encodeOverlapToImage(result.overlap, result.size, result.maxOverlap || 1);
  } else if (opts.kind === "density") {
    canvas = encodeDensityToImage(result.density ?? new Float32Array(result.size * result.size), result.size, opts.densityMin, opts.densityMax);
  } else {
    canvas = encodeGsdToImage(result.gsdMin, result.size, opts.gsdMin, opts.gsdMax);
  }

  // Convert to data URL (required for Mapbox image source)
  const url = canvas.toDataURL("image/png");

  const exists = !!map.getSource(sourceId);
  if (!exists) {
    map.addSource(sourceId, {
      type: "image",
      url,
      coordinates: corners,
    } as any);
    map.addLayer({
      id: layerId,
      type: "raster",
      source: sourceId,
      paint: { "raster-opacity": opts.opacity ?? 0.85 },
    });
  } else {
    // More efficient update: just update the source URL if possible
    try {
      const source: any = map.getSource(sourceId);
      if (source && source.updateImage) {
        source.updateImage({ url, coordinates: corners });
      } else {
        // Fallback to remove/add pattern for robustness
        if (map.getLayer(layerId)) map.removeLayer(layerId);
        if (map.getSource(sourceId)) map.removeSource(sourceId);
        map.addSource(sourceId, {
          type: "image",
          url,
          coordinates: corners,
        } as any);
        map.addLayer({
          id: layerId,
          type: "raster",
          source: sourceId,
          paint: { "raster-opacity": opts.opacity ?? 0.85 },
        });
      }
    } catch {
      // Fallback to remove/add pattern for robustness
      if (map.getLayer(layerId)) map.removeLayer(layerId);
      if (map.getSource(sourceId)) map.removeSource(sourceId);
      map.addSource(sourceId, {
        type: "image",
        url,
        coordinates: corners,
      } as any);
      map.addLayer({
        id: layerId,
        type: "raster",
        source: sourceId,
        paint: { "raster-opacity": opts.opacity ?? 0.85 },
      });
    }
  }
}

export function clearRunOverlays(map: mapboxgl.Map, runId: string) {
  if (!map.isStyleLoaded?.()) return;

  const layers = map.getStyle().layers || [];
  for (const layer of layers) {
    const id = layer.id;
    if (id.startsWith(`ogsd-${runId}-`)) {
      try {
        if (map.getLayer(id)) map.removeLayer(id);
        const sourceId = id; // we used same id
        if (map.getSource(sourceId)) map.removeSource(sourceId);
      } catch (e) {
        // Silently handle removal errors in case of rapid style changes
        console.warn(`Failed to remove overlay ${id}:`, e);
      }
    }
  }
}

// Remove all overlays produced by the GSD/overlap pipeline, regardless of runId.
export function clearAllOverlays(map: mapboxgl.Map) {
  if (!map.isStyleLoaded?.()) return;
  const layers = map.getStyle().layers || [];
  for (const layer of layers) {
    const id = layer.id;
    if (!id.startsWith('ogsd-')) continue;
    try {
      if (map.getLayer(id)) map.removeLayer(id);
      if (map.getSource(id)) map.removeSource(id);
    } catch { /* ignore */ }
  }
}
