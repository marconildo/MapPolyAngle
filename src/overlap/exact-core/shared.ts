import * as egm96 from "egm96-universal";

import type { DensityStats, GSDStats, PolygonLngLatWithId } from "../types";
import { tileMetersBounds, worldToPixel } from "../mercator";
import { rasterizeRingsToMask } from "../rasterize";

export function erode1px8(src: Uint8Array, size: number, dst: Uint8Array) {
  dst.fill(0);
  for (let y = 1; y < size - 1; y++) {
    const row = y * size;
    for (let x = 1; x < size - 1; x++) {
      const i = row + x;
      if (!src[i]) continue;
      const keep =
        src[i - 1] & src[i + 1] &
        src[i - size] & src[i + size] &
        src[i - size - 1] & src[i - size + 1] &
        src[i + size - 1] & src[i + size + 1];
      if (keep) dst[i] = 1;
    }
  }
}

export function erodeN8(src: Uint8Array, size: number, radiusPx: number): Uint8Array {
  if (!(radiusPx > 0)) return src;
  let a: Uint8Array = src;
  let b: Uint8Array = new Uint8Array(size * size);
  for (let k = 0; k < radiusPx; k++) {
    erode1px8(a, size, b);
    const tmp = a;
    a = b;
    b = tmp;
  }
  return a === src ? new Uint8Array(a) : a;
}

export function cropCenter(src: Uint8Array, sizePad: number, size: number, pad: number): Uint8Array {
  if (pad === 0) return src;
  const out = new Uint8Array(size * size);
  let w = 0;
  for (let y = pad; y < pad + size; y++) {
    const rowBase = y * sizePad + pad;
    for (let x = 0; x < size; x++) out[w++] = src[rowBase + x];
  }
  return out;
}

export function ringsToPixelsWithTx(
  polygons: PolygonLngLatWithId[],
  tx: { minX: number; maxX: number; minY: number; maxY: number; pixelSize: number },
) {
  const ringsPerPoly: Array<Array<[number, number]>> = [];
  const ids: string[] = [];
  for (let k = 0; k < polygons.length; k++) {
    const ring = polygons[k].ring;
    const ringPx: Array<[number, number]> = [];
    for (let i = 0; i < ring.length; i++) {
      const lng = ring[i][0];
      const lat = Math.max(-85.05112878, Math.min(85.05112878, ring[i][1]));
      const mx = (lng * Math.PI / 180) * 6378137;
      const my = 6378137 * Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI / 180) / 2));
      const wp = worldToPixel(tx, mx, my);
      ringPx.push([wp[0], wp[1]]);
    }
    if (ringPx.length >= 3) {
      ringsPerPoly.push(ringPx);
      ids.push(polygons[k].id ?? String(k));
    }
  }
  return { ringsPerPoly, ids };
}

export function buildPolygonMasks(
  polygons: PolygonLngLatWithId[],
  z: number,
  x: number,
  y: number,
  size: number,
  erodeRadiusPx: number,
): { tx: ReturnType<typeof tileMetersBounds>; masks: Uint8Array[]; unionMask: Uint8Array; ids: string[] } {
  const base = tileMetersBounds(z, x, y);
  const pix = (base.maxX - base.minX) / size;
  const pad = Math.max(0, erodeRadiusPx);
  const sizePad = size + 2 * pad;
  const txPad = {
    minX: base.minX - pad * pix,
    maxX: base.maxX + pad * pix,
    minY: base.minY - pad * pix,
    maxY: base.maxY + pad * pix,
    pixelSize: pix,
  };
  const { ringsPerPoly, ids } = ringsToPixelsWithTx(polygons, txPad);
  const masks: Uint8Array[] = [];
  for (let i = 0; i < ringsPerPoly.length; i++) {
    const maskPad = rasterizeRingsToMask([ringsPerPoly[i]], sizePad);
    const erodedPad = erodeN8(maskPad, sizePad, pad);
    masks.push(pad > 0 ? cropCenter(erodedPad, sizePad, size, pad) : erodedPad);
  }
  const unionMask = new Uint8Array(size * size);
  for (const mask of masks) {
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) unionMask[i] = 1;
    }
  }
  return { tx: base, masks, unionMask, ids };
}

export function convertElevationsToWGS84Ellipsoid(
  elevEGM96: Float32Array,
  size: number,
  tx: { minX: number; maxX: number; minY: number; maxY: number },
): Float32Array {
  const elevWGS84 = new Float32Array(size * size);
  const pixelSize = (tx.maxX - tx.minX) / size;
  const radius = 6378137;

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const idx = row * size + col;
      const x = tx.minX + (col + 0.5) * pixelSize;
      const y = tx.maxY - (row + 0.5) * pixelSize;
      const lon = (x / radius) * (180 / Math.PI);
      const lat = Math.atan(Math.sinh(y / radius)) * (180 / Math.PI);
      elevWGS84[idx] = egm96.egm96ToEllipsoid(lat, lon, elevEGM96[idx]);
    }
  }

  return elevWGS84;
}

export function calculateGSDStatsFast(
  gsdMin: Float32Array,
  activeIdxs: Uint32Array,
  pixelAreaEquator: number,
  size: number,
  cosLatPerRow: Float64Array,
): GSDStats {
  let count = 0;
  let sum = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = 0;
  let totalAreaM2 = 0;

  for (let i = 0; i < activeIdxs.length; i++) {
    const idx = activeIdxs[i];
    const gsd = gsdMin[idx];
    if (!(gsd > 0 && Number.isFinite(gsd))) continue;
    count++;
    sum += gsd;
    const row = (idx / size) | 0;
    const cosPhi = cosLatPerRow[row];
    const area = pixelAreaEquator * cosPhi * cosPhi;
    totalAreaM2 += area;
    if (gsd < min) min = gsd;
    if (gsd > max) max = gsd;
  }

  if (count === 0 || !Number.isFinite(min)) {
    return { min: 0, max: 0, mean: 0, count: 0, totalAreaM2: 0, histogram: [] };
  }

  const mean = sum / count;
  const maxBins = 20;
  const minBinSize = 0.01;
  const span = max - min;
  let numBins = span <= 0 ? 1 : maxBins;
  if (span > 0 && (span / numBins) < minBinSize) {
    numBins = Math.max(1, Math.floor(span / minBinSize));
  }

  const histogram = new Array<{ bin: number; count: number; areaM2?: number }>(numBins);
  for (let b = 0; b < numBins; b++) histogram[b] = { bin: 0, count: 0, areaM2: 0 };

  if (span <= 0) {
    histogram[0].count = count;
    histogram[0].bin = min;
    histogram[0].areaM2 = totalAreaM2;
  } else {
    const binSize = span / numBins;
    for (let i = 0; i < activeIdxs.length; i++) {
      const idx = activeIdxs[i];
      const value = gsdMin[idx];
      if (!(value > 0 && Number.isFinite(value))) continue;
      let binIndex = Math.floor((value - min) / binSize);
      if (binIndex >= numBins) binIndex = numBins - 1;
      const row = (idx / size) | 0;
      const cosPhi = cosLatPerRow[row];
      const area = pixelAreaEquator * cosPhi * cosPhi;
      histogram[binIndex].count += 1;
      histogram[binIndex].areaM2 = (histogram[binIndex].areaM2 || 0) + area;
    }
    for (let b = 0; b < numBins; b++) histogram[b].bin = min + (b + 0.5) * binSize;
  }

  return { min, max, mean, count, totalAreaM2, histogram };
}

export function calculateDensityStats(
  density: Float32Array,
  activeIdxs: Uint32Array,
  pixelAreaEquator: number,
  size: number,
  cosLatPerRow: Float64Array,
  includeZeroDensity = false,
): DensityStats {
  let count = 0;
  let sum = 0;
  let min = Number.POSITIVE_INFINITY;
  let max = 0;
  let totalAreaM2 = 0;

  for (let i = 0; i < activeIdxs.length; i++) {
    const idx = activeIdxs[i];
    const rawValue = density[idx];
    const value = includeZeroDensity
      ? ((Number.isFinite(rawValue) && rawValue > 0) ? rawValue : 0)
      : rawValue;
    if (!(includeZeroDensity || (value > 0 && Number.isFinite(value)))) continue;
    if (!Number.isFinite(value)) continue;
    const row = (idx / size) | 0;
    const cosPhi = cosLatPerRow[row];
    const areaM2 = pixelAreaEquator * cosPhi * cosPhi;
    count++;
    sum += value * areaM2;
    totalAreaM2 += areaM2;
    if (value < min) min = value;
    if (value > max) max = value;
  }

  if (count === 0 || !Number.isFinite(min)) {
    return { min: 0, max: 0, mean: 0, count: 0, totalAreaM2: 0, histogram: [] };
  }

  const mean = totalAreaM2 > 0 ? (sum / totalAreaM2) : 0;
  const maxBins = 20;
  const exactZeroBucket = { bin: 0, count: 0, areaM2: 0 };
  const positiveValues: number[] = [];

  if (includeZeroDensity) {
    for (let i = 0; i < activeIdxs.length; i++) {
      const idx = activeIdxs[i];
      const rawValue = density[idx];
      const value = Number.isFinite(rawValue) && rawValue > 0 ? rawValue : 0;
      if (!Number.isFinite(value)) continue;
      const row = (idx / size) | 0;
      const cosPhi = cosLatPerRow[row];
      const areaM2 = pixelAreaEquator * cosPhi * cosPhi;
      if (value === 0) {
        exactZeroBucket.count += 1;
        exactZeroBucket.areaM2 += areaM2;
      } else {
        positiveValues.push(value);
      }
    }
  }

  const positiveBinCount = exactZeroBucket.count > 0 ? maxBins - 1 : maxBins;
  const histogram: Array<{ bin: number; count: number; areaM2?: number }> = [];

  const positiveMin = includeZeroDensity && exactZeroBucket.count > 0
    ? (positiveValues.length > 0 ? Math.min(...positiveValues) : 0)
    : min;
  const positiveMax = max;
  const positiveSpan = positiveMax - positiveMin;

  if (!(positiveValues.length > 0) && exactZeroBucket.count > 0) {
    histogram.push(exactZeroBucket);
  } else if (positiveSpan <= 0) {
    if (exactZeroBucket.count > 0) histogram.push(exactZeroBucket);
    histogram.push({
      bin: positiveMin,
      count: count - exactZeroBucket.count,
      areaM2: totalAreaM2 - exactZeroBucket.areaM2,
    });
  } else {
    if (exactZeroBucket.count > 0) histogram.push(exactZeroBucket);
    const binSize = positiveSpan / positiveBinCount;
    const positiveBins = new Array<{ bin: number; count: number; areaM2?: number }>(positiveBinCount);
    for (let i = 0; i < positiveBinCount; i++) positiveBins[i] = { bin: 0, count: 0, areaM2: 0 };
    for (let i = 0; i < activeIdxs.length; i++) {
      const idx = activeIdxs[i];
      const rawValue = density[idx];
      const value = includeZeroDensity
        ? ((Number.isFinite(rawValue) && rawValue > 0) ? rawValue : 0)
        : rawValue;
      if (!(includeZeroDensity || (value > 0 && Number.isFinite(value)))) continue;
      if (!Number.isFinite(value)) continue;
      if (exactZeroBucket.count > 0 && value === 0) continue;
      let binIndex = Math.floor((value - positiveMin) / binSize);
      if (binIndex >= positiveBinCount) binIndex = positiveBinCount - 1;
      const row = (idx / size) | 0;
      const cosPhi = cosLatPerRow[row];
      const areaM2 = pixelAreaEquator * cosPhi * cosPhi;
      positiveBins[binIndex].count += 1;
      positiveBins[binIndex].areaM2 = (positiveBins[binIndex].areaM2 || 0) + areaM2;
    }
    for (let i = 0; i < positiveBinCount; i++) positiveBins[i].bin = positiveMin + (i + 0.5) * binSize;
    histogram.push(...positiveBins);
  }

  return {
    min,
    max,
    mean,
    count,
    totalAreaM2,
    histogram: histogram.filter((bin) => bin.count > 0 || (bin.areaM2 || 0) > 0),
  };
}
