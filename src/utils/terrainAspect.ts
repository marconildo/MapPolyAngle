/***********************************************************************
 * terrainAspect.ts
 *
 * Compute a representative aspect (mean or median) for a polygonal
 * footprint and return the direction 90° from that aspect, i.e. the
 * azimuth along which ground elevation varies least on average.
 *
 * Author : <your‑name>, 2025‑07‑30
 ***********************************************************************/

export type LngLat = [lng: number, lat: number];

/** Simple ring polygon, no holes.  First and last vertex may be equal. */
export interface Polygon {
  coordinates: LngLat[];
}

/** One Mapbox tile worth of raster data that *covers* the polygon. */
export interface TerrainTile {
  /** Slippy‑map indices. */
  x: number;
  y: number;
  z: number;

  /** Raster dimensions – normally 256 × 256, but allow any. */
  width: number;
  height: number;

  /**
   * Pixel payload.
   *  * Terrain‑RGB – supply interleaved R,G,B (ignore A) length = w × h × 3 or 4
   *  * Single‑band DEM – supply float32 (metres) length = w × h
   */
  data: Uint8ClampedArray | Float32Array;

  /** `"terrain-rgb"` | `"dem"` */
  format: 'terrain-rgb' | 'dem';
}

export interface Options {
  /**
   * `'mean'` for circular mean (default, recommended because it is
   * unbiased and fast) or `'median'` for circular median (slower but
   * resistant to bimodality).
   */
  statistic?: 'mean' | 'median';
  /** Skip every *n* pixels to speed up large polygons (default = 1). */
  sampleStep?: number;
}

/** Return value. */
export interface AspectResult {
  /** Average or median aspect of the ground under the polygon [deg 0–360). */
  aspectDeg: number;
  /**
   * Direction 90° clockwise from aspect (constant‑height flight line),
   * again in degrees clockwise from north [deg 0–360).
   */
  contourDirDeg: number;
  /** Number of raster samples that contributed. 0 → polygon too small. */
  samples: number;
}

// ---------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------

export function dominantContourDirection(
  polygon: Polygon,
  tiles: TerrainTile[],
  opts: Options = {},
): AspectResult {
  const {
    statistic = 'mean',
    sampleStep = 1,
  } = opts;

  const bearings: number[] = [];
  const gradientMagnitudes: number[] = []; // Track gradient strength

  for (const tile of tiles) {
    const proj = new WebMercatorProjector(tile.z);

    for (let py = 1; py < tile.height - 1; py += sampleStep) {
      for (let px = 1; px < tile.width - 1; px += sampleStep) {
        const [lng, lat] = proj.pixelToLngLat(tile.x, tile.y, px + 0.5, py + 0.5, tile.width);

        if (!pointInPolygon(lng, lat, polygon.coordinates)) continue;

        const elev = neighbourhood9(tile, px, py);
        if (elev === null) continue;

        const res = proj.pixelResolution(lat, tile.width);

        // Horn gradient components - FIXED Y-COORDINATE SIGN
        const dzdx = ((elev[2] + 2*elev[5] + elev[8]) - (elev[0] + 2*elev[3] + elev[6])) / (8*res);
        const dzdy = ((elev[0] + 2*elev[1] + elev[2]) - (elev[6] + 2*elev[7] + elev[8])) / (8*res); // FIXED: flipped sign

        if (!Number.isFinite(dzdx) || !Number.isFinite(dzdy)) continue;

        // Calculate gradient magnitude for filtering
        const gradMag = Math.sqrt(dzdx * dzdx + dzdy * dzdy);
        if (gradMag < 1e-6) continue; // Skip nearly flat areas (adjust threshold as needed)

        // Store gradient magnitude for validation
        gradientMagnitudes.push(gradMag);

        // Contour bearing: perpendicular to gradient
        // Gradient (dzdx, dzdy) points uphill
        // Contour direction is (-dzdy, dzdx) - perpendicular vector
        const theta = Math.atan2(-dzdy, dzdx);
        bearings.push(theta < 0 ? theta + 2*Math.PI : theta);
      }
    }
  }

  // Enhanced validation
  if (bearings.length < 10) { // Need minimum samples
    console.warn(`Only ${bearings.length} samples found - may be unreliable`);
    return { aspectDeg: NaN, contourDirDeg: NaN, samples: bearings.length };
  }

  // Check if terrain has sufficient variation
  const avgGradMag = gradientMagnitudes.reduce((a, b) => a + b, 0) / gradientMagnitudes.length;
  if (avgGradMag < 0.001) { // Very flat terrain threshold
    console.warn(`Terrain is very flat (avg gradient: ${avgGradMag.toFixed(6)}) - direction may be unreliable`);
  }

  // Calculate circular dispersion to assess reliability
  const dispersion = calculateCircularDispersion(bearings);
  if (dispersion > 0.8) { // High dispersion indicates random/conflicting directions
    console.warn(`High directional dispersion (${dispersion.toFixed(3)}) - terrain may have conflicting slope directions`);
  }

  const dirRad = (statistic === 'median')
    ? circularMedian(bearings)
    : circularMean(bearings);

  return {
    aspectDeg: NaN,
    contourDirDeg: radToDeg(dirRad),
    samples: bearings.length,
  };
}

// ---------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------

/**
 * Decode 3×3 Horn kernel neighbourhood centred on (px,py).
 * Returns a length‑9 float[] in row‑major (Z1..Z9) order or null on edge.
 */
function neighbourhood9(tile: TerrainTile, px: number, py: number): number[] | null {
  const { width, height } = tile;
  if (px < 1 || py < 1 || px >= width - 1 || py >= height - 1) return null;

  const elev = new Array<number>(9);
  let idx = 0;
  for (let dy = -1; dy <= 1; ++dy) {
    for (let dx = -1; dx <= 1; ++dx) {
      elev[idx++] = getElevation(tile, px + dx, py + dy);
    }
  }
  return elev;
}

// Add circular dispersion calculation for reliability assessment
function calculateCircularDispersion(angles: number[]): number {
  if (angles.length === 0) return 1;

  let sx = 0, sy = 0;
  for (const a of angles) {
    sx += Math.cos(a);
    sy += Math.sin(a);
  }

  const R = Math.sqrt(sx * sx + sy * sy) / angles.length;
  return 1 - R; // 0 = perfect agreement, 1 = completely random
}

/** Circular mean of angles in radians. */
function circularMean(arr: number[]): number {
  let sx = 0, sy = 0;
  for (const a of arr) {
    sx += Math.cos(a);
    sy += Math.sin(a);
  }
  return (Math.atan2(sy, sx) + 2*Math.PI) % (2*Math.PI);
}

/** Read elevation (metres) for pixel (px,py) in a tile. */
function getElevation(tile: TerrainTile, px: number, py: number): number {
  const idx = py * tile.width + px;
  if (tile.format === 'dem') {
    // Float32 DEM – 1 band
    return (tile.data as Float32Array)[idx];
  } else {
    // Terrain‑RGB – 4 bands (RGBA) or 3 bands (RGB)
    const base = idx * (tile.data.length === tile.width * tile.height * 4 ? 4 : 3);
    const r = tile.data[base];
    const g = tile.data[base + 1];
    const b = tile.data[base + 2];
    // Mapbox formula  height = -10000 + ((R*256*256 + G*256 + B)*0.1)  docs:
    // https://docs.mapbox.com/data/tilesets/reference/mapbox-terrain-rgb-v1/
    return -10000 + ((r * 256 * 256 + g * 256 + b) * 0.1);
  }
}

/** Projector utilities for Web‑Mercator tile mathematics. */
class WebMercatorProjector {
  private readonly z2: number;
  constructor(private readonly z: number) { this.z2 = 2 ** z; }

  /**
   * Convert pixel to lng/lat (EPSG:4326).
   * px/py are pixel indices inside the tile (0..width).
   * width = height because Mapbox tiles are square – pass width as arg.
   */
  pixelToLngLat(tx: number, ty: number, px: number, py: number, width: number): LngLat {
    const normX = (tx * width + px) / (this.z2 * width);
    const normY = (ty * width + py) / (this.z2 * width);

    const lng = normX * 360 - 180;
    const n = Math.PI - 2 * Math.PI * normY;
    const lat = (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
    return [lng, lat];
  }

  /** Horizontal ground resolution (metres per pixel) at latitude. */
  pixelResolution(lat: number, tilePx: number): number {
    const earthCircum = 40075016.68557849; // metres
    return Math.cos(degToRad(lat)) * earthCircum / (this.z2 * tilePx);
  }
}

/** Ray–crossing even–odd test for a single ring polygon. */
function pointInPolygon(lng: number, lat: number, ring: LngLat[]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];

    const intersect =
      ((yi > lat) !== (yj > lat)) &&
      (lng < (xj - xi) * (lat - yi) / (yj - yi + 1e-12) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

/** Fast circular median using angular sort (O(n log n)). */
function circularMedian(angles: number[]): number {
  // Map to unit circle
  const sorted = angles.slice().sort((a, b) => a - b);
  // Trick: duplicate list shifted by 2π so we can treat wrap‑around windows
  const dup = sorted.concat(sorted.map(a => a + 2 * Math.PI));

  // Sliding window of length n to find minimal range
  const n = angles.length;
  let bestRange = Infinity, bestStart = 0;
  let j = 0;
  for (let i = 0; i < n; ++i) {
    while (j < i + n && dup[j] - dup[i] <= Math.PI) ++j;
    const range = dup[j - 1] - dup[i];
    if (range < bestRange) {
      bestRange = range;
      bestStart = i;
    }
  }
  // The median is middle element of this minimal‑range window
  const medianIdx = bestStart + Math.floor(n / 2);
  return dup[medianIdx] % (2 * Math.PI);
}

/* Utility deg↔rad */
const degToRad = (d: number) => d * Math.PI / 180;
const radToDeg = (r: number) => (r * 180 / Math.PI + 360) % 360;

/* ------------------------------------------------------------------ */
/*                    End of terrainAspect.ts                         */
/* ------------------------------------------------------------------ */
