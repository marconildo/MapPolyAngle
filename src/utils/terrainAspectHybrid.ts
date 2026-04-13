/***********************************************************************
 * terrainAspectHybrid.ts
 *
 * Hybrid plane fitting combining the best of both approaches:
 * - ChatGPT's data centering and Huber weights for simplicity
 * - Enhanced quality metrics and error handling
 * - Proper EPSG:3857 projection with numerical conditioning
 *
 * Author: Asus Christmus, 2025-07-30
 ***********************************************************************/

export type LngLat = [lng: number, lat: number];

export interface Polygon {
  coordinates: LngLat[];
}

export interface TerrainTile {
  x: number;
  y: number;
  z: number;
  width: number;
  height: number;
  data: Uint8ClampedArray | Float32Array;
  format?: 'terrain-rgb' | 'dem';
}

export interface Options {
  statistic?: 'mean' | 'median';
  sampleStep?: number;
}

export interface AspectResult {
  aspectDeg: number;
  contourDirDeg: number;
  samples: number;
  // Enhanced metrics
  rSquared?: number;
  rmse?: number;
  slopeMagnitude?: number;
  fitQuality?: 'excellent' | 'good' | 'fair' | 'poor';
  maxElevation?: number; // Maximum elevation found in the terrain data (meters)
}

// ---------------------------------------------------------------------
// Public API - Hybrid plane-fit version
// ---------------------------------------------------------------------

export function dominantContourDirectionPlaneFit(
  polygon: Polygon,
  tiles: TerrainTile[],
  opts: Options = {},
): AspectResult {
  const { sampleStep = 1 } = opts;

  // --- 1. Gather samples inside the polygon --------------------------
  const samples: { x: number; y: number; z: number }[] = [];
  let maxZ = -Infinity; // Track maximum elevation

  for (const tile of tiles) {
    const proj = new WebMercatorProjector(tile.z);

    for (let py = 0; py < tile.height; py += sampleStep) {
      for (let px = 0; px < tile.width; px += sampleStep) {
        // Geographic center of this DEM pixel
        const [lng, lat] = proj.pixelToLngLat(
          tile.x, tile.y, px + 0.5, py + 0.5, tile.width,
        );

        if (!pointInPolygon(lng, lat, polygon.coordinates)) continue;

        const z = getElevation(tile, px, py);
        if (!Number.isFinite(z)) continue;

        // Track maximum elevation
        maxZ = Math.max(maxZ, z);

        // Convert to EPSG:3857 (Spherical Mercator) metres
        const [x, y] = lngLatToMercatorMeters(lng, lat);
        samples.push({ x, y, z });
      }
    }
  }

  if (samples.length < 6) {
    return {
      aspectDeg: NaN,
      contourDirDeg: NaN,
      samples: samples.length,
      fitQuality: 'poor',
      maxElevation: maxZ === -Infinity ? undefined : maxZ
    };
  }

  // --- 2. Robust least-squares plane fit -----------------------------
  const planeResult = fitPlaneHybrid(samples);
  if (!planeResult.ok) {
    return {
      aspectDeg: NaN,
      contourDirDeg: NaN,
      samples: samples.length,
      fitQuality: 'poor',
      maxElevation: maxZ === -Infinity ? undefined : maxZ
    };
  }

  const { a, b, rSquared, rmse } = planeResult;

  // Check if terrain is essentially flat
  const slopeMagnitude = Math.sqrt(a * a + b * b);
  if (slopeMagnitude < 1e-8) {
    return {
      aspectDeg: NaN,
      contourDirDeg: NaN,
      samples: samples.length,
      rSquared,
      rmse,
      slopeMagnitude,
      fitQuality: 'poor',
      maxElevation: maxZ === -Infinity ? undefined : maxZ
    };
  }

  // Gradient (∂z/∂x = a, ∂z/∂y = b) points uphill (east & north components)
  const aspectRad = (Math.atan2(a, b) + 2 * Math.PI) % (2 * Math.PI); // 0° = north, CW = east
  const contourRad = (aspectRad + Math.PI / 2) % (2 * Math.PI);        // +90° gives level line

  // Assess fit quality
  const fitQuality = assessFitQuality(rSquared, rmse, slopeMagnitude, samples.length);

  return {
    aspectDeg: radToDegWrapped(aspectRad),
    contourDirDeg: radToDegWrapped(contourRad),
    samples: samples.length,
    rSquared,
    rmse,
    slopeMagnitude,
    fitQuality,
    maxElevation: maxZ === -Infinity ? undefined : maxZ
  };
}

// ---------------------------------------------------------------------
// Hybrid Plane Fitting (Huber IRLS with quality metrics)
// ---------------------------------------------------------------------

interface PlaneResult {
  a: number;
  b: number;
  c: number;
  ok: boolean;
  rSquared: number;
  rmse: number;
}

function fitPlaneHybrid(pts: { x: number; y: number; z: number }[]): PlaneResult {
  const n = pts.length;

  // Center data for numerical stability (ChatGPT's key insight)
  let mx = 0, my = 0, mz = 0;
  for (const p of pts) {
    mx += p.x;
    my += p.y;
    mz += p.z;
  }
  mx /= n; my /= n; mz /= n;

  // 3 iterations of IRLS with Huber weights (simple and effective)
  let a = 0, b = 0, c = mz; // Initial: horizontal plane

  for (let iter = 0; iter < 3; ++iter) {
    // Accumulate weighted normal equations: [Sxx Sxy][a] = [Sxz]
    //                                       [Sxy Syy][b]   [Syz]
    let Sxx = 0, Sxy = 0, Syy = 0, Sxz = 0, Syz = 0;

    for (const p of pts) {
      const x = p.x - mx;
      const y = p.y - my;
      const z = p.z - mz;

      const residual = z - (a * x + b * y);
      const absResidual = Math.abs(residual);

      // Huber weights: full weight below threshold, 1/|r| above
      const threshold = 5; // 5 meters - reasonable for terrain data
      const weight = absResidual < threshold ? 1 : threshold / absResidual;

      Sxx += weight * x * x;
      Sxy += weight * x * y;
      Syy += weight * y * y;
      Sxz += weight * x * z;
      Syz += weight * y * z;
    }

    // Solve 2×2 system (much more efficient than 3×3)
    const det = Sxx * Syy - Sxy * Sxy;
    if (Math.abs(det) < 1e-12) {
      return { a: 0, b: 0, c: 0, ok: false, rSquared: 0, rmse: Infinity };
    }

    a = (Sxz * Syy - Sxy * Syz) / det;
    b = (-Sxz * Sxy + Sxx * Syz) / det;
    c = mz; // Plane passes through centroid by construction
  }

  // Calculate quality metrics
  const { rSquared, rmse } = calculateFitQuality(pts, a, b, c, mx, my, mz);

  return { a, b, c, ok: true, rSquared, rmse };
}

function calculateFitQuality(
  points: Array<{ x: number; y: number; z: number }>,
  a: number,
  b: number,
  c: number,
  mx: number,
  my: number,
  mz: number
): { rSquared: number; rmse: number } {
  const n = points.length;
  let ssRes = 0; // Sum of squares of residuals
  let ssTot = 0; // Total sum of squares

  for (const p of points) {
    const predicted = a * (p.x - mx) + b * (p.y - my) + c;
    const residual = p.z - predicted;

    ssRes += residual * residual;
    ssTot += (p.z - mz) * (p.z - mz);
  }

  const rSquared = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
  const rmse = Math.sqrt(ssRes / n);

  return { rSquared, rmse };
}

function assessFitQuality(
  rSquared: number,
  rmse: number,
  slopeMagnitude: number,
  samples: number
): 'excellent' | 'good' | 'fair' | 'poor' {
  // Quality assessment based on multiple factors
  if (samples < 10) return 'poor';
  if (rSquared > 0.95 && rmse < 2) return 'excellent';
  if (rSquared > 0.85 && rmse < 5) return 'good';
  if (rSquared > 0.7 && rmse < 10) return 'fair';
  return 'poor';
}

// ---------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------

/** Standard EPSG:3857 Spherical Mercator projection in meters */
const EARTH_RADIUS = 6378137; // WGS84 semi-major axis

function lngLatToMercatorMeters(lng: number, lat: number): [number, number] {
  const λ = degToRad(lng);
  const φ = degToRad(lat);

  // Clamp latitude to avoid projection singularities
  const clampedφ = Math.max(-85.0511 * Math.PI / 180, Math.min(85.0511 * Math.PI / 180, φ));

  return [
    EARTH_RADIUS * λ,
    EARTH_RADIUS * Math.log(Math.tan(Math.PI / 4 + clampedφ / 2)),
  ];
}

class WebMercatorProjector {
  private readonly z2: number;
  constructor(private readonly z: number) {
    this.z2 = 2 ** z;
  }

  pixelToLngLat(tx: number, ty: number, px: number, py: number, width: number): LngLat {
    const normX = (tx * width + px) / (this.z2 * width);
    const normY = (ty * width + py) / (this.z2 * width);

    const lng = normX * 360 - 180;
    const n = Math.PI - 2 * Math.PI * normY;
    const lat = (180 / Math.PI) * Math.atan(0.5 * (Math.exp(n) - Math.exp(-n)));
    return [lng, lat];
  }

  lngLatToPixel(lng: number, lat: number, tileX: number, tileY: number, width: number): [number, number] | null {
    // Convert lng/lat to normalized tile coordinates (0-1)
    const normX = (lng + 180) / 360;
    const normY = (1 - Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI / 180) / 2)) / Math.PI) / 2;

    // Convert to pixel coordinates within the specific tile
    const totalPixelsX = this.z2 * width;
    const totalPixelsY = this.z2 * width;

    const globalPixelX = normX * totalPixelsX;
    const globalPixelY = normY * totalPixelsY;

    // Check if this coordinate falls within the given tile
    const tileStartX = tileX * width;
    const tileStartY = tileY * width;
    const tileEndX = tileStartX + width;
    const tileEndY = tileStartY + width;

    if (globalPixelX >= tileStartX && globalPixelX < tileEndX &&
        globalPixelY >= tileStartY && globalPixelY < tileEndY) {
      return [globalPixelX - tileStartX, globalPixelY - tileStartY];
    }

    return null; // Point not in this tile
  }
}

const webMercatorProjectorCache = new Map<number, WebMercatorProjector>();

function getWebMercatorProjector(z: number) {
  let projector = webMercatorProjectorCache.get(z);
  if (!projector) {
    projector = new WebMercatorProjector(z);
    webMercatorProjectorCache.set(z, projector);
  }
  return projector;
}

function getElevation(tile: TerrainTile, px: number, py: number): number {
  if (px < 0 || py < 0 || px >= tile.width || py >= tile.height) {
    return NaN;
  }

  const idx = py * tile.width + px;
  if (tile.format === 'dem') {
    return (tile.data as Float32Array)[idx];
  } else {
    const base = idx * (tile.data.length === tile.width * tile.height * 4 ? 4 : 3);
    const r = tile.data[base];
    const g = tile.data[base + 1];
    const b = tile.data[base + 2];
    return -10000 + ((r * 256 * 256 + g * 256 + b) * 0.1);
  }
}

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

// Query elevation at a specific lng/lat coordinate using available tiles
export function queryElevationAtPoint(lng: number, lat: number, tiles: TerrainTile[]): number {
  for (const tile of tiles) {
    const proj = getWebMercatorProjector(tile.z);
    const pixelCoords = proj.lngLatToPixel(lng, lat, tile.x, tile.y, tile.width);

    if (pixelCoords) {
      const [px, py] = pixelCoords;
      const elevation = getElevation(tile, Math.floor(px), Math.floor(py));
      if (Number.isFinite(elevation)) {
        return elevation;
      }
    }
  }

  return NaN; // No elevation data found for this point
}

// Query maximum elevation along a line segment with sampling
export function queryMaxElevationAlongLine(
  startLng: number,
  startLat: number,
  endLng: number,
  endLat: number,
  tiles: TerrainTile[],
  sampleCount: number = 10
): number {
  let maxElevation = -Infinity;

  for (let i = 0; i <= sampleCount; i++) {
    const t = i / sampleCount;
    const lng = startLng + t * (endLng - startLng);
    const lat = startLat + t * (endLat - startLat);

    const elevation = queryElevationAtPoint(lng, lat, tiles);
    if (Number.isFinite(elevation)) {
      maxElevation = Math.max(maxElevation, elevation);
    }
  }

  return maxElevation === -Infinity ? NaN : maxElevation;
}

const degToRad = (d: number) => d * Math.PI / 180;
// For directional quantities (bearings/aspects) we want [0, 360)
const radToDegWrapped = (r: number) => (r * 180 / Math.PI + 360) % 360;
// For geographic coordinates we need signed degrees
const radToDegSigned = (r: number) => (r * 180 / Math.PI);
const normalizeLng = (deg: number) => ((deg + 180) % 360) - 180;

// Calculate destination point given start point, bearing, and distance
export function destination(start: [number, number], bearing: number, distance: number): [number, number] {
  const φ1 = degToRad(start[1]);
  const λ1 = degToRad(start[0]);
  const brng = degToRad(bearing);

  const R = 6371000; // Earth's radius in meters
  const δ = distance / R; // angular distance in radians

  const φ2 = Math.asin(
    Math.sin(φ1) * Math.cos(δ) +
    Math.cos(φ1) * Math.sin(δ) * Math.cos(brng)
  );

  const λ2 = λ1 + Math.atan2(
    Math.sin(brng) * Math.sin(δ) * Math.cos(φ1),
    Math.cos(δ) - Math.sin(φ1) * Math.sin(φ2)
  );
  // IMPORTANT: Return signed lon/lat. Longitude normalized to [-180,180), latitude kept in [-90,90].
  const lon = normalizeLng(radToDegSigned(λ2));
  const lat = radToDegSigned(φ2);
  return [lon, lat];
}

// Calculate bearing between two points
export function calculateBearing(from: [number, number], to: [number, number]): number {
  const φ1 = degToRad(from[1]);
  const φ2 = degToRad(to[1]);
  const Δλ = degToRad(to[0] - from[0]);

  const x = Math.sin(Δλ) * Math.cos(φ2);
  const y = Math.cos(φ1) * Math.sin(φ2) - Math.sin(φ1) * Math.cos(φ2) * Math.cos(Δλ);

  const bearing = Math.atan2(x, y);
  return radToDegWrapped(bearing);
}
