import type { LidarStripMeters, LidarWorkerOut, PolygonTileStats } from "../types";
import type { ExactLidarTileInput, ExactLidarTileOutput } from "./types";
import { tileMetersBounds } from "../mercator";
import { decodeTerrainRGBToElev } from "../terrain";
import { buildPolygonMasks, calculateDensityStats, convertElevationsToWGS84Ellipsoid } from "./shared";

const FIRST_RETURN_CHANNEL_TILT_DEG = 29;
const FIRST_RETURN_RANGE_TAPER_M = 20;

type PreparedStrip = LidarStripMeters & {
  index: number;
  x1s: number;
  y1s: number;
  x2s: number;
  y2s: number;
  dx: number;
  dy: number;
  dz: number;
  len: number;
  lenSq: number;
  minXs: number;
  maxXs: number;
  minYs: number;
  maxYs: number;
  halfWidthSq: number;
  ux: number;
  uy: number;
  perpX: number;
  perpY: number;
  frameRateHzResolved: number;
  mappingFovDegResolved: number;
  maxRangeMResolved: number;
  azimuthSectorCenterDegResolved: number;
  verticalAnglesDegResolved: number[];
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function normalizeVec3(x: number, y: number, z: number): [number, number, number] {
  const len = Math.hypot(x, y, z);
  if (!(len > 1e-9)) return [0, 0, -1];
  return [x / len, y / len, z / len];
}

function rotateAroundAxis(
  v: [number, number, number],
  axis: [number, number, number],
  angleDeg: number,
): [number, number, number] {
  if (!(Math.abs(angleDeg) > 1e-9)) return v;
  const [ax, ay, az] = normalizeVec3(axis[0], axis[1], axis[2]);
  const rad = (angleDeg * Math.PI) / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  const dot = v[0] * ax + v[1] * ay + v[2] * az;
  const crossX = ay * v[2] - az * v[1];
  const crossY = az * v[0] - ax * v[2];
  const crossZ = ax * v[1] - ay * v[0];
  return [
    v[0] * c + crossX * s + ax * dot * (1 - c),
    v[1] * c + crossY * s + ay * dot * (1 - c),
    v[2] * c + crossZ * s + az * dot * (1 - c),
  ];
}

function sampleDemBilinear(
  dem: Float32Array,
  size: number,
  minX: number,
  maxY: number,
  pixelSize: number,
  x: number,
  y: number,
): number {
  const colF = (x - minX) / pixelSize - 0.5;
  const rowF = (maxY - y) / pixelSize - 0.5;
  if (!(colF >= -0.5 && colF <= size - 0.5 && rowF >= -0.5 && rowF <= size - 0.5)) return Number.NaN;
  const c0 = clamp(Math.floor(colF), 0, size - 1);
  const r0 = clamp(Math.floor(rowF), 0, size - 1);
  const c1 = clamp(c0 + 1, 0, size - 1);
  const r1 = clamp(r0 + 1, 0, size - 1);
  const tx = clamp(colF - c0, 0, 1);
  const ty = clamp(rowF - r0, 0, 1);
  const i00 = r0 * size + c0;
  const i10 = r0 * size + c1;
  const i01 = r1 * size + c0;
  const i11 = r1 * size + c1;
  const z0 = dem[i00] * (1 - tx) + dem[i10] * tx;
  const z1 = dem[i01] * (1 - tx) + dem[i11] * tx;
  return z0 * (1 - ty) + z1 * ty;
}

function pointToPixelIndex(
  size: number,
  minX: number,
  maxY: number,
  pixelSize: number,
  x: number,
  y: number,
): { row: number; col: number; idx: number } | null {
  const col = Math.floor((x - minX) / pixelSize);
  const row = Math.floor((maxY - y) / pixelSize);
  if (row < 0 || row >= size || col < 0 || col >= size) return null;
  return { row, col, idx: row * size + col };
}

function intersectRayWithTileXYBounds(
  sx: number,
  sy: number,
  dx: number,
  dy: number,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number,
  maxRangeM: number,
): { sEnter: number; sExit: number } | null {
  let sEnter = 0;
  let sExit = maxRangeM;
  const EPS = 1e-9;

  if (Math.abs(dx) < EPS) {
    if (sx < minX || sx > maxX) return null;
  } else {
    let tx0 = (minX - sx) / dx;
    let tx1 = (maxX - sx) / dx;
    if (tx0 > tx1) [tx0, tx1] = [tx1, tx0];
    sEnter = Math.max(sEnter, tx0);
    sExit = Math.min(sExit, tx1);
  }

  if (Math.abs(dy) < EPS) {
    if (sy < minY || sy > maxY) return null;
  } else {
    let ty0 = (minY - sy) / dy;
    let ty1 = (maxY - sy) / dy;
    if (ty0 > ty1) [ty0, ty1] = [ty1, ty0];
    sEnter = Math.max(sEnter, ty0);
    sExit = Math.min(sExit, ty1);
  }

  if (!(sExit >= Math.max(0, sEnter))) return null;
  return { sEnter: Math.max(0, sEnter), sExit: Math.max(0, sExit) };
}

function clipSegmentParamToRect(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  minX: number,
  maxX: number,
  minY: number,
  maxY: number,
): { t0: number; t1: number } | null {
  const dx = x2 - x1;
  const dy = y2 - y1;
  let t0 = 0;
  let t1 = 1;
  const EPS = 1e-9;

  const clip = (p: number, q: number) => {
    if (Math.abs(p) < EPS) return q >= 0;
    const r = q / p;
    if (p < 0) {
      if (r > t1) return false;
      if (r > t0) t0 = r;
      return true;
    }
    if (r < t0) return false;
    if (r < t1) t1 = r;
    return true;
  };

  if (
    !clip(-dx, x1 - minX) ||
    !clip(dx, maxX - x1) ||
    !clip(-dy, y1 - minY) ||
    !clip(dy, maxY - y1)
  ) {
    return null;
  }

  if (!(t1 >= t0)) return null;
  return { t0, t1 };
}

function intersectBeamWithDem(
  dem: Float32Array,
  demSize: number,
  demMinX: number,
  demMaxX: number,
  demMinY: number,
  demMaxY: number,
  demMaxYOrigin: number,
  pixelSize: number,
  sx: number,
  sy: number,
  sz: number,
  dirX: number,
  dirY: number,
  dirZ: number,
  maxRangeM: number,
  stepM: number,
): { x: number; y: number; z: number } | null {
  if (!(dirZ < -1e-6)) return null;
  const boundsHit = intersectRayWithTileXYBounds(sx, sy, dirX, dirY, demMinX, demMaxX, demMinY, demMaxY, maxRangeM);
  if (!boundsHit) return null;

  const startS = boundsHit.sEnter;
  const endS = Math.min(boundsHit.sExit, maxRangeM);
  if (!(endS >= startS)) return null;

  const evalSignedHeight = (s: number): number => {
    const x = sx + dirX * s;
    const y = sy + dirY * s;
    const z = sz + dirZ * s;
    const terrainZ = sampleDemBilinear(dem, demSize, demMinX, demMaxYOrigin, pixelSize, x, y);
    if (!Number.isFinite(terrainZ)) return Number.NaN;
    return z - terrainZ;
  };

  let prevS = startS;
  let prevF = evalSignedHeight(prevS);
  if (!Number.isFinite(prevF)) return null;
  if (prevF <= 0) {
    if (startS > 1e-6) return null;
    return { x: sx + dirX * prevS, y: sy + dirY * prevS, z: sz + dirZ * prevS };
  }

  for (let s = startS + stepM; s <= endS + 1e-6; s += stepM) {
    const currS = Math.min(s, endS);
    const currF = evalSignedHeight(currS);
    if (!Number.isFinite(currF)) {
      prevS = currS;
      prevF = currF;
      continue;
    }
    if (currF <= 0) {
      let lo = prevS;
      let hi = currS;
      let flo = prevF;
      for (let iter = 0; iter < 8; iter++) {
        const mid = 0.5 * (lo + hi);
        const fmid = evalSignedHeight(mid);
        if (!Number.isFinite(fmid)) break;
        if (fmid > 0) {
          lo = mid;
          flo = fmid;
        } else {
          hi = mid;
        }
      }
      const hitS = flo > 0 ? hi : lo;
      return {
        x: sx + dirX * hitS,
        y: sy + dirY * hitS,
        z: sz + dirZ * hitS,
      };
    }
    prevS = currS;
    prevF = currF;
  }

  return null;
}

function chooseAzimuthSampleCount(mappingFovDeg: number, swathWidthM: number, pixelSizeM: number): number {
  const targetByWidth = Math.max(7, Math.round(swathWidthM / Math.max(1, pixelSizeM * 2)));
  const targetByAngle = Math.max(7, Math.round(mappingFovDeg / 6));
  const count = clamp(Math.max(targetByWidth, targetByAngle), 7, 25);
  return count % 2 === 0 ? count + 1 : count;
}

function buildBeamDirection(
  alongAxis: [number, number, number],
  crossAxis: [number, number, number],
  azimuthDeg: number,
  channelDeg: number,
  boresightYawDeg: number,
  boresightPitchDeg: number,
  boresightRollDeg: number,
): [number, number, number] {
  let dir: [number, number, number] = [0, 0, -1];
  dir = rotateAroundAxis(dir, alongAxis, azimuthDeg);
  dir = rotateAroundAxis(dir, crossAxis, channelDeg);
  dir = rotateAroundAxis(dir, [0, 0, -1], boresightYawDeg);
  dir = rotateAroundAxis(dir, crossAxis, boresightPitchDeg);
  dir = rotateAroundAxis(dir, alongAxis, boresightRollDeg);
  return normalizeVec3(dir[0], dir[1], dir[2]);
}

function emptyLidarResult(z: number, x: number, y: number, size: number): LidarWorkerOut {
  return {
    z,
    x,
    y,
    size,
    overlap: new Uint16Array(size * size),
    maxOverlap: 0,
    minGsd: Number.POSITIVE_INFINITY,
    gsdMin: new Float32Array(size * size).fill(Number.POSITIVE_INFINITY),
    density: new Float32Array(size * size),
    maxDensity: 0,
    densityStats: { min: 0, max: 0, mean: 0, count: 0, totalAreaM2: 0, histogram: [] },
    perPolygon: [],
  };
}

export function evaluateLidarTileExact(input: ExactLidarTileInput): ExactLidarTileOutput {
  const { tile, demTile, polygons, strips, options } = input;
  const { z, x, y, size, data } = tile;

  const tileBounds = tileMetersBounds(z, x, y);
  const tileWidthM = tileBounds.maxX - tileBounds.minX;
  const demPadTiles = Math.max(0, demTile?.padTiles ?? 0);
  const demSize = demTile?.size ?? size;
  const demBounds = {
    minX: tileBounds.minX - demPadTiles * tileWidthM,
    minY: tileBounds.minY - demPadTiles * tileWidthM,
    maxX: tileBounds.maxX + demPadTiles * tileWidthM,
    maxY: tileBounds.maxY + demPadTiles * tileWidthM,
  };
  const clipM = Math.max(0, options?.clipInnerBufferM ?? 0);
  const pixM = (tileBounds.maxX - tileBounds.minX) / size;
  const radiusPx = clipM > 0 ? Math.max(1, Math.ceil(clipM / pixM)) : 0;
  const { tx, masks: polyMasks, unionMask: polyMask, ids: polyIds } = buildPolygonMasks(polygons, z, x, y, size, radiusPx);

  let polyPixelCount = 0;
  for (let i = 0; i < polyMask.length; i++) polyPixelCount += polyMask[i];
  if (polyPixelCount === 0 || !Array.isArray(strips) || strips.length === 0) {
    return emptyLidarResult(z, x, y, size);
  }

  const demData = demTile?.data ?? data;
  const elevEGM96 = decodeTerrainRGBToElev(demData, demSize);
  const elev = convertElevationsToWGS84Ellipsoid(elevEGM96, demSize, demBounds);

  const radius = 6378137;
  const pixSizeRaw = (tx.maxX - tx.minX) / size;
  const ywRowRaw = new Float64Array(size);
  for (let r = 0; r < size; r++) ywRowRaw[r] = tx.maxY - (r + 0.5) * pixSizeRaw;

  const cosLatPerRow = new Float64Array(size);
  for (let r = 0; r < size; r++) {
    const latRad = Math.atan(Math.sinh(ywRowRaw[r] / radius));
    cosLatPerRow[r] = Math.cos(latRad);
  }

  const latCenter = Math.atan(Math.sinh(((tx.minY + tx.maxY) * 0.5) / radius));
  const scale = Math.cos(latCenter);
  const minX = tx.minX * scale;
  const maxX = tx.maxX * scale;
  const minY = tx.minY * scale;
  const maxY = tx.maxY * scale;
  const demMinX = demBounds.minX * scale;
  const demMaxX = demBounds.maxX * scale;
  const demMinY = demBounds.minY * scale;
  const demMaxY = demBounds.maxY * scale;
  const pixSize = pixSizeRaw * scale;
  const xwCol = new Float64Array(size);
  const ywRow = new Float64Array(size);
  for (let c = 0; c < size; c++) xwCol[c] = minX + (c + 0.5) * pixSize;
  for (let r = 0; r < size; r++) ywRow[r] = maxY - (r + 0.5) * pixSize;

  const activeIdxs = new Uint32Array(polyPixelCount);
  for (let i = 0, w = 0; i < polyMask.length; i++) {
    if (!polyMask[i]) continue;
    activeIdxs[w++] = i;
  }

  const prepared: PreparedStrip[] = strips
    .map((strip, index) => {
      const x1s = strip.x1 * scale;
      const y1s = strip.y1 * scale;
      const x2s = strip.x2 * scale;
      const y2s = strip.y2 * scale;
      const dx = x2s - x1s;
      const dy = y2s - y1s;
      const dz = (strip.z2 ?? strip.z1 ?? 0) - (strip.z1 ?? strip.z2 ?? 0);
      const len = Math.hypot(dx, dy);
      const ux = len > 1e-6 ? dx / len : 0;
      const uy = len > 1e-6 ? dy / len : 0;
      const halfWidth = Math.max(0, strip.halfWidthM);
      const frameRateHzResolved = Math.max(1, Number.isFinite(strip.frameRateHz) ? strip.frameRateHz! : 10);
      const mappingFovDegResolved = clamp(
        Number.isFinite(strip.mappingFovDeg) ? strip.mappingFovDeg! : 90,
        1,
        180,
      );
      const maxRangeMResolved = Math.max(1, Number.isFinite(strip.maxRangeM) ? strip.maxRangeM! : Number.POSITIVE_INFINITY);
      const azimuthSectorCenterDegResolved = Number.isFinite(strip.azimuthSectorCenterDeg) ? strip.azimuthSectorCenterDeg! : 0;
      const verticalAnglesDegResolved = Array.isArray(strip.verticalAnglesDeg) && strip.verticalAnglesDeg.length > 0
        ? strip.verticalAnglesDeg.filter((value): value is number => Number.isFinite(value))
        : [0];
      return {
        ...strip,
        index,
        x1s,
        y1s,
        x2s,
        y2s,
        dx,
        dy,
        dz,
        len,
        lenSq: dx * dx + dy * dy,
        minXs: Math.min(x1s, x2s) - halfWidth,
        maxXs: Math.max(x1s, x2s) + halfWidth,
        minYs: Math.min(y1s, y2s) - halfWidth,
        maxYs: Math.max(y1s, y2s) + halfWidth,
        halfWidthSq: halfWidth * halfWidth,
        ux,
        uy,
        perpX: -uy,
        perpY: ux,
        frameRateHzResolved,
        mappingFovDegResolved,
        maxRangeMResolved,
        azimuthSectorCenterDegResolved,
        verticalAnglesDegResolved,
      };
    })
    .filter((strip) => strip.densityPerPass > 0 && strip.halfWidthM > 0);

  if (prepared.length === 0) {
    return emptyLidarResult(z, x, y, size);
  }

  const overlap = new Uint16Array(size * size);
  const gsdMin = new Float32Array(size * size).fill(Number.POSITIVE_INFINITY);
  const density = new Float32Array(size * size);
  const hitLinesPerPolygon: Array<Set<number>> = polyIds.map(() => new Set<number>());
  const lastPassSeen = new Int32Array(size * size).fill(-1);
  const polyIndexById = new Map<string, number>();
  for (let i = 0; i < polyIds.length; i++) polyIndexById.set(polyIds[i], i);
  const pixelAreaEquator = pixSize * pixSize;
  const pixelAreaByRow = new Float64Array(size);
  for (let row = 0; row < size; row++) {
    const cosPhi = cosLatPerRow[row];
    pixelAreaByRow[row] = pixelAreaEquator * cosPhi * cosPhi;
  }

  let maxOverlap = 0;
  let maxDensity = 0;
  const stepM = Math.max(1.5, pixSize * 0.75);

  for (let i = 0; i < prepared.length; i++) {
    const strip = prepared[i];
    const polygonId = strip.polygonId ?? null;
    if (!polygonId) continue;
    const polyIndex = polyIndexById.get(polygonId);
    if (polyIndex === undefined) continue;
    const polyMask = polyMasks[polyIndex];
    const speedMps = strip.speedMps ?? 0;
    const effectivePointRate = strip.effectivePointRate ?? 0;
    if (!(speedMps > 0) || !(effectivePointRate > 0) || !(strip.len > 1e-6)) continue;

    const alongSpacingM = Math.max(speedMps / strip.frameRateHzResolved, pixSize * 0.75);
    const rawChannelAngles = strip.verticalAnglesDegResolved;
    const comparisonMode = strip.comparisonMode ?? "first-return";
    const channelAngles = comparisonMode === "first-return"
      ? rawChannelAngles.filter((angle) => angle <= 0)
      : rawChannelAngles;
    if (channelAngles.length === 0) continue;
    const swathWidthM = Math.max(2, strip.halfWidthM * 2);
    const azimuthSampleCount = chooseAzimuthSampleCount(strip.mappingFovDegResolved, swathWidthM, pixSize);

    let segmentStartT = 0;
    let segmentEndT = 1;
    if (Number.isFinite(strip.maxRangeMResolved)) {
      const reachPadM = strip.maxRangeMResolved;
      const clipped = clipSegmentParamToRect(
        strip.x1s,
        strip.y1s,
        strip.x2s,
        strip.y2s,
        minX - reachPadM,
        maxX + reachPadM,
        minY - reachPadM,
        maxY + reachPadM,
      );
      if (!clipped) continue;
      segmentStartT = clipped.t0;
      segmentEndT = clipped.t1;
    }
    const activeLen = strip.len * (segmentEndT - segmentStartT);
    if (!(activeLen > 1e-6)) continue;

    const alongAxis: [number, number, number] = [strip.ux, strip.uy, 0];
    const crossAxis: [number, number, number] = [-strip.uy, strip.ux, 0];
    const z1 = strip.z1 ?? strip.z2 ?? 0;
    const z2 = strip.z2 ?? strip.z1 ?? z1;
    const boresightYawDeg = Number.isFinite(strip.boresightYawDeg) ? strip.boresightYawDeg! : 0;
    const boresightPitchDeg = Number.isFinite(strip.boresightPitchDeg) ? strip.boresightPitchDeg! : 0;
    const boresightRollDeg = Number.isFinite(strip.boresightRollDeg) ? strip.boresightRollDeg! : 0;
    const passIndex = strip.passIndex ?? strip.index;
    const downwardRangeFactor = new Float32Array(channelAngles.length);
    const useChannelForFirstReturn = new Uint8Array(channelAngles.length);
    for (let channelIndex = 0; channelIndex < channelAngles.length; channelIndex++) {
      const downwardLookDeg = FIRST_RETURN_CHANNEL_TILT_DEG - channelAngles[channelIndex];
      if (comparisonMode !== "first-return") {
        useChannelForFirstReturn[channelIndex] = 1;
        continue;
      }
      if (!(downwardLookDeg > 0)) continue;
      const downRad = (downwardLookDeg * Math.PI) / 180;
      const factor = 1 / Math.sin(downRad);
      if (!Number.isFinite(factor) || !(factor > 0)) continue;
      downwardRangeFactor[channelIndex] = factor;
      useChannelForFirstReturn[channelIndex] = 1;
    }
    const beamDirections = new Float32Array(azimuthSampleCount * channelAngles.length * 3);
    for (let azimuthIndex = 0; azimuthIndex < azimuthSampleCount; azimuthIndex++) {
      const azimuthFraction = azimuthSampleCount === 1 ? 0.5 : azimuthIndex / (azimuthSampleCount - 1);
      const azimuthDeg = strip.azimuthSectorCenterDegResolved + (azimuthFraction - 0.5) * strip.mappingFovDegResolved;
      for (let channelIndex = 0; channelIndex < channelAngles.length; channelIndex++) {
        const dir = buildBeamDirection(
          alongAxis,
          crossAxis,
          azimuthDeg,
          channelAngles[channelIndex],
          boresightYawDeg,
          boresightPitchDeg,
          boresightRollDeg,
        );
        const dirBase = (azimuthIndex * channelAngles.length + channelIndex) * 3;
        beamDirections[dirBase] = dir[0];
        beamDirections[dirBase + 1] = dir[1];
        beamDirections[dirBase + 2] = dir[2];
      }
    }

    const sampleCount = Math.max(1, Math.ceil(activeLen / alongSpacingM));
    const representedDistancePerSampleM = activeLen / sampleCount;
    const pointsPerSampleAdjusted = effectivePointRate * (representedDistancePerSampleM / speedMps);
    const beamWeightAdjusted = pointsPerSampleAdjusted / (rawChannelAngles.length * azimuthSampleCount);
    if (!(beamWeightAdjusted > 0)) continue;

    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
      const tInActive = sampleCount === 1 ? 0.5 : (sampleIndex + 0.5) / sampleCount;
      const t = segmentStartT + (segmentEndT - segmentStartT) * tInActive;
      const sx = strip.x1s + strip.dx * t;
      const sy = strip.y1s + strip.dy * t;
      const sz = z1 + (z2 - z1) * t;
      let localAltitudeAGL = Number.NaN;
      if (comparisonMode === "first-return") {
        const terrainUnderSensor = sampleDemBilinear(elev, demSize, demMinX, demMaxY, pixSize, sx, sy);
        localAltitudeAGL = Number.isFinite(terrainUnderSensor)
          ? (sz - terrainUnderSensor)
          : (Number.isFinite(strip.plannedAltitudeAGL) ? strip.plannedAltitudeAGL! : Number.NaN);
      }

      for (let azimuthIndex = 0; azimuthIndex < azimuthSampleCount; azimuthIndex++) {
        for (let channelIndex = 0; channelIndex < channelAngles.length; channelIndex++) {
          let channelWeight = 1;
          if (comparisonMode === "first-return") {
            if (!useChannelForFirstReturn[channelIndex]) continue;
            if (Number.isFinite(localAltitudeAGL) && localAltitudeAGL > 0) {
              const expectedFlatRange = localAltitudeAGL * downwardRangeFactor[channelIndex];
              if (!Number.isFinite(expectedFlatRange)) continue;
              if (!(expectedFlatRange < strip.maxRangeMResolved + FIRST_RETURN_RANGE_TAPER_M)) continue;
              if (expectedFlatRange > strip.maxRangeMResolved) {
                channelWeight = clamp(
                  1 - (expectedFlatRange - strip.maxRangeMResolved) / FIRST_RETURN_RANGE_TAPER_M,
                  0,
                  1,
                );
              }
            }
          }

          const dirBase = (azimuthIndex * channelAngles.length + channelIndex) * 3;
          const dirX = beamDirections[dirBase];
          const dirY = beamDirections[dirBase + 1];
          const dirZ = beamDirections[dirBase + 2];
          if (!(dirZ < -1e-4)) continue;

          const hit = intersectBeamWithDem(
            elev,
            demSize,
            demMinX,
            demMaxX,
            demMinY,
            demMaxY,
            demMaxY,
            pixSize,
            sx,
            sy,
            sz,
            dirX,
            dirY,
            dirZ,
            strip.maxRangeMResolved,
            stepM,
          );
          if (!hit) continue;
          const hitPixel = pointToPixelIndex(size, minX, maxY, pixSize, hit.x, hit.y);
          if (!hitPixel) continue;
          if (!polyMask[hitPixel.idx]) continue;

          const areaM2 = pixelAreaByRow[hitPixel.row];
          const densityContribution = areaM2 > 0 ? ((beamWeightAdjusted * channelWeight) / areaM2) : 0;
          if (!(densityContribution > 0)) continue;
          density[hitPixel.idx] += densityContribution;
          if (density[hitPixel.idx] > maxDensity) maxDensity = density[hitPixel.idx];

          if (lastPassSeen[hitPixel.idx] !== passIndex) {
            lastPassSeen[hitPixel.idx] = passIndex;
            overlap[hitPixel.idx] += 1;
            if (overlap[hitPixel.idx] > maxOverlap) maxOverlap = overlap[hitPixel.idx];
          }
          hitLinesPerPolygon[polyIndex].add(passIndex);
        }
      }
    }
  }

  const perPolygon: PolygonTileStats[] = [];
  for (let p = 0; p < polyMasks.length; p++) {
    const mask = polyMasks[p];
    let count = 0;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] && density[i] > 0 && Number.isFinite(density[i])) count++;
    }
    if (count === 0) {
      perPolygon.push({
        polygonId: polyIds[p],
        activePixelCount: 0,
        densityStats: { min: 0, max: 0, mean: 0, count: 0, totalAreaM2: 0, histogram: [] },
        hitLineIds: new Uint32Array(0),
      });
      continue;
    }
    const activePolygonPixels = new Uint32Array(count);
    for (let i = 0, w = 0; i < mask.length; i++) {
      if (mask[i] && density[i] > 0 && Number.isFinite(density[i])) activePolygonPixels[w++] = i;
    }
    let polygonPixelCount = 0;
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) polygonPixelCount++;
    }
    const polygonPixels = new Uint32Array(polygonPixelCount);
    for (let i = 0, w = 0; i < mask.length; i++) {
      if (mask[i]) polygonPixels[w++] = i;
    }
    const stats = calculateDensityStats(density, polygonPixels, pixelAreaEquator, size, cosLatPerRow, true);
    const hitSet = hitLinesPerPolygon[p];
    const hitLineIds = new Uint32Array(hitSet.size);
    let w = 0;
    hitSet.forEach((id) => { hitLineIds[w++] = id; });
    perPolygon.push({
      polygonId: polyIds[p],
      activePixelCount: activePolygonPixels.length,
      densityStats: stats,
      hitLineIds,
    });
  }

  const densityStats = calculateDensityStats(density, activeIdxs, pixelAreaEquator, size, cosLatPerRow, true);
  return {
    z,
    x,
    y,
    size,
    overlap,
    maxOverlap,
    minGsd: Number.POSITIVE_INFINITY,
    gsdMin,
    density,
    maxDensity,
    densityStats,
    perPolygon,
  };
}
