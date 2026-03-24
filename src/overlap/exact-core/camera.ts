import type { CameraModel, PolygonTileStats, PoseMeters, WorkerOut } from "../types";
import type { ExactCameraTileInput, ExactCameraTileOutput } from "./types";
import { tileMetersBounds } from "../mercator";
import { decodeTerrainRGBToElev } from "../terrain";
import { camRayToPixel, normalFromDEM, rotMat } from "../math3d";
import { buildPolygonMasks, calculateGSDStatsFast, convertElevationsToWGS84Ellipsoid } from "./shared";

type PreparedPose = PoseMeters & {
  R: Float64Array;
  RT: Float64Array;
  radius: number;
  radiusSq: number;
  camIndex: number;
  bx0: number;
  bx1: number;
  bx2: number;
  by0: number;
  by1: number;
  by2: number;
};

function emptyCameraResult(z: number, x: number, y: number, size: number): WorkerOut {
  return {
    z,
    x,
    y,
    size,
    overlap: new Uint16Array(size * size),
    gsdMin: new Float32Array(size * size).fill(Number.POSITIVE_INFINITY),
    maxOverlap: 0,
    minGsd: Number.POSITIVE_INFINITY,
    perPolygon: [],
  };
}

export function evaluateCameraTileExact(input: ExactCameraTileInput): ExactCameraTileOutput {
  const { tile, polygons, poses, camera, cameras, poseCameraIndices, options } = input;
  const { z, x, y, size, data } = tile;

  const multi = Array.isArray(cameras) && cameras.length > 0 && poseCameraIndices instanceof Uint16Array && poseCameraIndices.length === poses.length;
  const camModels: CameraModel[] = multi ? cameras : (camera ? [camera] : []);
  const poseCamIdx: Uint16Array = multi ? poseCameraIndices! : new Uint16Array(poses.length);
  if (camModels.length === 0) {
    return emptyCameraResult(z, x, y, size);
  }

  const elevEGM96 = decodeTerrainRGBToElev(data, size);
  const tileBounds = tileMetersBounds(z, x, y);
  const elev = convertElevationsToWGS84Ellipsoid(elevEGM96, size, tileBounds);

  const clipM = Math.max(0, options?.clipInnerBufferM ?? 0);
  const pixM = (tileBounds.maxX - tileBounds.minX) / size;
  const radiusPx = clipM > 0 ? Math.max(1, Math.ceil(clipM / pixM)) : 0;

  const { tx, masks: polyMasks, unionMask: polyMask, ids: polyIds } = buildPolygonMasks(polygons, z, x, y, size, radiusPx);

  let polyPixelCount = 0;
  for (let i = 0; i < polyMask.length; i++) polyPixelCount += polyMask[i];
  if (polyPixelCount === 0) {
    return emptyCameraResult(z, x, y, size);
  }

  const radius = 6378137;
  const pixSizeRaw = (tx.maxX - tx.minX) / size;
  const minXRaw = tx.minX;
  const maxYRaw = tx.maxY;
  const ywRowRaw = new Float64Array(size);
  for (let r = 0; r < size; r++) ywRowRaw[r] = maxYRaw - (r + 0.5) * pixSizeRaw;
  const cosLatPerRow = new Float64Array(size);
  for (let r = 0; r < size; r++) {
    const latRad = Math.atan(Math.sinh(ywRowRaw[r] / radius));
    cosLatPerRow[r] = Math.cos(latRad);
  }

  const latCenter = Math.atan(Math.sinh(((tx.minY + tx.maxY) * 0.5) / radius));
  const scale = Math.cos(latCenter);
  const minX = minXRaw * scale;
  const maxX = tx.maxX * scale;
  const minY = tx.minY * scale;
  const maxY = tx.maxY * scale;
  const pixSize = pixSizeRaw * scale;
  const xwCol = new Float64Array(size);
  const ywRow = new Float64Array(size);
  for (let c = 0; c < size; c++) xwCol[c] = minX + (c + 0.5) * pixSize;
  for (let r = 0; r < size; r++) ywRow[r] = maxY - (r + 0.5) * pixSize;

  const normals = new Float32Array(size * size * 3);
  const activeIdxs = new Uint32Array(polyPixelCount);
  {
    let w = 0;
    for (let idx = 0; idx < elev.length; idx++) {
      if (!polyMask[idx]) continue;
      const row = (idx / size) | 0;
      const col = idx - row * size;
      const pixSizeGround = pixSizeRaw * cosLatPerRow[row];
      const n = normalFromDEM(elev, size, row, col, pixSizeGround);
      const base = idx * 3;
      normals[base] = n[0];
      normals[base + 1] = n[1];
      normals[base + 2] = n[2];
      activeIdxs[w++] = idx;
    }
  }

  const camDiagTan: number[] = new Array(camModels.length);
  for (let ci = 0; ci < camModels.length; ci++) {
    const camModel = camModels[ci];
    const sensorW = camModel.w_px * camModel.sx_m;
    const sensorH = camModel.h_px * camModel.sy_m;
    const diagFovHalf = Math.atan(0.5 * Math.hypot(sensorW, sensorH) / camModel.f_m);
    camDiagTan[ci] = Math.tan(diagFovHalf);
  }

  const prepared: PreparedPose[] = new Array(poses.length);
  for (let i = 0; i < poses.length; i++) {
    const pose = poses[i];
    const idxVal = poseCamIdx[i];
    const camIdx = idxVal < camModels.length ? idxVal : 0;
    const rotation = rotMat(pose.omega_deg, pose.phi_deg, pose.kappa_deg);
    const RT = new Float64Array([
      rotation[0], rotation[3], rotation[6],
      rotation[1], rotation[4], rotation[7],
      rotation[2], rotation[5], rotation[8],
    ]);
    const colP = Math.min(size - 1, Math.max(0, Math.floor((pose.x - minXRaw) / pixSizeRaw)));
    const rowP = Math.min(size - 1, Math.max(0, Math.floor((maxYRaw - pose.y) / pixSizeRaw)));
    const zLocal = elev[rowP * size + colP];
    const H = Math.max(1, pose.z - zLocal);
    const diagTan = camDiagTan[camIdx];
    const coverageRadius = H * diagTan * 1.25;
    const xs = pose.x * scale;
    const ys = pose.y * scale;
    prepared[i] = {
      ...pose,
      x: xs,
      y: ys,
      R: new Float64Array([
        rotation[0], rotation[1], rotation[2],
        rotation[3], rotation[4], rotation[5],
        rotation[6], rotation[7], rotation[8],
      ]),
      RT,
      radius: coverageRadius,
      radiusSq: coverageRadius * coverageRadius,
      camIndex: camIdx,
      bx0: rotation[0],
      bx1: rotation[3],
      bx2: rotation[6],
      by0: rotation[1],
      by1: rotation[4],
      by2: rotation[7],
    };
  }

  const candidateIdxs: number[] = [];
  for (let i = 0; i < prepared.length; i++) {
    const pose = prepared[i];
    const dx = pose.x < minX ? (minX - pose.x) : pose.x > maxX ? (pose.x - maxX) : 0;
    const dy = pose.y < minY ? (minY - pose.y) : pose.y > maxY ? (pose.y - maxY) : 0;
    if (dx * dx + dy * dy <= pose.radiusSq) candidateIdxs.push(i);
  }
  const useIdxs = polyPixelCount > 0 && candidateIdxs.length === 0 ? prepared.map((_, i) => i) : candidateIdxs;

  const gridSize = Math.max(2, Math.min(32, options?.gridSize ?? 8));
  const cellW = (maxX - minX) / gridSize;
  const cellH = (maxY - minY) / gridSize;
  const grid: number[][] = new Array(gridSize * gridSize);
  for (let i = 0; i < grid.length; i++) grid[i] = [];
  for (let j = 0; j < useIdxs.length; j++) {
    const i = useIdxs[j];
    const pose = prepared[i];
    const minXb = Math.max(minX, pose.x - pose.radius);
    const maxXb = Math.min(maxX, pose.x + pose.radius);
    const minYb = Math.max(minY, pose.y - pose.radius);
    const maxYb = Math.min(maxY, pose.y + pose.radius);
    if (minXb > maxXb || minYb > maxYb) continue;
    const x0 = Math.max(0, Math.floor((minXb - minX) / cellW));
    const x1 = Math.min(gridSize - 1, Math.floor((maxXb - minX) / cellW));
    const y0 = Math.max(0, Math.floor((maxY - maxYb) / cellH));
    const y1 = Math.min(gridSize - 1, Math.floor((maxY - minYb) / cellH));
    for (let gy = y0; gy <= y1; gy++) {
      const rowBase = gy * gridSize;
      for (let gx = x0; gx <= x1; gx++) grid[rowBase + gx].push(j);
    }
  }

  const col2cellX = new Uint16Array(size);
  const row2cellY = new Uint16Array(size);
  for (let c = 0; c < size; c++) col2cellX[c] = Math.min(gridSize - 1, (c * gridSize / size) | 0);
  for (let r = 0; r < size; r++) row2cellY[r] = Math.min(gridSize - 1, (r * gridSize / size) | 0);

  const overlap = new Uint16Array(size * size);
  const gsdMin = new Float32Array(size * size);
  gsdMin.fill(Number.POSITIVE_INFINITY);
  const maxOverlapNeeded = Number.isFinite(options?.maxOverlapNeeded) ? options!.maxOverlapNeeded! : Infinity;
  const poseHitsPerPoly: Array<Set<number>> = polyIds.map(() => new Set<number>());
  const cosIncMin = 1e-3;

  for (let t = 0; t < activeIdxs.length; t++) {
    const idx = activeIdxs[t];
    const row = (idx / size) | 0;
    const col = idx - row * size;
    const xw = xwCol[col];
    const yw = ywRow[row];
    const zw = elev[idx];
    const polysHere: number[] = [];
    for (let p = 0; p < polyMasks.length; p++) {
      if (polyMasks[p][idx]) polysHere.push(p);
    }
    if (polysHere.length === 0) continue;
    const nb = idx * 3;
    const nx = normals[nb];
    const ny = normals[nb + 1];
    const nz = normals[nb + 2];

    let localOverlap = 0;
    let localMinG = Number.POSITIVE_INFINITY;
    const cellIdx = (row2cellY[row] * gridSize + col2cellX[col]) | 0;
    const cellList = grid[cellIdx];
    if (cellList.length === 0) continue;

    for (let u = 0; u < cellList.length; u++) {
      const k = cellList[u];
      const poseIdx = useIdxs[k];
      const pose = prepared[poseIdx];
      const dx = xw - pose.x;
      const dy = yw - pose.y;
      if (dx * dx + dy * dy > pose.radiusSq) continue;
      const camIdx = pose.camIndex;
      const camModel = camModels[camIdx];
      const camHit = camRayToPixel(camModel, pose.RT, pose.x, pose.y, pose.z, xw, yw, zw);
      if (!camHit) continue;

      const vz = zw - pose.z;
      const L = Math.hypot(dx, dy, vz);
      if (!(L > 0)) continue;
      const invL = 1 / L;
      const rx = dx * invL;
      const ry = dy * invL;
      const rz = vz * invL;
      const cosInc = -(nx * rx + ny * ry + nz * rz);
      if (cosInc <= cosIncMin) continue;

      const rcx = pose.RT[0] * rx + pose.RT[1] * ry + pose.RT[2] * rz;
      const rcy = pose.RT[3] * rx + pose.RT[4] * ry + pose.RT[5] * rz;
      const rcz = pose.RT[6] * rx + pose.RT[7] * ry + pose.RT[8] * rz;
      if (rcz >= 0) continue;

      const f = camModel.f_m;
      const u_m = f * (rcx / rcz);
      const v_m = f * (rcy / rcz);

      const a0 = pose.R[0] * u_m + pose.R[1] * v_m + pose.R[2] * f;
      const a1 = pose.R[3] * u_m + pose.R[4] * v_m + pose.R[5] * f;
      const a2 = pose.R[6] * u_m + pose.R[7] * v_m + pose.R[8] * f;
      const denom = nx * a0 + ny * a1 + nz * a2;
      if (Math.abs(denom) < 1e-12) continue;

      const Hn = nx * (xw - pose.x) + ny * (yw - pose.y) + nz * (zw - pose.z);
      const invDen2 = 1 / (denom * denom);

      const nbx = nx * pose.bx0 + ny * pose.bx1 + nz * pose.bx2;
      const Jux = (denom * pose.bx0 - nbx * a0) * Hn * invDen2;
      const Juy = (denom * pose.bx1 - nbx * a1) * Hn * invDen2;
      const Juz = (denom * pose.bx2 - nbx * a2) * Hn * invDen2;

      const nby = nx * pose.by0 + ny * pose.by1 + nz * pose.by2;
      const Jvx = (denom * pose.by0 - nby * a0) * Hn * invDen2;
      const Jvy = (denom * pose.by1 - nby * a1) * Hn * invDen2;
      const Jvz = (denom * pose.by2 - nby * a2) * Hn * invDen2;

      const gsdx = Math.hypot(Jux, Juy, Juz) * camModel.sx_m;
      const gsdy = Math.hypot(Jvx, Jvy, Jvz) * camModel.sy_m;
      const gsd = Math.sqrt(gsdx * gsdy);

      for (let pi = 0; pi < polysHere.length; pi++) poseHitsPerPoly[polysHere[pi]].add(poseIdx);
      if (localOverlap < maxOverlapNeeded) localOverlap++;
      if (gsd < localMinG) localMinG = gsd;
    }

    if (localOverlap > 0) {
      overlap[idx] = localOverlap;
      gsdMin[idx] = localMinG;
    }
  }

  const minOverlapForGsd = Number.isFinite(options?.minOverlapForGsd)
    ? Math.max(1, Math.round(options!.minOverlapForGsd!))
    : 4;
  for (let t = 0; t < activeIdxs.length; t++) {
    const idx = activeIdxs[t];
    if (overlap[idx] > 0 && overlap[idx] < minOverlapForGsd) {
      overlap[idx] = 0;
      gsdMin[idx] = Number.POSITIVE_INFINITY;
    }
  }

  let maxOverlap = 0;
  let minGsd = Number.POSITIVE_INFINITY;
  for (let t = 0; t < activeIdxs.length; t++) {
    const idx = activeIdxs[t];
    const ov = overlap[idx];
    if (ov > maxOverlap) maxOverlap = ov;
    const g = gsdMin[idx];
    if (g > 0 && Number.isFinite(g) && g < minGsd) minGsd = g;
  }

  const perPolygon: PolygonTileStats[] = [];
  for (let p = 0; p < polyMasks.length; p++) {
    let cnt = 0;
    const mask = polyMasks[p];
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] && Number.isFinite(gsdMin[i]) && gsdMin[i] > 0) cnt++;
    }
    if (cnt === 0) {
      perPolygon.push({
        polygonId: polyIds[p],
        activePixelCount: 0,
        gsdStats: { min: 0, max: 0, mean: 0, count: 0, histogram: [] },
        hitPoseIds: new Uint32Array(0),
      });
      continue;
    }
    const activePolygonPixels = new Uint32Array(cnt);
    for (let i = 0, w = 0; i < mask.length; i++) {
      if (mask[i] && Number.isFinite(gsdMin[i]) && gsdMin[i] > 0) activePolygonPixels[w++] = i;
    }
    const stats = calculateGSDStatsFast(gsdMin, activePolygonPixels, pixSize * pixSize, size, cosLatPerRow);
    const hits = poseHitsPerPoly[p];
    const hitPoseIds = new Uint32Array(hits.size);
    let w = 0;
    hits.forEach((id) => { hitPoseIds[w++] = id; });
    perPolygon.push({
      polygonId: polyIds[p],
      activePixelCount: stats.count,
      gsdStats: stats,
      hitPoseIds,
    });
  }

  const gsdStats = calculateGSDStatsFast(gsdMin, activeIdxs, pixSize * pixSize, size, cosLatPerRow);
  return {
    z,
    x,
    y,
    size,
    overlap,
    gsdMin,
    maxOverlap,
    minGsd,
    gsdStats,
    perPolygon,
  };
}
