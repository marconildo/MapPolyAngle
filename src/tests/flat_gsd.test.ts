/**
 * Flat-ground GSD test (no GUI)
 *
 * Scenario:
 * - Single square area of 1 km x 1 km (1,000,000 m^2), perfectly flat
 * - Flight at 100 m AGL
 * - 70% front overlap, 70% side overlap
 * - Direction does not matter for flat terrain
 *
 * This script computes per‑camera flight spacing and theoretical GSD, and emits
 * a simple GSDStats object (no assertions yet). It does not fetch terrain or use workers.
 *
 * Run (one option):
 *   npx ts-node src/tests/flat_gsd.test.ts
 * or configure your preferred TS runner.
 */

import {
  SONY_RX1R2,
  SONY_RX1R3,
  SONY_A6100_20MM,
  DJI_ZENMUSE_P1_24MM,
  ILX_LR1_INSPECT_85MM,
  MAP61_17MM,
  RGB61_24MM,
  calculateGSD,
  forwardSpacing,
  lineSpacing,
} from "../domain/camera.ts";
// Minimal rotation-matrix builder (camera->world) matching worker.ts
function rotMat(omega_deg: number, phi_deg: number, kappa_deg: number): Float64Array {
  const o = (omega_deg*Math.PI)/180, p = (phi_deg*Math.PI)/180, k = (kappa_deg*Math.PI)/180;
  const co=Math.cos(o), so=Math.sin(o), cp=Math.cos(p), sp=Math.sin(p), ck=Math.cos(k), sk=Math.sin(k);
  const Rx = [1,0,0, 0,co,-so, 0,so,co];
  const Ry = [cp,0,sp, 0,1,0, -sp,0,cp];
  const Rz = [ck,-sk,0, sk,ck,0, 0,0,1];
  const mul3 = (A:number[], B:number[]) => {
    const R = new Float64Array(9);
    R[0]=A[0]*B[0]+A[1]*B[3]+A[2]*B[6];
    R[1]=A[0]*B[1]+A[1]*B[4]+A[2]*B[7];
    R[2]=A[0]*B[2]+A[1]*B[5]+A[2]*B[8];
    R[3]=A[3]*B[0]+A[4]*B[3]+A[5]*B[6];
    R[4]=A[3]*B[1]+A[4]*B[4]+A[5]*B[7];
    R[5]=A[3]*B[2]+A[4]*B[5]+A[5]*B[8];
    R[6]=A[6]*B[0]+A[7]*B[3]+A[8]*B[6];
    R[7]=A[6]*B[1]+A[7]*B[4]+A[8]*B[7];
    R[8]=A[6]*B[2]+A[7]*B[5]+A[8]*B[8];
    return R;
  };
  const A = mul3(Ry,Rx); return mul3(Rz, Array.from(A));
}

import type { CameraModel } from "../domain/types.ts";

type GSDStats = {
  min: number;
  max: number;
  mean: number;
  count: number;
  totalAreaM2?: number;
  histogram: { bin: number; count: number; areaM2?: number }[];
};

const AREA_SIDE_M = 1000; // 1 km
const AREA_M2 = AREA_SIDE_M * AREA_SIDE_M; // 1,000,000 m^2

const FRONT_OVERLAP = 70; // %
const SIDE_OVERLAP = 70; // %

function gsdStatsForFlat(camera: CameraModel, altitudeAGL: number, areaM2: number): GSDStats {
  const gsd = calculateGSD(camera, altitudeAGL);
  // Flat, nadir assumption -> GSD is uniform. Single-bin histogram over whole area.
  return {
    min: gsd,
    max: gsd,
    mean: gsd,
    count: 1, // not meaningful here; using 1 to indicate a single uniform value
    totalAreaM2: areaM2,
    histogram: [{ bin: gsd, count: 1, areaM2: areaM2 }],
  };
}

// Build simple axis-aligned flight lines across a W x H rectangle in meters.
// Lines run along X from (0,y) to (W,y), spaced by `lineSpacing` along Y.
function buildFlightLinesRect(widthM: number, heightM: number, lineSpacingM: number): Array<[[number,number],[number,number]]> {
  const lines: Array<[[number,number],[number,number]]> = [];
  if (lineSpacingM <= 0) return lines;
  const rows = Math.max(1, Math.ceil(heightM / lineSpacingM));
  for (let i = 0; i < rows; i++) {
    const y = Math.min(heightM, i * lineSpacingM);
    lines.push([[0, y], [widthM, y]]);
  }
  return lines;
}

// Sample photo trigger points along each line at `forwardSpacingM`.
function samplePhotosOnLines(lines: Array<[[number,number],[number,number]]>, forwardSpacingM: number): Array<[number,number]> {
  const points: Array<[number,number]> = [];
  if (forwardSpacingM <= 0) return points;
  for (const [[x0, y0], [x1, y1]] of lines) {
    const length = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.floor(length / forwardSpacingM));
    for (let s = 0; s <= steps; s++) {
      const t = Math.min(1, (s * forwardSpacingM) / length);
      const x = x0 + (x1 - x0) * t;
      const y = y0 + (y1 - y0) * t;
      points.push([x, y]);
    }
  }
  return points;
}

// Compute GSD using the same Jacobian-based math as worker.ts on a flat plane (z=0),
// sampling a coarse grid over the 1 km² area and taking the min GSD from all poses per pixel.
function computeGSDStatsViaWorkerMath(
  cam: CameraModel,
  altitudeAGL: number,
  frontOverlapPct: number,
  sideOverlapPct: number,
  gridSize = 64
): GSDStats {
  const widthM = AREA_SIDE_M, heightM = AREA_SIDE_M;
  const pixSize = widthM / gridSize; // square pixels

  // Flight lines: run along +X, spaced along Y
  const spacingLine = lineSpacing(cam, altitudeAGL, sideOverlapPct);
  const lines = buildFlightLinesRect(widthM, heightM, spacingLine);
  const spacingForward = forwardSpacing(cam, altitudeAGL, frontOverlapPct);
  const photos = samplePhotosOnLines(lines, spacingForward);

  // Build poses: nadir (omega=phi=0). Bearing along +X, wide side perpendicular (u axis along Y)
  // To make u-axis perpendicular to flightline (Y), yaw=90 deg.
  const kappa_deg = 90;
  const R = rotMat(0, 0, kappa_deg); // camera->world
  const RT = new Float64Array([ R[0],R[3],R[6], R[1],R[4],R[7], R[2],R[5],R[8] ]); // world->camera
  const bx = { x: R[0], y: R[3], z: R[6] }; // R*[1,0,0]
  const by = { x: R[1], y: R[4], z: R[7] }; // R*[0,1,0]

  // Per-camera constants
  const sensorW = cam.w_px * cam.sx_m;
  const sensorH = cam.h_px * cam.sy_m;
  const diagFovHalf = Math.atan(0.5 * Math.hypot(sensorW, sensorH) / cam.f_m);
  const diagTan = Math.tan(diagFovHalf);
  const cosIncMin = 1e-3;

  // Prepare pose structures
  const prepared = photos.map(([x, y]) => {
    const z = altitudeAGL; // above plane z=0
    const H = Math.max(1.0, z); // height above zMin=0
    const radius = H * diagTan * 1.25;
    return { x, y, z, R, RT, bx, by, radius, radiusSq: radius*radius };
  });

  // Iterate pixels and compute min GSD across poses
  const N = gridSize * gridSize;
  const gsdMin = new Float32Array(N).fill(Number.POSITIVE_INFINITY);
  const overlap = new Uint16Array(N);

  for (let row = 0; row < gridSize; row++) {
    for (let col = 0; col < gridSize; col++) {
      const idx = row * gridSize + col;
      const xw = (col + 0.5) * pixSize;
      const yw = (row + 0.5) * pixSize; // doesn't matter top/bottom for flat plane
      const zw = 0;

      let localMin = Number.POSITIVE_INFINITY;
      let localOverlap = 0;

      for (let p = 0; p < prepared.length; p++) {
        const pose = prepared[p];
        const dx = xw - pose.x;
        const dy = yw - pose.y;
        if (dx*dx + dy*dy > pose.radiusSq) continue; // coarse cull

        const vz = zw - pose.z; // negative
        const L = Math.hypot(dx, dy, vz);
        if (!(L > 0)) continue;
        const invL = 1 / L;
        const rx = dx * invL, ry = dy * invL, rz = vz * invL;
        const nx = 0, ny = 0, nz = 1; // flat plane normal
        const cosInc = -(nx*rx + ny*ry + nz*rz);
        if (cosInc <= cosIncMin) continue;

        // r_cam = R^T * r_world
        const rcx = pose.RT[0]*rx + pose.RT[1]*ry + pose.RT[2]*rz;
        const rcy = pose.RT[3]*rx + pose.RT[4]*ry + pose.RT[5]*rz;
        const rcz = pose.RT[6]*rx + pose.RT[7]*ry + pose.RT[8]*rz;
        if (Math.abs(rcz) < 1e-12) continue;

        const f = cam.f_m;
        const u_m = f * (rcx / rcz);
        const v_m = f * (rcy / rcz);

        // a = R * [u,v,f]^T
        const a0 = pose.R[0]*u_m + pose.R[1]*v_m + pose.R[2]*f;
        const a1 = pose.R[3]*u_m + pose.R[4]*v_m + pose.R[5]*f;
        const a2 = pose.R[6]*u_m + pose.R[7]*v_m + pose.R[8]*f;
        const denom = nx*a0 + ny*a1 + nz*a2; // = a2
        if (Math.abs(denom) < 1e-12) continue;

        const Hn = nx*(xw - pose.x) + ny*(yw - pose.y) + nz*(zw - pose.z); // = zw - z
        const invDen2 = 1.0 / (denom*denom);

        const nbx = nx*pose.bx.x + ny*pose.bx.y + nz*pose.bx.z; // = bz.z
        const Jux = (denom*pose.bx.x - nbx*a0) * Hn * invDen2;
        const Juy = (denom*pose.bx.y - nbx*a1) * Hn * invDen2;
        const Juz = (denom*pose.bx.z - nbx*a2) * Hn * invDen2;

        const nby = nx*pose.by.x + ny*pose.by.y + nz*pose.by.z;
        const Jvx = (denom*pose.by.x - nby*a0) * Hn * invDen2;
        const Jvy = (denom*pose.by.y - nby*a1) * Hn * invDen2;
        const Jvz = (denom*pose.by.z - nby*a2) * Hn * invDen2;

        const gsdx = Math.hypot(Jux, Juy, Juz) * cam.sx_m;
        const gsdy = Math.hypot(Jvx, Jvy, Jvz) * cam.sy_m;
        const gsd = Math.sqrt(gsdx * gsdy);

        localOverlap += 1;
        if (gsd < localMin) localMin = gsd;
      }

      if (localOverlap > 0) {
        overlap[idx] = localOverlap;
        gsdMin[idx] = localMin;
      }
    }
  }

  // Build stats from gsdMin over valid pixels
  let min = Number.POSITIVE_INFINITY, max = 0, sum = 0, count = 0;
  for (let i = 0; i < N; i++) {
    const v = gsdMin[i];
    if (!(v > 0 && isFinite(v))) continue;
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v; count++;
  }
  if (count === 0) return { min: 0, max: 0, mean: 0, count: 0, histogram: [] } as any;

  const mean = sum / count;
  return {
    min, max, mean, count,
    totalAreaM2: count * pixSize * pixSize,
    histogram: [{ bin: mean, count, areaM2: count * pixSize * pixSize }],
  };
}

function describeCamera(cam: CameraModel): string {
  // Prefer first friendly name if available
  // @ts-ignore optional names in CameraModel
  const names: string[] | undefined = cam.names;
  return names?.[0] || `f=${cam.f_m}m, ${cam.w_px}x${cam.h_px}`;
}

function runForAltitude(ALT_AGL: number) {
  const cameras: Record<string, CameraModel> = {
    SONY_RX1R2,
    SONY_RX1R3,
    SONY_A6100_20MM,
    DJI_ZENMUSE_P1_24MM,
    ILX_LR1_INSPECT_85MM,
    MAP61_17MM,
    RGB61_24MM,
  };

  console.log("=== Flat-ground GSD test ===");
  console.log(`Area: ${AREA_SIDE_M}m x ${AREA_SIDE_M}m (${(AREA_M2/1_000_000).toFixed(2)} km²)`);
  console.log(`Altitude AGL: ${ALT_AGL} m`);
  console.log(`Overlap: front=${FRONT_OVERLAP}%, side=${SIDE_OVERLAP}%`);
  console.log("");

  for (const [key, cam] of Object.entries(cameras)) {
    const spacingForward = forwardSpacing(cam, ALT_AGL, FRONT_OVERLAP);
    const spacingLine = lineSpacing(cam, ALT_AGL, SIDE_OVERLAP);
    const lines = buildFlightLinesRect(AREA_SIDE_M, AREA_SIDE_M, spacingLine);
    const photos = samplePhotosOnLines(lines, spacingForward);
    const stats = computeGSDStatsViaWorkerMath(cam, ALT_AGL, FRONT_OVERLAP, SIDE_OVERLAP, 64);

    console.log(`Camera: ${key} (${describeCamera(cam)})`);
    console.log(`  Forward spacing (70%): ${spacingForward.toFixed(2)} m`);
    console.log(`  Line spacing (70%):    ${spacingLine.toFixed(2)} m`);
    console.log(`  Flight lines: ${lines.length}, photos: ${photos.length}`);
    console.log(`  GSD stats: min=${(stats.min*100).toFixed(2)} cm, mean=${(stats.mean*100).toFixed(2)} cm, max=${(stats.max*100).toFixed(2)} cm`);
    console.log(`  Histogram bins: ${stats.histogram.length} (avg @ ${(stats.histogram[0].bin*100).toFixed(2)} cm, area=${(stats.histogram[0].areaM2||0).toFixed(0)} m²)`);
    console.log("");
  }
}

function run() {
  const altitudes = [70, 100, 120];
  for (const alt of altitudes) {
    runForAltitude(alt);
  }
}

// Execute when run directly
if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("flat_gsd.test.ts")) {
  run();
}

export { run };
