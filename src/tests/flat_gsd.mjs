// Flat-ground GSD test (no GUI, pure JS runner)
// Node 18+: run: node src/tests/flat_gsd.mjs

// --- Camera models (copied from src/domain/camera.ts) ---
const SONY_RX1R2 = {
  f_m: 0.035,
  sx_m: 4.88e-6,
  sy_m: 4.88e-6,
  w_px: 7952,
  h_px: 5304,
  names: [ 'RX1RII 42MP', 'RX1RII', 'RX1R2', 'SONY_RX1R2' ],
};

const SONY_RX1R3 = {
  f_m: 0.035,
  sx_m: 35.7e-3 / 9504,
  sy_m: 23.8e-3 / 6336,
  w_px: 9504,
  h_px: 6336,
  names: [ 'SURVEY61', 'SURVEY61 v5', 'SUR61', 'RX1R3_v5', 'RX1R3', 'SONY_RX1R3' ],
};

const SONY_A6100_20MM = {
  f_m: 0.020,
  sx_m: 23.5e-3 / 6000,
  sy_m: 15.6e-3 / 4000,
  w_px: 6000,
  h_px: 4000,
  names: [ 'SURVEY24', 'A6100_v5', 'SURVEY24 v5', 'SUR24', 'SONY_A6100_20MM' ],
};

const DJI_ZENMUSE_P1_24MM = {
  f_m: 5626.690009970837 * 4.27246e-6,
  sx_m: 4.27246e-6,
  sy_m: 4.27246e-6,
  w_px: 8192,
  h_px: 5460,
  cx_px: 4075.470103874583,
  cy_px: 2747.220102704297,
  names: [ 'DJI Zenmuse P1 24mm', 'Zenmuse P1 24mm', 'P1 24mm', 'ZENMUSE_P1_24MM' ],
};

const ILX_LR1_INSPECT_85MM = {
  f_m: 0.085,
  sx_m: 35.7e-3 / 9504,
  sy_m: 23.8e-3 / 6336,
  w_px: 9504,
  h_px: 6336,
  names: [ 'INSPECT', 'ILX-LR1 85mm', 'ILX_LR1_INSPECT_85MM', 'MAPSTARHighRes_v4' ],
};

const MAP61_17MM = {
  f_m: 0.017,
  sx_m: 35.7e-3 / 9504,
  sy_m: 23.8e-3 / 6336,
  w_px: 9504,
  h_px: 6336,
  names: [ 'MAP61', 'MAP61 17mm', 'MAP61_17MM', 'MAPSTAROblique_v4' ],
};

const RGB61_24MM = {
  f_m: 0.024,
  sx_m: 36.0e-3 / 9504,
  sy_m: 24.0e-3 / 6336,
  w_px: 9504,
  h_px: 6336,
  names: [ 'RGB61', 'RGB61 24mm', 'RGB61_24MM', 'RGB61_v4' ],
};

// --- Spacing helpers (from src/domain/camera.ts) ---
function forwardSpacing(camera, altitudeAGL, frontOverlapPct) {
  const gsd = (camera.sy_m * altitudeAGL) / camera.f_m;
  const footprint = camera.h_px * gsd;
  return footprint * (1 - frontOverlapPct / 100);
}
function lineSpacing(camera, altitudeAGL, sideOverlapPct) {
  const gsd = (camera.sx_m * altitudeAGL) / camera.f_m;
  const footprint = camera.w_px * gsd;
  return footprint * (1 - sideOverlapPct / 100);
}

// --- Minimal math utilities (as in worker) ---
function rotMat(omega_deg, phi_deg, kappa_deg) {
  const o = (omega_deg*Math.PI)/180, p = (phi_deg*Math.PI)/180, k = (kappa_deg*Math.PI)/180;
  const co=Math.cos(o), so=Math.sin(o), cp=Math.cos(p), sp=Math.sin(p), ck=Math.cos(k), sk=Math.sin(k);
  const Rx = [1,0,0, 0,co,-so, 0,so,co];
  const Ry = [cp,0,sp, 0,1,0, -sp,0,cp];
  const Rz = [ck,-sk,0, sk,ck,0, 0,0,1];
  const mul3 = (A,B)=>{
    const R = new Float64Array(9);
    R[0]=A[0]*B[0]+A[1]*B[3]+A[2]*B[6]; R[1]=A[0]*B[1]+A[1]*B[4]+A[2]*B[7]; R[2]=A[0]*B[2]+A[1]*B[5]+A[2]*B[8];
    R[3]=A[3]*B[0]+A[4]*B[3]+A[5]*B[6]; R[4]=A[3]*B[1]+A[4]*B[4]+A[5]*B[7]; R[5]=A[3]*B[2]+A[4]*B[5]+A[5]*B[8];
    R[6]=A[6]*B[0]+A[7]*B[3]+A[8]*B[6]; R[7]=A[6]*B[1]+A[7]*B[4]+A[8]*B[7]; R[8]=A[6]*B[2]+A[7]*B[5]+A[8]*B[8];
    return R;
  };
  const A = mul3(Ry,Rx); return mul3(Rz, Array.from(A));
}

// --- Flight lines and sampling ---
const AREA_SIDE_M = 1000;
const AREA_M2 = AREA_SIDE_M * AREA_SIDE_M;
const FRONT = 70, SIDE = 70;

function buildFlightLinesRect(widthM, heightM, lineSpacingM) {
  const lines = [];
  if (lineSpacingM <= 0) return lines;
  const rows = Math.max(1, Math.ceil(heightM / lineSpacingM));
  for (let i = 0; i < rows; i++) {
    const y = Math.min(heightM, i * lineSpacingM);
    lines.push([[0, y], [widthM, y]]);
  }
  return lines;
}
function samplePhotosOnLines(lines, forwardSpacingM) {
  const points = [];
  if (forwardSpacingM <= 0) return points;
  for (const [[x0, y0], [x1, y1]] of lines) {
    const length = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.max(1, Math.floor(length / forwardSpacingM));
    for (let s = 0; s <= steps; s++) {
      const t = Math.min(1, (s * forwardSpacingM) / length);
      points.push([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]);
    }
  }
  return points;
}

function computeGSDStatsViaWorkerMath(cam, altitudeAGL, gridSize = 64) {
  const widthM = AREA_SIDE_M, heightM = AREA_SIDE_M;
  const pixSize = widthM / gridSize;
  const spacingLine = lineSpacing(cam, altitudeAGL, SIDE);
  const lines = buildFlightLinesRect(widthM, heightM, spacingLine);
  const spacingForward = forwardSpacing(cam, altitudeAGL, FRONT);
  const photos = samplePhotosOnLines(lines, spacingForward);

  const kappa_deg = 90; // yaw so wide side ⟂ flight lines
  const R = rotMat(0, 0, kappa_deg);
  const RT = new Float64Array([ R[0],R[3],R[6], R[1],R[4],R[7], R[2],R[5],R[8] ]);
  const bx = { x: R[0], y: R[3], z: R[6] };
  const by = { x: R[1], y: R[4], z: R[7] };
  const sensorW = cam.w_px * cam.sx_m;
  const sensorH = cam.h_px * cam.sy_m;
  const diagFovHalf = Math.atan(0.5 * Math.hypot(sensorW, sensorH) / cam.f_m);
  const diagTan = Math.tan(diagFovHalf);
  const cosIncMin = 1e-3;

  const prepared = photos.map(([x,y])=>{
    const z = altitudeAGL; const H = Math.max(1.0, z); const radius = H * diagTan * 1.25;
    return { x,y,z, R, RT, bx, by, radiusSq: radius*radius };
  });

  const N = gridSize * gridSize; const gsdMin = new Float32Array(N).fill(Number.POSITIVE_INFINITY);
  let min=Infinity, max=0, sum=0, count=0;
  for (let row=0; row<gridSize; row++) {
    for (let col=0; col<gridSize; col++) {
      const idx = row*gridSize + col;
      const xw = (col + 0.5) * pixSize; const yw = (row + 0.5) * pixSize; const zw = 0;
      let localMin = Infinity; let localOverlap = 0;
      for (let i=0;i<prepared.length;i++) {
        const p = prepared[i]; const dx = xw - p.x; const dy = yw - p.y; if (dx*dx + dy*dy > p.radiusSq) continue;
        const vz = zw - p.z; const L = Math.hypot(dx,dy,vz); if (!(L>0)) continue; const invL = 1/L; const rx = dx*invL, ry=dy*invL, rz=vz*invL;
        const cosInc = -(0*rx + 0*ry + 1*rz); if (cosInc <= cosIncMin) continue;
        const rcx = RT[0]*rx + RT[1]*ry + RT[2]*rz; const rcy = RT[3]*rx + RT[4]*ry + RT[5]*rz; const rcz = RT[6]*rx + RT[7]*ry + RT[8]*rz; if (Math.abs(rcz)<1e-12) continue;
        const f = cam.f_m; const u_m = f*(rcx/rcz); const v_m = f*(rcy/rcz);
        const a0 = R[0]*u_m + R[1]*v_m + R[2]*f; const a1 = R[3]*u_m + R[4]*v_m + R[5]*f; const a2 = R[6]*u_m + R[7]*v_m + R[8]*f; const denom = a2; if (Math.abs(denom)<1e-12) continue;
        const Hn = zw - p.z; const invDen2 = 1/(denom*denom);
        const nbx = p.bx.z; const Jux = (denom*p.bx.x - nbx*a0)*Hn*invDen2; const Juy = (denom*p.bx.y - nbx*a1)*Hn*invDen2; const Juz = (denom*p.bx.z - nbx*a2)*Hn*invDen2;
        const nby = p.by.z; const Jvx = (denom*p.by.x - nby*a0)*Hn*invDen2; const Jvy = (denom*p.by.y - nby*a1)*Hn*invDen2; const Jvz = (denom*p.by.z - nby*a2)*Hn*invDen2;
        const gsdx = Math.hypot(Jux,Juy,Juz)*cam.sx_m; const gsdy = Math.hypot(Jvx,Jvy,Jvz)*cam.sy_m; const gsd = Math.sqrt(gsdx*gsdy);
        localOverlap++; if (gsd < localMin) localMin = gsd;
      }
      if (localOverlap > 0 && isFinite(localMin)) { gsdMin[idx]=localMin; if(localMin<min)min=localMin; if(localMin>max)max=localMin; sum+=localMin; count++; }
    }
  }
  if (!(count>0)) return { min:0, max:0, mean:0, count:0, histogram:[] };
  const mean = sum/count; return { min, max, mean, count, totalAreaM2: count*pixSize*pixSize, histogram:[{ bin: mean, count, areaM2: count*pixSize*pixSize }] };
}

function describeCamera(cam){ return cam.names?.[0] || `f=${cam.f_m}m ${cam.w_px}x${cam.h_px}`; }

function runForAltitude(ALT){
  const cameras = { SONY_RX1R2, SONY_RX1R3, SONY_A6100_20MM, DJI_ZENMUSE_P1_24MM, ILX_LR1_INSPECT_85MM, MAP61_17MM, RGB61_24MM };
  console.log("=== Flat-ground GSD test ===");
  console.log(`Area: ${AREA_SIDE_M}m x ${AREA_SIDE_M}m (${(AREA_M2/1_000_000).toFixed(2)} km²)`);
  console.log(`Altitude AGL: ${ALT} m`);
  console.log(`Overlap: front=${FRONT}%, side=${SIDE}%\n`);
  for (const [key, cam] of Object.entries(cameras)){
    const fwd = forwardSpacing(cam, ALT, FRONT); const line = lineSpacing(cam, ALT, SIDE);
    const lines = buildFlightLinesRect(AREA_SIDE_M, AREA_SIDE_M, line); const photos = samplePhotosOnLines(lines, fwd);
    const stats = computeGSDStatsViaWorkerMath(cam, ALT, 64);
    console.log(`Camera: ${key} (${describeCamera(cam)})`);
    console.log(`  Forward spacing (70%): ${fwd.toFixed(2)} m`);
    console.log(`  Line spacing (70%):    ${line.toFixed(2)} m`);
    console.log(`  Flight lines: ${lines.length}, photos: ${photos.length}`);
    console.log(`  GSD stats: min=${(stats.min*100).toFixed(2)} cm, mean=${(stats.mean*100).toFixed(2)} cm, max=${(stats.max*100).toFixed(2)} cm`);
    console.log(`  Histogram: ${stats.histogram.length} bin(s)\n`);
  }
}

function run(){ [70,100,120].forEach(runForAltitude); }

run();
