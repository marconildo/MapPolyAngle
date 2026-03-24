import type { CameraModel } from "./types";

export function rotMat(omega_deg: number, phi_deg: number, kappa_deg: number): Float64Array {
  const o = omega_deg*Math.PI/180, p = phi_deg*Math.PI/180, k = kappa_deg*Math.PI/180;
  const co=Math.cos(o), so=Math.sin(o), cp=Math.cos(p), sp=Math.sin(p), ck=Math.cos(k), sk=Math.sin(k);
  const Rx = [1,0,0, 0,co,-so, 0,so,co];
  const Ry = [cp,0,sp, 0,1,0, -sp,0,cp];
  const Rz = [ck,-sk,0, sk,ck,0, 0,0,1];
  // R = Rz * Ry * Rx
  const A = mul3(Ry,Rx); return mul3(Rz, A);
}

function mul3(A: ArrayLike<number>, B: ArrayLike<number>): Float64Array {
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
}

export function camRayToPixel(camera: CameraModel, RT: Float64Array, Sx:number,Sy:number,Sz:number,
                              Px:number,Py:number,Pz:number) {
  // world->camera coordinates: pc = R^T (P - S)
  const Rx0=RT[0], Rx1=RT[3], Rx2=RT[6];
  const Ry0=RT[1], Ry1=RT[4], Ry2=RT[7];
  const Rz0=RT[2], Rz1=RT[5], Rz2=RT[8];
  const vx = Px - Sx, vy = Py - Sy, vz = Pz - Sz;
  const cx = Rx0*vx + Rx1*vy + Rx2*vz;
  const cy = Ry0*vx + Ry1*vy + Ry2*vz;
  const cz = Rz0*vx + Rz1*vy + Rz2*vz; // camera looks along -Z; cz should be negative when in front
  if (cz >= -1e-9) return null;

  const cxpx = camera.cx_px ?? camera.w_px * 0.5;
  const cypx = camera.cy_px ?? camera.h_px * 0.5;
  const u = cxpx + (camera.f_m * (cx / -cz)) / camera.sx_m;
  const v = cypx + (camera.f_m * (cy / -cz)) / camera.sy_m;

  if (u < 0 || v < 0 || u >= camera.w_px || v >= camera.h_px) return null;
  return {u, v, range: Math.hypot(vx, vy, vz)};
}

export function normalFromDEM(
  dem: Float32Array, size: number, row: number, col: number, pixelSizeM: number
): [number,number,number] {
  const clamp = (v:number,lo:number,hi:number)=>Math.max(lo,Math.min(hi,v));
  const r0 = clamp(row-1,0,size-1), r1 = clamp(row+1,0,size-1);
  const c0 = clamp(col-1,0,size-1), c1 = clamp(col+1,0,size-1);
  const idx = (r:number,c:number)=>r*size + c;
  const dzdx = (dem[idx(row,c1)] - dem[idx(row,c0)]) / ((c1-c0)*pixelSizeM);
  const dzdy = (dem[idx(r1,col)] - dem[idx(r0,col)]) / ((r1-r0)*pixelSizeM);
  let nx = -dzdx, ny = -dzdy, nz = 1;
  const inv = 1 / Math.hypot(nx,ny,nz);
  return [nx*inv, ny*inv, nz*inv];
}
