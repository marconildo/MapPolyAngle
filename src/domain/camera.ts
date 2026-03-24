/**
 * Standard camera models and spacing calculations for flight planning.
 */

import type { CameraModel } from './types';

// Sony RX1R II camera specifications (canonical definition)
export const SONY_RX1R2: CameraModel = {
  f_m: 0.035,          // 35 mm fixed lens
  sx_m: 4.88e-6,       // 4.88 µm pixel pitch (42.4MP full frame)
  sy_m: 4.88e-6,
  w_px: 7952,          // 7952 x 5304 pixels
  h_px: 5304,
  names: [ 'RX1RII 42MP', 'RX1RII', 'RX1R2', 'SONY_RX1R2' ],
};

// DJI Zenmuse P1 24mm (8192 x 5460, 4.27246 µm pixels, ~24 mm focal length)
export const DJI_ZENMUSE_P1_24MM: CameraModel = {
  f_m: 5626.690009970837 * 4.27246e-6, // convert focal length in px to meters using pixel size => ≈0.02404 m
  sx_m: 4.27246e-6,
  sy_m: 4.27246e-6,
  w_px: 8192,
  h_px: 5460,
  cx_px: 4075.470103874583, // provided principal point
  cy_px: 2747.220102704297,
  names: [ 'DJI Zenmuse P1 24mm', 'Zenmuse P1 24mm', 'P1 24mm', 'ZENMUSE_P1_24MM' ],
};

// INSPECT, 85mm configuration (9504 x 6336)
// Pixel size derived from sensorWidth(35.7mm)/imageWidth(9504) ≈ 3.756e-6 m
export const ILX_LR1_INSPECT_85MM: CameraModel = {
  f_m: 0.085,          // 85 mm lens
  sx_m: 35.7e-3 / 9504, // ≈3.756 µm
  sy_m: 23.8e-3 / 6336, // ≈3.756 µm
  w_px: 9504,
  h_px: 6336,
  names: [ 'INSPECT', 'ILX-LR1 85mm', 'ILX_LR1_INSPECT_85MM', 'MAPSTARHighRes', 'MAPSTARHighRes_v4', 'MAPSTARHighRes_v5' ],
};

// MAP61 17mm
export const MAP61_17MM: CameraModel = {
  f_m: 0.017,          // 17 mm lens
  sx_m: 35.7e-3 / 9504,
  sy_m: 23.8e-3 / 6336,
  w_px: 9504,
  h_px: 6336,
  names: [ 'MAP61', 'MAP61 17mm', 'MAP61_17MM', 'MAPSTAROblique', 'MAPSTAROblique_v4', 'MAPSTAROblique_v5' ],
};

// RGB61 24mm (36.0 x 24.0 mm sensor, 9504 x 6336, 24 mm lens)
// Pixel size: 36.0mm/9504 ≈ 3.787 µm
export const RGB61_24MM: CameraModel = {
  f_m: 0.024,          // 24 mm lens
  sx_m: 36.0e-3 / 9504,
  sy_m: 24.0e-3 / 6336,
  w_px: 9504,
  h_px: 6336,
  names: [ 'RGB61', 'RGB61 24mm', 'RGB61_24MM', 'RGB61_v4', 'RGB61_v5' ],
};

/**
 * Calculate the forward spacing between photos based on overlap percentage.
 */
export function forwardSpacing(
  camera: CameraModel,
  altitudeAGL: number,
  frontOverlapPct: number
): number {
  // Ground sample distance (GSD)
  const gsd = (camera.sy_m * altitudeAGL) / camera.f_m;
  // Photo footprint in the forward direction (image height aligned with flight)
  const photoFootprintForward = camera.h_px * gsd;
  // Forward spacing accounting for overlap
  const overlapFraction = frontOverlapPct / 100;
  return photoFootprintForward * (1 - overlapFraction);
}

/**
 * Calculate forward spacing, optionally swapping width/height (rotate 90°).
 */
export function forwardSpacingRotated(
  camera: CameraModel,
  altitudeAGL: number,
  frontOverlapPct: number,
  rotate90: boolean
): number {
  const gsdX = (camera.sx_m * altitudeAGL) / camera.f_m;
  const gsdY = (camera.sy_m * altitudeAGL) / camera.f_m;
  const alongPx = rotate90 ? camera.w_px : camera.h_px;
  const alongGsd = rotate90 ? gsdX : gsdY;
  const photoFootprintForward = alongPx * alongGsd;
  const overlapFraction = frontOverlapPct / 100;
  return photoFootprintForward * (1 - overlapFraction);
}

/**
 * Calculate the spacing between flight lines based on side overlap percentage.
 */
export function lineSpacing(
  camera: CameraModel,
  altitudeAGL: number,
  sideOverlapPct: number
): number {
  // Ground sample distance (GSD)
  const gsd = (camera.sx_m * altitudeAGL) / camera.f_m;
  // Photo footprint in the side direction (image width aligned cross-track)
  const photoFootprintSide = camera.w_px * gsd;
  // Line spacing accounting for overlap
  const overlapFraction = sideOverlapPct / 100;
  return photoFootprintSide * (1 - overlapFraction);
}

/**
 * Calculate line spacing, optionally swapping width/height (rotate 90°).
 */
export function lineSpacingRotated(
  camera: CameraModel,
  altitudeAGL: number,
  sideOverlapPct: number,
  rotate90: boolean
): number {
  const gsdX = (camera.sx_m * altitudeAGL) / camera.f_m;
  const gsdY = (camera.sy_m * altitudeAGL) / camera.f_m;
  const acrossPx = rotate90 ? camera.h_px : camera.w_px;
  const acrossGsd = rotate90 ? gsdY : gsdX;
  const photoFootprintSide = acrossPx * acrossGsd;
  const overlapFraction = sideOverlapPct / 100;
  return photoFootprintSide * (1 - overlapFraction);
}

/**
 * Calculate Ground Sample Distance (GSD) at given altitude.
 */
export function calculateGSD(camera: CameraModel, altitudeAGL: number): number {
  // Use the larger pixel size for conservative GSD estimate
  const pixelSize = Math.max(camera.sx_m, camera.sy_m);
  return (pixelSize * altitudeAGL) / camera.f_m;
}
