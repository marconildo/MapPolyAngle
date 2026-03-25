import { DJI_ZENMUSE_P1_24MM, ILX_LR1_INSPECT_85MM, MAP61_17MM, RGB61_24MM, SONY_RX1R2, SONY_RX1R3, SONY_A6100_20MM, calculateGSD } from "@/domain/camera";
import { getLidarMappingFovDeg, getLidarModel, lidarDeliverableDensity } from "@/domain/lidar";
import type { FlightParams } from "@/domain/types";

import type { GSDStats } from "../types";
import type { ExactCameraScore, ExactLidarScore, ExactScoreBreakdown } from "./types";

const CAMERA_REGISTRY: Record<string, typeof SONY_RX1R2> = {
  SONY_RX1R2,
  SONY_RX1R3,
  SONY_A6100_20MM,
  DJI_ZENMUSE_P1_24MM,
  ILX_LR1_INSPECT_85MM,
  MAP61_17MM,
  RGB61_24MM,
};

const DEFAULT_CAMERA = SONY_RX1R2;

function buildScoreBreakdown(
  modelVersion: string,
  signals: Record<string, number>,
  weights: Record<string, number>,
) {
  const contributions = Object.fromEntries(
    Object.entries(weights).map(([key, weight]) => [key, (signals[key] ?? 0) * weight]),
  );
  const total = Object.values(contributions).reduce((sum, value) => sum + value, 0);
  return {
    modelVersion,
    total,
    signals,
    weights,
    contributions,
  } satisfies ExactScoreBreakdown;
}

export function statsTotalAreaM2(stats: GSDStats) {
  if (stats.totalAreaM2 && stats.totalAreaM2 > 0) return stats.totalAreaM2;
  return stats.histogram.reduce((sum, bin) => sum + (bin.areaM2 || 0), 0);
}

export function sortedHistogramBins(stats: GSDStats) {
  return [...stats.histogram]
    .filter((bin) => (bin.areaM2 || 0) > 0)
    .sort((a, b) => a.bin - b.bin);
}

export function histogramBinEdges(stats: GSDStats) {
  const bins = sortedHistogramBins(stats);
  if (bins.length === 0) return { bins, edges: [] as number[] };
  if (bins.length === 1) {
    const only = bins[0].bin;
    const lo = Number.isFinite(stats.min) ? Math.min(stats.min, only) : only;
    const hi = Number.isFinite(stats.max) ? Math.max(stats.max, only) : only;
    return { bins, edges: [lo, hi > lo ? hi : lo + 1] };
  }
  const centers = bins.map((bin) => bin.bin);
  const edges = new Array<number>(bins.length + 1);
  edges[0] = centers[0] - (centers[1] - centers[0]) * 0.5;
  for (let index = 1; index < centers.length; index++) {
    edges[index] = (centers[index - 1] + centers[index]) * 0.5;
  }
  edges[bins.length] = centers[bins.length - 1] + (centers[bins.length - 1] - centers[bins.length - 2]) * 0.5;
  return { bins, edges };
}

export function histogramAreaBelow(stats: GSDStats, threshold: number) {
  const { bins, edges } = histogramBinEdges(stats);
  let area = 0;
  for (let index = 0; index < bins.length; index++) {
    const bin = bins[index];
    const binArea = bin.areaM2 || 0;
    if (binArea <= 0) continue;
    const lo = edges[index];
    const hi = edges[index + 1];
    if (threshold >= hi) {
      area += binArea;
      continue;
    }
    if (threshold <= lo) continue;
    const width = hi - lo;
    if (!(width > 0)) {
      if (threshold >= bin.bin) area += binArea;
      continue;
    }
    area += binArea * Math.max(0, Math.min(1, (threshold - lo) / width));
  }
  return area;
}

export function histogramQuantile(stats: GSDStats, q: number) {
  const { bins, edges } = histogramBinEdges(stats);
  const totalArea = statsTotalAreaM2(stats);
  if (!(totalArea > 0) || bins.length === 0) return 0;
  const target = Math.max(0, Math.min(1, q)) * totalArea;
  let cumulative = 0;
  for (let index = 0; index < bins.length; index++) {
    const bin = bins[index];
    const binArea = bin.areaM2 || 0;
    if (binArea <= 0) continue;
    const next = cumulative + binArea;
    if (target <= next || index === bins.length - 1) {
      const lo = edges[index];
      const hi = edges[index + 1];
      const width = hi - lo;
      if (!(width > 0)) return bin.bin;
      const within = Math.max(0, Math.min(1, (target - cumulative) / binArea));
      return lo + width * within;
    }
    cumulative = next;
  }
  return bins[bins.length - 1].bin;
}

export function scoreExactLidarStats(stats: GSDStats, params: FlightParams): ExactLidarScore {
  const model = getLidarModel(params.lidarKey);
  const mappingFovDeg = getLidarMappingFovDeg(model, params.mappingFovDeg);
  const speedMps = params.speedMps ?? model.defaultSpeedMps;
  const returnMode = params.lidarReturnMode ?? "single";
  const targetDensityPtsM2 = params.pointDensityPtsM2
    ?? lidarDeliverableDensity(model, params.altitudeAGL, params.sideOverlap, speedMps, returnMode, mappingFovDeg);
  const totalAreaM2 = Math.max(1, statsTotalAreaM2(stats));
  const holeThreshold = Math.max(5, targetDensityPtsM2 * 0.2);
  const weakThreshold = Math.max(holeThreshold + 1e-6, targetDensityPtsM2 * 0.7);
  const q10 = histogramQuantile(stats, 0.1);
  const q25 = histogramQuantile(stats, 0.25);
  const holeFraction = histogramAreaBelow(stats, holeThreshold) / totalAreaM2;
  const lowFraction = histogramAreaBelow(stats, weakThreshold) / totalAreaM2;
  const q10Deficit = Math.max(0, 1 - q10 / Math.max(1e-6, targetDensityPtsM2));
  const q25Deficit = Math.max(0, 1 - q25 / Math.max(1e-6, targetDensityPtsM2));
  const meanDeficit = Math.max(0, 1 - stats.mean / Math.max(1e-6, targetDensityPtsM2));
  const breakdown = buildScoreBreakdown(
    "lidar-region-v1",
    {
      holeFraction,
      lowFraction,
      q10Deficit,
      q25Deficit,
      meanDeficit,
    },
    {
      holeFraction: 4.2,
      lowFraction: 2.4,
      q10Deficit: 1.9,
      q25Deficit: 1.2,
      meanDeficit: 0.8,
    },
  );
  const qualityCost = breakdown.total;
  return {
    qualityCost,
    targetDensityPtsM2,
    holeFraction,
    lowFraction,
    holeThreshold,
    weakThreshold,
    q10,
    q25,
    q10Deficit,
    q25Deficit,
    meanDeficit,
    breakdown,
  };
}

export function scoreExactCameraStats(stats: GSDStats, params: FlightParams): ExactCameraScore {
  const cameraKey = params.cameraKey;
  const camera = cameraKey ? CAMERA_REGISTRY[cameraKey] || DEFAULT_CAMERA : DEFAULT_CAMERA;
  const targetGsdM = calculateGSD(camera, params.altitudeAGL);
  const totalAreaM2 = Math.max(1, statsTotalAreaM2(stats));
  const q75 = histogramQuantile(stats, 0.75);
  const q90 = histogramQuantile(stats, 0.9);
  const overTargetAreaFraction = Math.max(0, totalAreaM2 - histogramAreaBelow(stats, targetGsdM)) / totalAreaM2;
  const meanOvershoot = Math.max(0, stats.mean / Math.max(1e-6, targetGsdM) - 1);
  const q75Overshoot = Math.max(0, q75 / Math.max(1e-6, targetGsdM) - 1);
  const q90Overshoot = Math.max(0, q90 / Math.max(1e-6, targetGsdM) - 1);
  const maxOvershoot = Math.max(0, stats.max / Math.max(1e-6, targetGsdM) - 1);
  const breakdown = buildScoreBreakdown(
    "camera-region-v1",
    {
      q90Overshoot,
      overTargetAreaFraction,
      meanOvershoot,
      q75Overshoot,
      maxOvershoot,
    },
    {
      q90Overshoot: 1.85,
      overTargetAreaFraction: 1.25,
      meanOvershoot: 0.95,
      q75Overshoot: 0.55,
      maxOvershoot: 0.2,
    },
  );
  const qualityCost = breakdown.total;
  return {
    qualityCost,
    targetGsdM,
    overTargetAreaFraction,
    q75,
    q90,
    meanOvershoot,
    q75Overshoot,
    q90Overshoot,
    maxOvershoot,
    breakdown,
  };
}
