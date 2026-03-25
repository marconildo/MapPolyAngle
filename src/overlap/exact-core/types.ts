import type { FlightParams } from "@/domain/types";
import type { DensityStats, GSDStats, LidarWorkerIn, LidarWorkerOut, WorkerIn, WorkerOut } from "../types";

export type ExactCameraTileInput = WorkerIn;
export type ExactCameraTileOutput = WorkerOut;

export type ExactLidarTileInput = LidarWorkerIn;
export type ExactLidarTileOutput = LidarWorkerOut;

export type ExactStats = GSDStats;
export type ExactDensityStats = DensityStats;
export type ExactFlightParams = FlightParams;

export interface ExactScoreBreakdown {
  modelVersion: string;
  total: number;
  signals: Record<string, number>;
  weights: Record<string, number>;
  contributions: Record<string, number>;
}

export interface ExactLidarScore {
  qualityCost: number;
  targetDensityPtsM2: number;
  holeFraction: number;
  lowFraction: number;
  holeThreshold: number;
  weakThreshold: number;
  q10: number;
  q25: number;
  q10Deficit: number;
  q25Deficit: number;
  meanDeficit: number;
  breakdown: ExactScoreBreakdown;
}

export interface ExactCameraScore {
  qualityCost: number;
  targetGsdM: number;
  overTargetAreaFraction: number;
  q75: number;
  q90: number;
  meanOvershoot: number;
  q75Overshoot: number;
  q90Overshoot: number;
  maxOvershoot: number;
  breakdown: ExactScoreBreakdown;
}
