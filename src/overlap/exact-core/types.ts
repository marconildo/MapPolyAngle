import type { FlightParams } from "@/domain/types";
import type { DensityStats, GSDStats, LidarWorkerIn, LidarWorkerOut, WorkerIn, WorkerOut } from "../types";

export type ExactCameraTileInput = WorkerIn;
export type ExactCameraTileOutput = WorkerOut;

export type ExactLidarTileInput = LidarWorkerIn;
export type ExactLidarTileOutput = LidarWorkerOut;

export type ExactStats = GSDStats;
export type ExactDensityStats = DensityStats;
export type ExactFlightParams = FlightParams;

export interface ExactLidarScore {
  qualityCost: number;
  targetDensityPtsM2: number;
  holeFraction: number;
  lowFraction: number;
  q10: number;
  q25: number;
}

export interface ExactCameraScore {
  qualityCost: number;
  targetGsdM: number;
  overTargetAreaFraction: number;
  q75: number;
  q90: number;
}
