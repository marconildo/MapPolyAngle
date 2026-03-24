export type PartitionRankingSource = "surrogate" | "backend-exact" | "frontend-exact";
export type PartitionExactMetricKind = "gsd" | "density";

export interface TerrainPartitionRegionPreview {
  areaM2: number;
  bearingDeg: number;
  atomCount: number;
  ring: [number, number][];
  convexity: number;
  compactness: number;
  baseAltitudeAGL?: number;
  exactScore?: number | null;
  exactSeedBearingDeg?: number | null;
}

export interface TerrainPartitionSolutionPreview {
  signature: string;
  tradeoff: number;
  regionCount: number;
  totalMissionTimeSec: number;
  normalizedQualityCost: number;
  weightedMeanMismatchDeg: number;
  hierarchyLevel: number;
  largestRegionFraction: number;
  meanConvexity: number;
  boundaryBreakAlignment: number;
  isFirstPracticalSplit: boolean;
  rankingSource?: PartitionRankingSource;
  exactScore?: number | null;
  exactQualityCost?: number | null;
  exactMissionTimeSec?: number | null;
  exactMetricKind?: PartitionExactMetricKind | null;
  regions: TerrainPartitionRegionPreview[];
}
