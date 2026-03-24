import type { FlightParams } from "@/domain/types";
import type { ExactMetricKind } from "@/overlap/exact-region";
import type { GSDStats } from "@/overlap/types";
import type { TerrainPartitionSolutionPreview } from "@/terrain-partition/types";
import type { TerrainSourceSelection } from "@/terrain/types";

export type ExactRuntimeOperation =
  | "terrain-batch"
  | "evaluate-region"
  | "evaluate-solution"
  | "optimize-bearing"
  | "rerank-solutions";

export type ExactRuntimeTileRequest = {
  z: number;
  x: number;
  y: number;
  padTiles?: number;
};

export type ExactRuntimeTilePayload = {
  z: number;
  x: number;
  y: number;
  size: number;
  pngBase64: string;
  demPngBase64?: string | null;
  demSize?: number | null;
  demPadTiles?: number | null;
};

export type ExactRuntimeTerrainBatchRequest = {
  operation: "terrain-batch";
  terrainSource: TerrainSourceSelection;
  tiles: ExactRuntimeTileRequest[];
};

export type ExactRuntimeTerrainBatchResponse = {
  operation: "terrain-batch";
  tiles: ExactRuntimeTilePayload[];
};

export type ExactRuntimeCommonRequest = {
  payloadKind: "camera" | "lidar";
  terrainSource: TerrainSourceSelection;
  params: FlightParams;
  ring: [number, number][];
  altitudeMode: "legacy" | "min-clearance";
  minClearanceM: number;
  turnExtendM: number;
  exactOptimizeZoom?: number;
  clipInnerBufferM?: number;
  minOverlapForGsd?: number;
  timeWeight?: number;
};

export type ExactRuntimeEvaluateRegionRequest = ExactRuntimeCommonRequest & {
  operation: "evaluate-region";
  scopeId?: string;
  bearingDeg: number;
};

export type ExactRuntimeOptimizeBearingRequest = ExactRuntimeCommonRequest & {
  operation: "optimize-bearing";
  polygonId?: string;
  scopeId?: string;
  seedBearingDeg: number;
  mode?: "local" | "global";
  halfWindowDeg?: number;
};

export type ExactRuntimeEvaluateSolutionRequest = ExactRuntimeCommonRequest & {
  operation: "evaluate-solution";
  polygonId: string;
  solution: TerrainPartitionSolutionPreview;
  fastestMissionTimeSec: number;
  rankingSource?: "backend-exact" | "frontend-exact";
};

export type ExactRuntimeRerankSolutionsRequest = ExactRuntimeCommonRequest & {
  operation: "rerank-solutions";
  polygonId: string;
  solutions: TerrainPartitionSolutionPreview[];
  rankingSource?: "backend-exact" | "frontend-exact";
};

export type ExactRuntimeRequest =
  | ExactRuntimeTerrainBatchRequest
  | ExactRuntimeEvaluateRegionRequest
  | ExactRuntimeEvaluateSolutionRequest
  | ExactRuntimeOptimizeBearingRequest
  | ExactRuntimeRerankSolutionsRequest;

export type ExactRuntimeRegionCandidate = {
  bearingDeg: number;
  exactCost: number;
  qualityCost: number;
  missionTimeSec: number;
  normalizedTimeCost: number;
  metricKind: ExactMetricKind;
  stats: GSDStats;
  diagnostics: Record<string, number>;
};

export type ExactRuntimeEvaluateRegionResponse = {
  operation: "evaluate-region";
  candidate: ExactRuntimeRegionCandidate | null;
};

export type ExactRuntimeOptimizeBearingResponse = {
  operation: "optimize-bearing";
  best: ExactRuntimeRegionCandidate | null;
  evaluated: ExactRuntimeRegionCandidate[];
  seedBearingDeg: number;
  lineSpacingM: number;
  metricKind: ExactMetricKind | null;
};

export type ExactRuntimePartitionPreviewPayload = {
  metricKind: ExactMetricKind;
  stats: GSDStats;
  regionStats: GSDStats[];
  regionCount: number;
  sampleCount: number;
  sampleLabel: string;
};

export type ExactRuntimeEvaluateSolutionResponse = {
  operation: "evaluate-solution";
  solution: TerrainPartitionSolutionPreview;
  preview: ExactRuntimePartitionPreviewPayload;
};

export type ExactRuntimeRerankSolutionsResponse = {
  operation: "rerank-solutions";
  bestIndex: number;
  solutions: TerrainPartitionSolutionPreview[];
  previewsBySignature: Record<string, ExactRuntimePartitionPreviewPayload>;
};

export type ExactRuntimeResponse =
  | ExactRuntimeTerrainBatchResponse
  | ExactRuntimeEvaluateRegionResponse
  | ExactRuntimeEvaluateSolutionResponse
  | ExactRuntimeOptimizeBearingResponse
  | ExactRuntimeRerankSolutionsResponse;

export type ExactRuntimeEnvelope = {
  id: string;
  request: ExactRuntimeRequest;
};

export type ExactRuntimeEnvelopeResponse = {
  id: string;
  ok: true;
  response: ExactRuntimeResponse;
} | {
  id: string;
  ok: false;
  error: string;
};
