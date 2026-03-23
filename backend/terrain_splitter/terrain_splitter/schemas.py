from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

PayloadKind = Literal["camera", "lidar"]
AltitudeMode = Literal["legacy", "min-clearance"]
LidarReturnMode = Literal["single", "dual", "triple"]
LidarComparisonMode = Literal["first-return", "all-returns"]
TerrainSourceMode = Literal["mapbox", "blended"]
DsmProcessingStatus = Literal["ready"]
PartitionRankingSource = Literal["surrogate", "backend-exact", "frontend-exact"]
ExactMetricKind = Literal["gsd", "density"]


class FlightParamsModel(BaseModel):
    payloadKind: PayloadKind = "camera"
    altitudeAGL: float = Field(..., gt=0)
    frontOverlap: float = Field(70, ge=0, le=95)
    sideOverlap: float = Field(70, ge=0, le=95)
    cameraKey: str | None = None
    lidarKey: str | None = None
    triggerDistanceM: float | None = None
    cameraYawOffsetDeg: float | None = None
    speedMps: float | None = None
    lidarReturnMode: LidarReturnMode | None = None
    mappingFovDeg: float | None = None
    lidarFrameRateHz: float | None = None
    lidarAzimuthSectorCenterDeg: float | None = None
    lidarBoresightYawDeg: float | None = None
    lidarBoresightPitchDeg: float | None = None
    lidarBoresightRollDeg: float | None = None
    lidarComparisonMode: LidarComparisonMode | None = None
    maxLidarRangeM: float | None = None
    pointDensityPtsM2: float | None = None
    useCustomBearing: bool | None = None
    customBearingDeg: float | None = None


class BoundsModel(BaseModel):
    minX: float
    minY: float
    maxX: float
    maxY: float


class LngLatBoundsModel(BaseModel):
    minLng: float
    minLat: float
    maxLng: float
    maxLat: float


class DsmSourceDescriptorModel(BaseModel):
    id: str
    name: str
    fileSizeBytes: int = Field(..., ge=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    sourceBounds: BoundsModel
    footprint3857: BoundsModel
    footprintLngLat: LngLatBoundsModel
    footprintRingLngLat: list[tuple[float, float]]
    sourceCrsCode: str | None = None
    sourceCrsLabel: str
    sourceProj4: str
    horizontalUnits: str | None = None
    verticalScaleToMeters: float = Field(1.0, gt=0)
    noDataValue: float | None = None
    nativeResolutionXM: float | None = Field(default=None, gt=0)
    nativeResolutionYM: float | None = Field(default=None, gt=0)
    validCoverageRatio: float | None = Field(default=None, ge=0, le=1)
    loadedAtIso: str


class TerrainSourceModel(BaseModel):
    mode: TerrainSourceMode = "mapbox"
    datasetId: str | None = None

    @model_validator(mode="after")
    def validate_dataset_requirement(self) -> "TerrainSourceModel":
        if self.mode == "blended" and (self.datasetId is None or not self.datasetId.strip()):
            raise ValueError("datasetId is required when terrainSource.mode is 'blended'.")
        return self


class PartitionSolveRequest(BaseModel):
    polygonId: str | None = None
    ring: list[tuple[float, float]]
    payloadKind: PayloadKind
    params: FlightParamsModel
    terrainSource: TerrainSourceModel = Field(default_factory=TerrainSourceModel)
    altitudeMode: AltitudeMode = "legacy"
    minClearanceM: float = Field(60, ge=0)
    turnExtendM: float = Field(96, ge=0)
    tradeoff: float | None = Field(default=None, ge=0, le=1)
    debug: bool = False

    @model_validator(mode="after")
    def validate_ring_and_payload(self) -> "PartitionSolveRequest":
        if len(self.ring) < 3:
            raise ValueError("Polygon ring must have at least 3 coordinates.")
        if self.params.payloadKind != self.payloadKind:
            self.params.payloadKind = self.payloadKind
        return self


class RegionPreview(BaseModel):
    areaM2: float
    bearingDeg: float
    atomCount: int
    ring: list[tuple[float, float]]
    convexity: float
    compactness: float
    baseAltitudeAGL: float | None = None
    exactScore: float | None = None
    exactSeedBearingDeg: float | None = None


class DebugArtifacts(BaseModel):
    requestId: str
    artifactPaths: list[str] = Field(default_factory=list)


class PartitionSolutionPreviewModel(BaseModel):
    signature: str
    tradeoff: float
    regionCount: int
    totalMissionTimeSec: float
    normalizedQualityCost: float
    weightedMeanMismatchDeg: float
    hierarchyLevel: int
    largestRegionFraction: float
    meanConvexity: float
    boundaryBreakAlignment: float
    isFirstPracticalSplit: bool
    rankingSource: PartitionRankingSource | None = None
    exactScore: float | None = None
    exactQualityCost: float | None = None
    exactMissionTimeSec: float | None = None
    exactMetricKind: ExactMetricKind | None = None
    regions: list[RegionPreview]
    debug: DebugArtifacts | None = None


class PartitionSolveResponse(BaseModel):
    requestId: str
    solutions: list[PartitionSolutionPreviewModel]
    debug: DebugArtifacts | None = None


class DsmStatusResponse(BaseModel):
    datasetId: str | None = None
    descriptor: DsmSourceDescriptorModel | None = None
    processingStatus: DsmProcessingStatus | None = None
    reusedExisting: bool = False
    terrainTileUrlTemplate: str | None = None


class DsmDatasetListResponse(BaseModel):
    datasets: list[DsmStatusResponse] = Field(default_factory=list)


class TerrainBatchTileRequestModel(BaseModel):
    z: int = Field(..., ge=0)
    x: int
    y: int
    padTiles: int = Field(0, ge=0, le=2)


class TerrainBatchRequestModel(BaseModel):
    operation: Literal["terrain-batch"] = "terrain-batch"
    terrainSource: TerrainSourceModel = Field(default_factory=TerrainSourceModel)
    tiles: list[TerrainBatchTileRequestModel] = Field(default_factory=list)


class TerrainBatchTileResponseModel(BaseModel):
    z: int
    x: int
    y: int
    size: int = Field(..., gt=0)
    pngBase64: str
    demPngBase64: str | None = None
    demSize: int | None = Field(default=None, gt=0)
    demPadTiles: int | None = Field(default=None, ge=0)


class TerrainBatchResponseModel(BaseModel):
    operation: Literal["terrain-batch"] = "terrain-batch"
    tiles: list[TerrainBatchTileResponseModel] = Field(default_factory=list)


class ExactOptimizeBearingRequest(BaseModel):
    polygonId: str | None = None
    ring: list[tuple[float, float]]
    payloadKind: PayloadKind
    params: FlightParamsModel
    terrainSource: TerrainSourceModel = Field(default_factory=TerrainSourceModel)
    altitudeMode: AltitudeMode = "legacy"
    minClearanceM: float = Field(60, ge=0)
    turnExtendM: float = Field(96, ge=0)
    seedBearingDeg: float = 0
    mode: Literal["local", "global"] = "global"
    halfWindowDeg: float | None = Field(default=None, gt=0, le=180)

    @model_validator(mode="after")
    def validate_ring_and_payload(self) -> "ExactOptimizeBearingRequest":
        if len(self.ring) < 3:
            raise ValueError("Polygon ring must have at least 3 coordinates.")
        if self.params.payloadKind != self.payloadKind:
            self.params.payloadKind = self.payloadKind
        return self


class ExactOptimizeBearingResponse(BaseModel):
    bearingDeg: float | None = None
    exactScore: float | None = None
    qualityCost: float | None = None
    missionTimeSec: float | None = None
    normalizedTimeCost: float | None = None
    metricKind: ExactMetricKind | None = None
    seedBearingDeg: float
    lineSpacingM: float | None = None
    diagnostics: dict[str, float] = Field(default_factory=dict)
