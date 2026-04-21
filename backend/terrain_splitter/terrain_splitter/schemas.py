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
DsmPrepareUploadStatus = Literal["existing", "upload-required"]
MissionSolveMode = Literal["exact-dp", "greedy-fallback"]
LoiterDirection = Literal["climb", "descent"]
ManeuverDirection = Literal["clockwise", "counterclockwise"]


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
    turnExtendM: float = Field(96, ge=0)  # deprecated compatibility field; solver ignores it
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


class DsmPrepareUploadRequest(BaseModel):
    sha256: str = Field(..., min_length=64, max_length=64)
    fileSizeBytes: int = Field(..., ge=0)
    originalName: str = Field(..., min_length=1)
    contentType: str | None = None

    @model_validator(mode="after")
    def validate_sha256(self) -> "DsmPrepareUploadRequest":
        normalized = self.sha256.strip().lower()
        if len(normalized) != 64 or any(ch not in "0123456789abcdef" for ch in normalized):
            raise ValueError("sha256 must be a 64-character lowercase hexadecimal string.")
        self.sha256 = normalized
        self.originalName = self.originalName.strip()
        if not self.originalName:
            raise ValueError("originalName is required.")
        if self.contentType is not None:
            self.contentType = self.contentType.strip() or None
        return self


class DsmUploadTargetModel(BaseModel):
    url: str
    method: Literal["PUT"] = "PUT"
    headers: dict[str, str] = Field(default_factory=dict)
    expiresAtIso: str


class DsmPrepareUploadResponse(BaseModel):
    status: DsmPrepareUploadStatus
    dataset: DsmStatusResponse | None = None
    uploadId: str | None = None
    uploadTarget: DsmUploadTargetModel | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "DsmPrepareUploadResponse":
        if self.status == "existing":
            if self.dataset is None:
                raise ValueError("dataset is required when status is 'existing'.")
            self.uploadId = None
            self.uploadTarget = None
            return self
        if self.uploadId is None or self.uploadTarget is None:
            raise ValueError("uploadId and uploadTarget are required when status is 'upload-required'.")
        self.dataset = None
        return self


class DsmFinalizeUploadRequest(BaseModel):
    uploadId: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_upload_id(self) -> "DsmFinalizeUploadRequest":
        self.uploadId = self.uploadId.strip()
        if not self.uploadId:
            raise ValueError("uploadId is required.")
        return self


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
    turnExtendM: float = Field(96, ge=0)  # deprecated compatibility field; solver ignores it
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


class MissionAreaTraversalRequestModel(BaseModel):
    altitudeAGL: float = Field(..., gt=0)
    startPoint: tuple[float, float]
    endPoint: tuple[float, float]
    startTerrainElevationWgs84M: float
    endTerrainElevationWgs84M: float
    startAltitudeWgs84M: float
    endAltitudeWgs84M: float
    leadIn: "MissionTraversalLoiterModel"
    leadOut: "MissionTraversalLoiterModel"


class MissionTraversalLoiterModel(BaseModel):
    centerPoint: tuple[float, float]
    radiusM: float = Field(..., gt=0)
    direction: ManeuverDirection


class MissionAreaRequest(BaseModel):
    polygonId: str = Field(..., min_length=1)
    ring: list[tuple[float, float]]
    bearingDeg: float
    payloadKind: PayloadKind
    params: FlightParamsModel
    forwardTraversal: MissionAreaTraversalRequestModel | None = None
    flippedTraversal: MissionAreaTraversalRequestModel | None = None

    @model_validator(mode="after")
    def validate_ring_and_payload(self) -> "MissionAreaRequest":
        if len(self.ring) < 3:
            raise ValueError("Area ring must have at least 3 coordinates.")
        if self.params.payloadKind != self.payloadKind:
            self.params.payloadKind = self.payloadKind
        self.polygonId = self.polygonId.strip()
        if not self.polygonId:
            raise ValueError("polygonId is required.")
        if (self.forwardTraversal is None) != (self.flippedTraversal is None):
            raise ValueError("forwardTraversal and flippedTraversal must be provided together.")
        return self


class MissionSequenceEndpointModel(BaseModel):
    point: tuple[float, float]
    altitudeWgs84M: float
    headingDeg: float | None = None
    loiterRadiusM: float | None = Field(default=None, gt=0)


class MissionOptimizeAreaSequenceRequest(BaseModel):
    areas: list[MissionAreaRequest]
    terrainSource: TerrainSourceModel = Field(default_factory=TerrainSourceModel)
    altitudeMode: AltitudeMode = "legacy"
    minClearanceM: float = Field(60, ge=0)
    turnExtendM: float = Field(96, ge=0)  # deprecated compatibility field; current optimizer ignores it
    maxHeightAboveGroundM: float = Field(120, gt=0)
    exactSearchMaxAreas: int = Field(17, ge=1, le=17)
    transferCost: "MissionTransferCostModel" = Field(default_factory=lambda: MissionTransferCostModel())
    startEndpoint: MissionSequenceEndpointModel | None = None
    endEndpoint: MissionSequenceEndpointModel | None = None

    @model_validator(mode="after")
    def validate_areas(self) -> "MissionOptimizeAreaSequenceRequest":
        if len(self.areas) == 0:
            raise ValueError("At least one area is required.")
        polygon_ids = [area.polygonId for area in self.areas]
        if len(set(polygon_ids)) != len(polygon_ids):
            raise ValueError("polygonId values must be unique.")
        return self


class MissionAreaTraversalModel(BaseModel):
    polygonId: str
    orderIndex: int = Field(..., ge=0)
    flipped: bool = False
    bearingDeg: float
    startPoint: tuple[float, float]
    endPoint: tuple[float, float]
    startAltitudeWgs84M: float
    endAltitudeWgs84M: float


class MissionConnectionLoiterStepModel(BaseModel):
    point: tuple[float, float]
    targetAltitudeWgs84M: float
    terrainElevationWgs84M: float
    heightAboveGroundM: float
    direction: LoiterDirection
    loopCount: int | None = Field(default=None, ge=1)


class MissionConnectionModel(BaseModel):
    fromPolygonId: str
    toPolygonId: str
    connectionMode: Literal["wic", "wic-adjusted", "stepped-fallback", "direct-fallback"] = "wic"
    fromFlipped: bool = False
    toFlipped: bool = False
    line: list[tuple[float, float]]
    trajectory: list[tuple[float, float]]
    trajectory3D: list[tuple[float, float, float]] = Field(default_factory=list)
    loiterSteps: list[MissionConnectionLoiterStepModel] = Field(default_factory=list)
    requestedMaxHeightAboveGroundM: float = Field(..., gt=0)
    transferDistanceM: float = Field(..., ge=0)
    transferTimeSec: float = Field(..., ge=0)
    transferCost: float = Field(..., ge=0)
    transferMinClearanceM: float = Field(..., ge=0)
    startAltitudeWgs84M: float
    endAltitudeWgs84M: float
    resolvedMaxHeightAboveGroundM: float = Field(..., gt=0)
    transferHorizontalDistanceM: float = Field(..., ge=0)
    transferClimbM: float = Field(..., ge=0)
    transferDescentM: float = Field(..., ge=0)
    transferHorizontalTimeSec: float = Field(..., ge=0)
    transferClimbTimeSec: float = Field(..., ge=0)
    transferDescentTimeSec: float = Field(..., ge=0)
    transferHorizontalSpeedMps: float = Field(..., gt=0)
    transferClimbRateMps: float = Field(..., gt=0)
    transferDescentRateMps: float = Field(..., gt=0)
    transferHorizontalEnergyRate: float = Field(..., ge=0)
    transferClimbEnergyRate: float = Field(..., ge=0)
    transferDescentEnergyRate: float = Field(..., ge=0)


class MissionTransferCostModel(BaseModel):
    mode: Literal["fixed-wing-energy"] = "fixed-wing-energy"
    horizontalSpeedMps: float | None = Field(default=None, gt=0)
    climbRateMps: float = Field(4.0, gt=0)
    descentRateMps: float = Field(6.0, gt=0)
    horizontalEnergyRate: float = Field(1.0, ge=0)
    climbEnergyRate: float = Field(2.5, ge=0)
    descentEnergyRate: float = Field(0.6, ge=0)


class MissionOptimizeAreaSequenceResponse(BaseModel):
    requestId: str
    solveMode: MissionSolveMode
    solvedExactly: bool
    areas: list[MissionAreaTraversalModel]
    connections: list[MissionConnectionModel] = Field(default_factory=list)
    startConnection: MissionConnectionModel | None = None
    endConnection: MissionConnectionModel | None = None
    totalTransferDistanceM: float = Field(..., ge=0)
    totalTransferTimeSec: float = Field(..., ge=0)
    totalTransferCost: float = Field(..., ge=0)


MissionOptimizeAreaSequenceRequest.model_rebuild()
