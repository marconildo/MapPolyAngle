from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

PayloadKind = Literal["camera", "lidar"]
AltitudeMode = Literal["legacy", "min-clearance"]
LidarReturnMode = Literal["single", "dual", "triple"]
LidarComparisonMode = Literal["first-return", "all-returns"]
TerrainSourceMode = Literal["mapbox", "blended"]
DsmProcessingStatus = Literal["ready"]


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
