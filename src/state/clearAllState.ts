export function createCoveragePanelResetState(nowMs: number) {
  return {
    runId: `${nowMs}`,
    running: false,
    autoRetryCount: 0,
    importedPoses: [] as unknown[],
    poseAreaRing: [] as [number, number][],
    overallStats: { gsd: null, density: null },
    perPolygonStats: new Map<string, unknown>(),
    partitionOptionsByPolygon: {} as Record<string, unknown>,
    partitionSelectionByPolygon: {} as Record<string, number>,
    loadingPartitionOptionsIds: {} as Record<string, true>,
    applyingPartitionIds: {} as Record<string, true>,
    exactPartitionPreviewByKey: {} as Record<string, unknown>,
    splittingPolygonIds: {} as Record<string, true>,
    selectedPolygonId: null as string | null,
  };
}

export function createHomeClearAllState() {
  return {
    polygonResults: [] as unknown[],
    analyzingPolygons: new Set<string>(),
    paramsByPolygon: {} as Record<string, unknown>,
    paramsDialog: { open: false, polygonId: null as string | null },
    importedOriginals: {} as Record<string, unknown>,
    overrides: {} as Record<string, unknown>,
    importedPoseCount: 0,
    selectedPolygonId: null as string | null,
    mergeState: {
      mode: 'idle' as const,
      primaryPolygonId: null as string | null,
      selectedPolygonIds: [] as string[],
      eligiblePolygonIds: [] as string[],
      previewRing: null as [number, number][] | null,
      canConfirm: false,
      warning: null as string | null,
    },
    historyState: {
      isApplyingOperation: false,
      canUndo: false,
      canRedo: false,
      undoLabel: undefined as string | undefined,
      redoLabel: undefined as string | undefined,
    },
  };
}

export function shouldConsumeClearAllEpoch(lastHandledEpoch: number, currentEpoch: number) {
  return currentEpoch !== 0 && currentEpoch !== lastHandledEpoch;
}
