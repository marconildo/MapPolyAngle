// @ts-ignore Turf typings are inconsistent in this repo.
import * as turf from '@turf/turf';

import type {
  BearingOverride,
  PolygonFeatureSnapshot,
  PolygonHistoryState,
  PolygonImportedOriginal,
  PolygonMergeState,
  PolygonOperationTransaction,
  PolygonSnapshot,
} from '@/components/MapFlightDirection/api';
import type { PolygonParams } from '@/components/MapFlightDirection/types';

type PolygonOperationMaps = {
  params: Map<string, PolygonParams>;
  overrides: Map<string, BearingOverride>;
  importedOriginals: Map<string, PolygonImportedOriginal>;
};

type MergeablePolygonFeature = {
  id: string;
  properties: Record<string, any>;
  ring: [number, number][];
};

type PolygonOperationHistory = {
  undoStack: PolygonOperationTransaction[];
  redoStack: PolygonOperationTransaction[];
};

const MERGE_ELIGIBLE_SOURCES = new Set(['terrain-face-split', 'terrain-face-merge']);
const HISTORY_LIMIT = 50;
const AREA_EPSILON_M2 = 0.01;
const SHARED_BOUNDARY_MIN_METERS = 1;
const COORD_EPSILON_DEG = 1e-10;
const SHARED_BOUNDARY_TOLERANCE_KM = 0.0001;
const SHARED_BOUNDARY_SLOP_METERS = 0.25;
const OVERLAP_WIDTH_TOLERANCE_METERS = 0.25;

function roundCoordPair(coord: [number, number]): [number, number] {
  return [Number(coord[0].toFixed(12)), Number(coord[1].toFixed(12))];
}

function areCoordsNear(a: [number, number], b: [number, number], epsilon = COORD_EPSILON_DEG): boolean {
  return Math.abs(a[0] - b[0]) <= epsilon && Math.abs(a[1] - b[1]) <= epsilon;
}

function normalizeRing(ring: [number, number][]): [number, number][] | null {
  const coords = Array.isArray(ring)
    ? ring
        .filter((coord): coord is [number, number] => (
          Array.isArray(coord) &&
          coord.length >= 2 &&
          Number.isFinite(coord[0]) &&
          Number.isFinite(coord[1])
        ))
        .map((coord) => roundCoordPair(coord))
    : [];
  if (coords.length < 3) return null;

  const deduped: [number, number][] = [];
  for (const coord of coords) {
    if (deduped.length === 0 || !areCoordsNear(deduped[deduped.length - 1], coord)) {
      deduped.push(coord);
    }
  }
  if (deduped.length < 3) return null;

  const first = deduped[0];
  const last = deduped[deduped.length - 1];
  if (areCoordsNear(first, last)) {
    deduped[deduped.length - 1] = first;
  } else {
    deduped.push(first);
  }

  return deduped.length >= 4 ? deduped : null;
}

function normalizePolygonFeature(feature: any) {
  if (!feature?.geometry) return feature;
  let next = feature;
  try {
    next = turf.cleanCoords(next as any);
  } catch {}
  try {
    const truncateFn = (turf as any).truncate;
    if (typeof truncateFn === 'function') {
      next = truncateFn(next, { precision: 10, coordinates: 2, mutate: false });
    }
  } catch {}
  try {
    next = turf.cleanCoords(next as any);
  } catch {}
  return next;
}

function unionPolygonFeatures(a: any, b: any) {
  const unionFn = (turf as any).union;
  if (typeof unionFn !== 'function') return null;
  const attempt = (left: any, right: any) => (
    unionFn.length >= 2
      ? unionFn(left, right)
      : unionFn(turf.featureCollection([left, right]))
  );
  try {
    return attempt(a, b);
  } catch {
    try {
      return attempt(normalizePolygonFeature(a), normalizePolygonFeature(b));
    } catch {
      return null;
    }
  }
}

function intersectPolygonFeatures(a: any, b: any) {
  const intersectFn = (turf as any).intersect;
  if (typeof intersectFn !== 'function') return null;
  const attempt = (left: any, right: any) => (
    intersectFn.length >= 2
      ? intersectFn(left, right)
      : intersectFn(turf.featureCollection([left, right]))
  );
  try {
    return attempt(a, b);
  } catch {
    try {
      return attempt(normalizePolygonFeature(a), normalizePolygonFeature(b));
    } catch {
      return null;
    }
  }
}

function polygonFeatureFromRing(ring: [number, number][]) {
  const normalizedRing = normalizeRing(ring);
  if (!normalizedRing) return null;
  return normalizePolygonFeature(turf.polygon([normalizedRing]));
}

function singleOuterRingFromFeature(feature: any): [number, number][] | null {
  if (!feature?.geometry || feature.geometry.type !== 'Polygon') return null;
  if (!Array.isArray(feature.geometry.coordinates) || feature.geometry.coordinates.length !== 1) return null;
  return normalizeRing(feature.geometry.coordinates[0] as [number, number][]) ?? null;
}

function sharedBoundaryMeters(featureA: any, featureB: any): number {
  let strictLengthMeters = 0;
  try {
    const overlap = turf.lineOverlap(featureA, featureB, {
      tolerance: SHARED_BOUNDARY_TOLERANCE_KM,
    });
    strictLengthMeters = overlap.features.reduce((total: number, feature: any) => {
      if (!feature?.geometry) return total;
      return total + (turf.length(feature, { units: 'kilometers' }) * 1000);
    }, 0);
  } catch {
    strictLengthMeters = 0;
  }

  const ringA = singleOuterRingFromFeature(featureA);
  const ringB = singleOuterRingFromFeature(featureB);
  if (!ringA || !ringB) return strictLengthMeters;

  const averageLatitudeDeg = [...ringA, ...ringB].reduce((sum, coord) => sum + coord[1], 0) / (ringA.length + ringB.length);
  const latScale = 111_320;
  const lonScale = Math.max(1e-6, Math.cos((averageLatitudeDeg * Math.PI) / 180)) * 111_320;
  const toLocalMeters = (coord: [number, number]) => ({
    x: coord[0] * lonScale,
    y: coord[1] * latScale,
  });
  const segmentsA = ringA.slice(0, -1).map((start, index) => [toLocalMeters(start), toLocalMeters(ringA[index + 1])] as const);
  const segmentsB = ringB.slice(0, -1).map((start, index) => [toLocalMeters(start), toLocalMeters(ringB[index + 1])] as const);

  const tolerantLengthMeters = segmentsA.reduce((total, [a0, a1]) => {
    const ax = a1.x - a0.x;
    const ay = a1.y - a0.y;
    const aLen = Math.hypot(ax, ay);
    if (aLen <= 1e-6) return total;
    const ux = ax / aLen;
    const uy = ay / aLen;

    let bestOverlap = 0;
    for (const [b0, b1] of segmentsB) {
      const bx = b1.x - b0.x;
      const by = b1.y - b0.y;
      const bLen = Math.hypot(bx, by);
      if (bLen <= 1e-6) continue;
      const vx = bx / bLen;
      const vy = by / bLen;
      const parallelScore = Math.abs(ux * vy - uy * vx);
      if (parallelScore > 0.08) continue;

      const offset0 = Math.abs((b0.x - a0.x) * uy - (b0.y - a0.y) * ux);
      const offset1 = Math.abs((b1.x - a0.x) * uy - (b1.y - a0.y) * ux);
      if (Math.max(offset0, offset1) > SHARED_BOUNDARY_SLOP_METERS) continue;

      const proj0 = (b0.x - a0.x) * ux + (b0.y - a0.y) * uy;
      const proj1 = (b1.x - a0.x) * ux + (b1.y - a0.y) * uy;
      const overlapStart = Math.max(0, Math.min(proj0, proj1));
      const overlapEnd = Math.min(aLen, Math.max(proj0, proj1));
      const overlapLength = overlapEnd - overlapStart;
      if (overlapLength > bestOverlap) bestOverlap = overlapLength;
    }

    return total + bestOverlap;
  }, 0);

  return Math.max(strictLengthMeters, tolerantLengthMeters);
}

function bufferedUnionPolygonFeatures(a: any, b: any) {
  const bufferFn = (turf as any).buffer;
  if (typeof bufferFn !== 'function') return null;
  try {
    const bufferedA = bufferFn(a, SHARED_BOUNDARY_SLOP_METERS, { units: 'meters' });
    const bufferedB = bufferFn(b, SHARED_BOUNDARY_SLOP_METERS, { units: 'meters' });
    const union = unionPolygonFeatures(bufferedA, bufferedB);
    if (!union) return null;
    const unbuffered = bufferFn(union, -SHARED_BOUNDARY_SLOP_METERS, { units: 'meters' });
    return normalizePolygonFeature(unbuffered);
  } catch {
    return null;
  }
}

function isMergeEligibleSource(source: unknown): boolean {
  return typeof source === 'string' && MERGE_ELIGIBLE_SOURCES.has(source);
}

function uniqueOrderedPolygonIds(primaryPolygonId: string, selectedPolygonIds: string[]) {
  const unique = new Set<string>();
  unique.add(primaryPolygonId);
  for (const polygonId of selectedPolygonIds) {
    if (polygonId && polygonId !== primaryPolygonId) unique.add(polygonId);
  }
  return Array.from(unique);
}

function evaluateMergeCandidate(
  unionFeature: any,
  candidateFeature: any,
) {
  const boundaryMeters = sharedBoundaryMeters(unionFeature, candidateFeature);
  if (boundaryMeters < SHARED_BOUNDARY_MIN_METERS) {
    return { ok: false as const, reason: 'Only polygons sharing a boundary can be merged.' };
  }

  const intersection = intersectPolygonFeatures(unionFeature, candidateFeature);
  const intersectionArea = intersection ? turf.area(intersection) : 0;
  const overlapWidthMeters = boundaryMeters > 1e-6 ? intersectionArea / boundaryMeters : Number.POSITIVE_INFINITY;
  if (intersectionArea > AREA_EPSILON_M2 && overlapWidthMeters > OVERLAP_WIDTH_TOLERANCE_METERS) {
    return { ok: false as const, reason: 'Overlapping polygons cannot be merged.' };
  }

  let union = unionPolygonFeatures(unionFeature, candidateFeature);
  let ring = singleOuterRingFromFeature(union);
  if (!union || !ring) {
    union = bufferedUnionPolygonFeatures(unionFeature, candidateFeature);
    ring = singleOuterRingFromFeature(union);
  }
  if (!union || !ring) {
    return { ok: false as const, reason: 'Merge would create holes or multiple polygons.' };
  }

  return {
    ok: true as const,
    union,
    ring,
    sharedBoundaryMeters: boundaryMeters,
  };
}

export function createIdlePolygonMergeState(): PolygonMergeState {
  return {
    mode: 'idle',
    primaryPolygonId: null,
    selectedPolygonIds: [],
    eligiblePolygonIds: [],
    previewRing: null,
    canConfirm: false,
    warning: null,
  };
}

export function createPolygonHistoryState(
  history?: PolygonOperationHistory,
  isApplyingOperation = false,
): PolygonHistoryState {
  const undoHead = history?.undoStack?.[history.undoStack.length - 1];
  const redoHead = history?.redoStack?.[history.redoStack.length - 1];
  return {
    isApplyingOperation,
    canUndo: !!undoHead,
    canRedo: !!redoHead,
    undoLabel: undoHead?.label,
    redoLabel: redoHead?.label,
  };
}

export function createEmptyPolygonOperationHistory(): PolygonOperationHistory {
  return {
    undoStack: [],
    redoStack: [],
  };
}

export function pushPolygonOperationTransaction(
  history: PolygonOperationHistory,
  transaction: PolygonOperationTransaction,
): PolygonOperationHistory {
  const undoStack = [...history.undoStack, transaction];
  const trimmedUndoStack = undoStack.length > HISTORY_LIMIT
    ? undoStack.slice(undoStack.length - HISTORY_LIMIT)
    : undoStack;
  return {
    undoStack: trimmedUndoStack,
    redoStack: [],
  };
}

export function clearPolygonOperationHistory(): PolygonOperationHistory {
  return createEmptyPolygonOperationHistory();
}

export function clearPolygonOperationRedo(history: PolygonOperationHistory): PolygonOperationHistory {
  if (history.redoStack.length === 0) return history;
  return {
    undoStack: [...history.undoStack],
    redoStack: [],
  };
}

export function popUndoPolygonOperation(history: PolygonOperationHistory) {
  if (history.undoStack.length === 0) {
    return { history, transaction: null as PolygonOperationTransaction | null };
  }
  const undoStack = [...history.undoStack];
  const transaction = undoStack.pop() ?? null;
  return {
    transaction,
    history: transaction
      ? { undoStack, redoStack: [...history.redoStack, transaction] }
      : history,
  };
}

export function popRedoPolygonOperation(history: PolygonOperationHistory) {
  if (history.redoStack.length === 0) {
    return { history, transaction: null as PolygonOperationTransaction | null };
  }
  const redoStack = [...history.redoStack];
  const transaction = redoStack.pop() ?? null;
  return {
    transaction,
    history: transaction
      ? { undoStack: [...history.undoStack, transaction], redoStack }
      : history,
  };
}

export function collectAffectedPolygonIds(transaction: Pick<PolygonOperationTransaction, 'before' | 'after'>) {
  const ids = new Set<string>();
  for (const snapshot of transaction.before) ids.add(snapshot.feature.id);
  for (const snapshot of transaction.after) ids.add(snapshot.feature.id);
  return Array.from(ids);
}

export function applyPolygonSnapshotsToMetadata(
  state: PolygonOperationMaps,
  snapshots: PolygonSnapshot[],
  affectedPolygonIds: string[],
): PolygonOperationMaps {
  const params = new Map(state.params);
  const overrides = new Map(state.overrides);
  const importedOriginals = new Map(state.importedOriginals);

  for (const polygonId of affectedPolygonIds) {
    params.delete(polygonId);
    overrides.delete(polygonId);
    importedOriginals.delete(polygonId);
  }

  for (const snapshot of snapshots) {
    const polygonId = snapshot.feature.id;
    if (snapshot.params) params.set(polygonId, snapshot.params);
    if (snapshot.override) overrides.set(polygonId, snapshot.override);
    if (snapshot.importedOriginal) importedOriginals.set(polygonId, snapshot.importedOriginal);
  }

  return { params, overrides, importedOriginals };
}

export function clonePolygonFeatureSnapshot(feature: PolygonFeatureSnapshot): PolygonFeatureSnapshot {
  const ring = feature.geometry.coordinates?.[0] ?? [];
  return {
    type: 'Feature',
    id: feature.id,
    properties: { ...(feature.properties ?? {}) },
    geometry: {
      type: 'Polygon',
      coordinates: [[...ring.map((coord) => [coord[0], coord[1]] as [number, number])]],
    },
  };
}

export function clonePolygonSnapshot(snapshot: PolygonSnapshot): PolygonSnapshot {
  return {
    feature: clonePolygonFeatureSnapshot(snapshot.feature),
    params: snapshot.params ? { ...snapshot.params } : undefined,
    override: snapshot.override ? { ...snapshot.override } : undefined,
    importedOriginal: snapshot.importedOriginal ? { ...snapshot.importedOriginal } : undefined,
  };
}

export function createPolygonFeatureSnapshot(args: {
  id: string;
  ring: [number, number][];
  properties?: Record<string, any>;
}): PolygonFeatureSnapshot | null {
  const normalizedRing = normalizeRing(args.ring);
  if (!normalizedRing) return null;
  return {
    type: 'Feature',
    id: args.id,
    properties: { ...(args.properties ?? {}) },
    geometry: {
      type: 'Polygon',
      coordinates: [normalizedRing],
    },
  };
}

export function derivePolygonMergeState(options: {
  features: MergeablePolygonFeature[];
  primaryPolygonId: string | null;
  selectedPolygonIds: string[];
}): PolygonMergeState {
  const { features, primaryPolygonId } = options;
  if (!primaryPolygonId) return createIdlePolygonMergeState();

  const featureById = new Map(features.map((feature) => [feature.id, feature]));
  const primary = featureById.get(primaryPolygonId);
  if (!primary) {
    return {
      ...createIdlePolygonMergeState(),
      mode: 'selecting',
      primaryPolygonId,
      warning: 'Selected polygon is no longer available.',
    };
  }
  if (!isMergeEligibleSource(primary.properties?.source)) {
    return {
      ...createIdlePolygonMergeState(),
      mode: 'selecting',
      primaryPolygonId,
      selectedPolygonIds: [primaryPolygonId],
      previewRing: normalizeRing(primary.ring),
      warning: 'Only autosplit-derived polygons can be merged.',
    };
  }

  const orderedSelectedIds = uniqueOrderedPolygonIds(primaryPolygonId, options.selectedPolygonIds);
  const primaryFeature = polygonFeatureFromRing(primary.ring);
  const primaryRing = normalizeRing(primary.ring);
  if (!primaryFeature || !primaryRing) {
    return {
      ...createIdlePolygonMergeState(),
      mode: 'selecting',
      primaryPolygonId,
      selectedPolygonIds: [primaryPolygonId],
      warning: 'Selected polygon is invalid for merging.',
    };
  }

  let warning: string | null = null;
  let unionFeature = primaryFeature;
  let previewRing = primaryRing;
  const appliedSelectedIds = [primaryPolygonId];

  for (const polygonId of orderedSelectedIds.slice(1)) {
    const candidate = featureById.get(polygonId);
    if (!candidate || !isMergeEligibleSource(candidate.properties?.source)) {
      warning = 'Only autosplit-derived polygons can be merged.';
      continue;
    }
    const candidateFeature = polygonFeatureFromRing(candidate.ring);
    if (!candidateFeature) {
      warning = 'One of the selected polygons is invalid for merging.';
      continue;
    }
    const evaluation = evaluateMergeCandidate(unionFeature, candidateFeature);
    if (!evaluation.ok) {
      warning = evaluation.reason;
      continue;
    }
    appliedSelectedIds.push(polygonId);
    unionFeature = evaluation.union;
    previewRing = evaluation.ring;
  }

  const eligiblePolygonIds: string[] = [];
  for (const feature of features) {
    if (appliedSelectedIds.includes(feature.id)) continue;
    if (!isMergeEligibleSource(feature.properties?.source)) continue;
    const candidateFeature = polygonFeatureFromRing(feature.ring);
    if (!candidateFeature) continue;
    const evaluation = evaluateMergeCandidate(unionFeature, candidateFeature);
    if (evaluation.ok) eligiblePolygonIds.push(feature.id);
  }

  if (!warning && eligiblePolygonIds.length === 0 && appliedSelectedIds.length === 1) {
    warning = 'No touching autosplit polygons are available to merge.';
  }

  return {
    mode: 'selecting',
    primaryPolygonId,
    selectedPolygonIds: appliedSelectedIds,
    eligiblePolygonIds,
    previewRing,
    canConfirm: appliedSelectedIds.length > 1,
    warning,
  };
}

export function toMergeablePolygonFeature(feature: any): MergeablePolygonFeature | null {
  const id = String(feature?.id ?? '');
  const ring = normalizeRing(feature?.geometry?.coordinates?.[0] as [number, number][]);
  if (!id || feature?.geometry?.type !== 'Polygon' || !ring) return null;
  return {
    id,
    properties: { ...(feature?.properties ?? {}) },
    ring,
  };
}
