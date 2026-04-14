import assert from 'node:assert/strict';

import type { PolygonOperationTransaction, PolygonSnapshot } from '../components/MapFlightDirection/api.ts';
import {
  applyPolygonSnapshotsToMetadata,
  clearPolygonOperationHistory,
  clearPolygonOperationRedo,
  collectAffectedPolygonIds,
  createEmptyPolygonOperationHistory,
  createPolygonFeatureSnapshot,
  createPolygonHistoryState,
  derivePolygonMergeState,
  popRedoPolygonOperation,
  popUndoPolygonOperation,
  pushPolygonOperationTransaction,
} from '../state/polygonOperations.ts';

function square(minX: number, minY: number, maxX: number, maxY: number): [number, number][] {
  return [
    [minX, minY],
    [maxX, minY],
    [maxX, maxY],
    [minX, maxY],
    [minX, minY],
  ];
}

function polygonFeature(id: string, ring: [number, number][], source = 'terrain-face-split') {
  return {
    id,
    properties: { source, name: id },
    ring,
  };
}

function polygonSnapshot(id: string, ring: [number, number][], options?: {
  source?: string;
  altitudeAGL?: number;
  overrideBearingDeg?: number;
  importedOriginalBearingDeg?: number;
}) {
  const feature = createPolygonFeatureSnapshot({
    id,
    ring,
    properties: {
      source: options?.source ?? 'terrain-face-split',
      name: id,
    },
  });
  assert.ok(feature, `expected feature for ${id}`);
  return {
    feature,
    params: {
      payloadKind: 'camera',
      altitudeAGL: options?.altitudeAGL ?? 100,
      frontOverlap: 70,
      sideOverlap: 70,
      cameraKey: 'MAP61_17MM',
    },
    override: typeof options?.overrideBearingDeg === 'number'
      ? {
          bearingDeg: options.overrideBearingDeg,
          lineSpacingM: 25,
          source: 'partition' as const,
        }
      : undefined,
    importedOriginal: typeof options?.importedOriginalBearingDeg === 'number'
      ? {
          bearingDeg: options.importedOriginalBearingDeg,
          lineSpacingM: 30,
        }
      : undefined,
  } satisfies PolygonSnapshot;
}

function testSharedEdgeNeighborIsEligible() {
  const state = derivePolygonMergeState({
    features: [
      polygonFeature('a', square(0, 0, 1, 1)),
      polygonFeature('b', square(1, 0, 2, 1)),
      polygonFeature('c', square(3, 0, 4, 1)),
    ],
    primaryPolygonId: 'a',
    selectedPolygonIds: ['a'],
  });

  assert.equal(state.mode, 'selecting');
  assert.deepEqual(state.selectedPolygonIds, ['a']);
  assert.deepEqual(state.eligiblePolygonIds, ['b']);
  assert.equal(state.canConfirm, false);
  assert.equal(state.warning, null);
}

function testPointTouchOnlyPolygonIsRejected() {
  const state = derivePolygonMergeState({
    features: [
      polygonFeature('a', square(0, 0, 1, 1)),
      polygonFeature('corner', square(1, 1, 2, 2)),
    ],
    primaryPolygonId: 'a',
    selectedPolygonIds: ['a', 'corner'],
  });

  assert.deepEqual(state.selectedPolygonIds, ['a']);
  assert.equal(state.warning, 'Only polygons sharing a boundary can be merged.');
  assert.deepEqual(state.eligiblePolygonIds, []);
}

function testHoleProducingMergeIsRejected() {
  const state = derivePolygonMergeState({
    features: [
      polygonFeature('bottom', square(0, 0, 3, 1)),
      polygonFeature('left', square(0, 1, 1, 2)),
      polygonFeature('top', square(0, 2, 3, 3)),
      polygonFeature('right', square(2, 1, 3, 2)),
    ],
    primaryPolygonId: 'bottom',
    selectedPolygonIds: ['bottom', 'left', 'top', 'right'],
  });

  assert.equal(state.selectedPolygonIds[0], 'bottom');
  assert.equal(state.selectedPolygonIds.length, 3);
  assert.ok(state.warning);
  assert.equal(state.eligiblePolygonIds.length, 0);
  assert.equal(state.canConfirm, true);
}

function testChainedMergeUpdatesEligibility() {
  const first = derivePolygonMergeState({
    features: [
      polygonFeature('a', square(0, 0, 1, 1)),
      polygonFeature('b', square(1, 0, 2, 1)),
      polygonFeature('c', square(2, 0, 3, 1)),
    ],
    primaryPolygonId: 'a',
    selectedPolygonIds: ['a'],
  });
  assert.deepEqual(first.eligiblePolygonIds, ['b']);

  const second = derivePolygonMergeState({
    features: [
      polygonFeature('a', square(0, 0, 1, 1)),
      polygonFeature('b', square(1, 0, 2, 1)),
      polygonFeature('c', square(2, 0, 3, 1)),
    ],
    primaryPolygonId: 'a',
    selectedPolygonIds: ['a', 'b'],
  });
  assert.deepEqual(second.selectedPolygonIds, ['a', 'b']);
  assert.deepEqual(second.eligiblePolygonIds, ['c']);
  assert.equal(second.canConfirm, true);
}

function testSlightlyPerturbedSharedEdgeIsEligible() {
  const state = derivePolygonMergeState({
    features: [
      polygonFeature('a', [
        [16, 48],
        [16.0002, 48],
        [16.0002, 48.0002],
        [16, 48.0002],
        [16, 48],
      ]),
      polygonFeature('b', [
        [16.000200002, 48],
        [16.0004, 48],
        [16.0004, 48.0002],
        [16.000199998, 48.0002],
        [16.000200002, 48],
      ]),
    ],
    primaryPolygonId: 'a',
    selectedPolygonIds: ['a'],
  });

  assert.equal(state.mode, 'selecting');
  assert.deepEqual(state.selectedPolygonIds, ['a']);
  assert.deepEqual(state.eligiblePolygonIds, ['b']);
  assert.equal(state.warning, null);
}

function testNestedReMergedAreaCanMergeBackWithSibling() {
  const features = [
    polygonFeature('merged-a1', [
      [16, 48],
      [16.0002003, 48],
      [16.0002003, 48.0002],
      [16, 48.0002],
      [16, 48],
    ], 'terrain-face-merge'),
    polygonFeature('a2', [
      [16.0002, 48],
      [16.0004, 48],
      [16.0004, 48.0002],
      [16.0002, 48.0002],
      [16.0002, 48],
    ]),
  ];

  const state = derivePolygonMergeState({
    features,
    primaryPolygonId: 'merged-a1',
    selectedPolygonIds: ['merged-a1'],
  });

  assert.equal(state.mode, 'selecting');
  assert.deepEqual(state.selectedPolygonIds, ['merged-a1']);
  assert.deepEqual(state.eligiblePolygonIds, ['a2']);
  assert.equal(state.warning, null);
}

function testSmallBoundaryGapCanStillMergeBack() {
  const features = [
    polygonFeature('merged-a1', [
      [16, 48],
      [16.0001997, 48],
      [16.0001997, 48.0002],
      [16, 48.0002],
      [16, 48],
    ], 'terrain-face-merge'),
    polygonFeature('a2', [
      [16.0002, 48],
      [16.0004, 48],
      [16.0004, 48.0002],
      [16.0002, 48.0002],
      [16.0002, 48],
    ]),
  ];

  const state = derivePolygonMergeState({
    features,
    primaryPolygonId: 'merged-a1',
    selectedPolygonIds: ['merged-a1'],
  });

  assert.equal(state.mode, 'selecting');
  assert.deepEqual(state.selectedPolygonIds, ['merged-a1']);
  assert.deepEqual(state.eligiblePolygonIds, ['a2']);
  assert.equal(state.warning, null);
}

function testSplitMergeUndoRedoPreservesMetadataAndSelection() {
  const parent = polygonSnapshot('parent', square(0, 0, 2, 1), {
    altitudeAGL: 120,
    importedOriginalBearingDeg: 87,
  });
  const childA = polygonSnapshot('child-a', square(0, 0, 1, 1), {
    altitudeAGL: 110,
    overrideBearingDeg: 15,
  });
  const childB = polygonSnapshot('child-b', square(1, 0, 2, 1), {
    altitudeAGL: 115,
    overrideBearingDeg: 25,
  });
  const merged = polygonSnapshot('merged', square(0, 0, 2, 1), {
    source: 'terrain-face-merge',
    altitudeAGL: 110,
  });

  const splitTransaction: PolygonOperationTransaction = {
    kind: 'split',
    label: 'Auto Split Area',
    before: [parent],
    after: [childA, childB],
    selectionBefore: 'parent',
    selectionAfter: 'child-a',
  };
  const mergeTransaction: PolygonOperationTransaction = {
    kind: 'merge',
    label: 'Merge Areas',
    before: [childA, childB],
    after: [merged],
    selectionBefore: 'child-a',
    selectionAfter: 'merged',
  };

  const initialMetadata = applyPolygonSnapshotsToMetadata({
    params: new Map(),
    overrides: new Map(),
    importedOriginals: new Map(),
  }, splitTransaction.before, ['parent']);
  assert.equal(initialMetadata.params.get('parent')?.altitudeAGL, 120);
  assert.equal(initialMetadata.importedOriginals.get('parent')?.bearingDeg, 87);

  const splitMetadata = applyPolygonSnapshotsToMetadata(
    initialMetadata,
    splitTransaction.after,
    collectAffectedPolygonIds(splitTransaction),
  );
  assert.equal(splitMetadata.params.has('parent'), false);
  assert.equal(splitMetadata.params.get('child-a')?.altitudeAGL, 110);
  assert.equal(splitMetadata.overrides.get('child-b')?.bearingDeg, 25);

  const mergedMetadata = applyPolygonSnapshotsToMetadata(
    splitMetadata,
    mergeTransaction.after,
    collectAffectedPolygonIds(mergeTransaction),
  );
  assert.equal(mergedMetadata.params.has('child-a'), false);
  assert.equal(mergedMetadata.params.get('merged')?.altitudeAGL, 110);
  assert.equal(mergedMetadata.overrides.has('merged'), false);

  const undoMergedMetadata = applyPolygonSnapshotsToMetadata(
    mergedMetadata,
    mergeTransaction.before,
    collectAffectedPolygonIds(mergeTransaction),
  );
  assert.equal(undoMergedMetadata.params.get('child-a')?.altitudeAGL, 110);
  assert.equal(undoMergedMetadata.overrides.get('child-a')?.bearingDeg, 15);

  const undoSplitMetadata = applyPolygonSnapshotsToMetadata(
    undoMergedMetadata,
    splitTransaction.before,
    collectAffectedPolygonIds(splitTransaction),
  );
  assert.equal(undoSplitMetadata.params.get('parent')?.altitudeAGL, 120);
  assert.equal(undoSplitMetadata.importedOriginals.get('parent')?.bearingDeg, 87);

  let history = createEmptyPolygonOperationHistory();
  history = pushPolygonOperationTransaction(history, splitTransaction);
  history = pushPolygonOperationTransaction(history, mergeTransaction);
  assert.deepEqual(createPolygonHistoryState(history), {
    isApplyingOperation: false,
    canUndo: true,
    canRedo: false,
    undoLabel: 'Merge Areas',
    redoLabel: undefined,
  });

  const undoMerge = popUndoPolygonOperation(history);
  assert.equal(undoMerge.transaction?.label, 'Merge Areas');
  history = undoMerge.history;

  const undoSplit = popUndoPolygonOperation(history);
  assert.equal(undoSplit.transaction?.label, 'Auto Split Area');
  history = undoSplit.history;

  const redoSplit = popRedoPolygonOperation(history);
  assert.equal(redoSplit.transaction?.label, 'Auto Split Area');
  history = redoSplit.history;

  const redoMerge = popRedoPolygonOperation(history);
  assert.equal(redoMerge.transaction?.label, 'Merge Areas');
  history = redoMerge.history;
  assert.deepEqual(createPolygonHistoryState(clearPolygonOperationHistory()), {
    isApplyingOperation: false,
    canUndo: false,
    canRedo: false,
    undoLabel: undefined,
    redoLabel: undefined,
  });

  const afterUndo = popUndoPolygonOperation(pushPolygonOperationTransaction(createEmptyPolygonOperationHistory(), splitTransaction));
  const branchHistory = pushPolygonOperationTransaction(afterUndo.history, mergeTransaction);
  assert.equal(branchHistory.redoStack.length, 0);
}

function testClearingRedoPreservesUndoHistory() {
  const splitTransaction: PolygonOperationTransaction = {
    kind: 'split',
    label: 'Auto Split Area',
    before: [polygonSnapshot('parent', square(0, 0, 2, 1))],
    after: [
      polygonSnapshot('child-a', square(0, 0, 1, 1)),
      polygonSnapshot('child-b', square(1, 0, 2, 1)),
    ],
    selectionBefore: 'parent',
    selectionAfter: 'child-a',
  };
  const mergeTransaction: PolygonOperationTransaction = {
    kind: 'merge',
    label: 'Merge Areas',
    before: [
      polygonSnapshot('child-a', square(0, 0, 1, 1)),
      polygonSnapshot('child-b', square(1, 0, 2, 1)),
    ],
    after: [polygonSnapshot('merged', square(0, 0, 2, 1), { source: 'terrain-face-merge' })],
    selectionBefore: 'child-a',
    selectionAfter: 'merged',
  };

  let history = pushPolygonOperationTransaction(createEmptyPolygonOperationHistory(), splitTransaction);
  history = pushPolygonOperationTransaction(history, mergeTransaction);
  history = popUndoPolygonOperation(history).history;
  assert.equal(history.undoStack.length, 1);
  assert.equal(history.redoStack.length, 1);

  const cleared = clearPolygonOperationRedo(history);
  assert.equal(cleared.undoStack.length, 1);
  assert.equal(cleared.redoStack.length, 0);
}

testSharedEdgeNeighborIsEligible();
testPointTouchOnlyPolygonIsRejected();
testHoleProducingMergeIsRejected();
testChainedMergeUpdatesEligibility();
testSlightlyPerturbedSharedEdgeIsEligible();
testNestedReMergedAreaCanMergeBackWithSibling();
testSmallBoundaryGapCanStillMergeBack();
testSplitMergeUndoRedoPreservesMetadataAndSelection();
testClearingRedoPreservesUndoHistory();

console.log('polygon_operations.test.ts passed');
