import assert from "node:assert/strict";

import { clearAllOverlays, clearRunOverlays } from "../overlap/overlay.ts";

type MockLayer = { id: string };

function createMockMap() {
  const layers: MockLayer[] = [
    { id: "background" },
    { id: "ogsd-123-density-14-1-1" },
    { id: "ogsd-123-density-14-1-2" },
    { id: "ogsd-999-gsd-14-2-2" },
  ];
  const sources = new Map<string, { id: string }>([
    ["background", { id: "background" }],
    ["ogsd-123-density-14-1-1", { id: "ogsd-123-density-14-1-1" }],
    ["ogsd-123-density-14-1-2", { id: "ogsd-123-density-14-1-2" }],
    ["ogsd-999-gsd-14-2-2", { id: "ogsd-999-gsd-14-2-2" }],
  ]);
  const removedLayers: string[] = [];
  const removedSources: string[] = [];

  const map = {
    isStyleLoaded: () => false,
    getStyle: () => ({
      layers: [...layers],
      sources: Object.fromEntries(Array.from(sources.entries())),
    }),
    getLayer: (id: string) => layers.find((layer) => layer.id === id) ?? null,
    removeLayer: (id: string) => {
      removedLayers.push(id);
      const index = layers.findIndex((layer) => layer.id === id);
      if (index >= 0) layers.splice(index, 1);
    },
    getSource: (id: string) => (sources.has(id) ? sources.get(id) : null),
    removeSource: (id: string) => {
      removedSources.push(id);
      sources.delete(id);
    },
  };

  return {
    map: map as any,
    getRemainingOverlayIds: () => layers.filter((layer) => layer.id.startsWith("ogsd-")).map((layer) => layer.id),
    getRemovedLayers: () => removedLayers,
    getRemovedSources: () => removedSources,
  };
}

function testClearAllOverlaysIgnoresStyleLoadedFlag() {
  const mock = createMockMap();

  clearAllOverlays(mock.map);

  assert.deepEqual(mock.getRemainingOverlayIds(), []);
  assert.deepEqual(
    mock.getRemovedLayers(),
    ["ogsd-123-density-14-1-1", "ogsd-123-density-14-1-2", "ogsd-999-gsd-14-2-2"],
  );
  assert.deepEqual(
    mock.getRemovedSources(),
    ["ogsd-123-density-14-1-1", "ogsd-123-density-14-1-2", "ogsd-999-gsd-14-2-2"],
  );
}

function testClearRunOverlaysRemovesOnlyMatchingRun() {
  const mock = createMockMap();

  clearRunOverlays(mock.map, "123");

  assert.deepEqual(mock.getRemainingOverlayIds(), ["ogsd-999-gsd-14-2-2"]);
  assert.deepEqual(
    mock.getRemovedLayers(),
    ["ogsd-123-density-14-1-1", "ogsd-123-density-14-1-2"],
  );
  assert.deepEqual(
    mock.getRemovedSources(),
    ["ogsd-123-density-14-1-1", "ogsd-123-density-14-1-2"],
  );
}

testClearAllOverlaysIgnoresStyleLoadedFlag();
testClearRunOverlaysRemovesOnlyMatchingRun();

console.log("overlay_clear.test.ts passed");
