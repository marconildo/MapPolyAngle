import assert from "node:assert/strict";

import type { DsmSourceDescriptor } from "../terrain/types.ts";

class MemoryStorage {
  private readonly store = new Map<string, string>();

  getItem(key: string) {
    return this.store.has(key) ? this.store.get(key)! : null;
  }

  setItem(key: string, value: string) {
    this.store.set(key, String(value));
  }

  removeItem(key: string) {
    this.store.delete(key);
  }

  clear() {
    this.store.clear();
  }
}

const storage = new MemoryStorage();
const backendBaseUrl = "http://terrain-backend.test";

function makeDescriptor(id: string, name: string): DsmSourceDescriptor {
  return {
    id,
    name,
    fileSizeBytes: 1024,
    width: 128,
    height: 64,
    sourceBounds: { minX: 0, minY: 0, maxX: 128, maxY: 64 },
    footprint3857: { minX: 0, minY: 0, maxX: 128, maxY: 64 },
    footprintLngLat: { minLng: 7, minLat: 47, maxLng: 7.01, maxLat: 47.01 },
    footprintRingLngLat: [[7, 47], [7.01, 47], [7.01, 47.01], [7, 47.01], [7, 47]],
    sourceCrsCode: "EPSG:3857",
    sourceCrsLabel: "EPSG:3857",
    sourceProj4: "EPSG:3857",
    horizontalUnits: "metre",
    verticalScaleToMeters: 1,
    noDataValue: null,
    nativeResolutionXM: 1,
    nativeResolutionYM: 1,
    validCoverageRatio: 0.75,
    loadedAtIso: id === "dsm-1" ? "2026-03-21T10:00:00Z" : "2026-03-21T11:00:00Z",
  };
}

const datasets = [makeDescriptor("dsm-1", "First DSM"), makeDescriptor("dsm-2", "Second DSM")];

type MockResponseInit = {
  ok?: boolean;
  status?: number;
  jsonBody?: unknown;
  textBody?: string;
};

function mockResponse(init: MockResponseInit) {
  return {
    ok: init.ok ?? true,
    status: init.status ?? 200,
    async json() {
      return init.jsonBody;
    },
    async text() {
      return init.textBody ?? JSON.stringify(init.jsonBody ?? "");
    },
  };
}

const fetchCalls: string[] = [];

globalThis.window = { localStorage: storage } as typeof window;
globalThis.__TERRAIN_BACKEND_URL_FOR_TESTS__ = backendBaseUrl;
globalThis.fetch = (async (input: string | URL) => {
  const url = String(input);
  fetchCalls.push(url);
  if (url === `${backendBaseUrl}/v1/dsm/datasets`) {
    return mockResponse({
      jsonBody: {
        datasets: datasets.map((descriptor) => ({
          datasetId: descriptor.id,
          descriptor,
          processingStatus: "ready",
          reusedExisting: true,
          terrainTileUrlTemplate: `${backendBaseUrl}/v1/terrain-rgb/{z}/{x}/{y}.png?mode=blended&datasetId=${descriptor.id}`,
        })),
      },
    });
  }
  if (url.startsWith(`${backendBaseUrl}/v1/dsm/datasets/`)) {
    const datasetId = decodeURIComponent(url.slice(`${backendBaseUrl}/v1/dsm/datasets/`.length));
    const descriptor = datasets.find((candidate) => candidate.id === datasetId);
    if (!descriptor) {
      return mockResponse({ ok: false, status: 404, textBody: "not found" });
    }
    return mockResponse({
      jsonBody: {
        datasetId: descriptor.id,
        descriptor,
        processingStatus: "ready",
        reusedExisting: true,
        terrainTileUrlTemplate: `${backendBaseUrl}/v1/terrain-rgb/{z}/{x}/{y}.png?mode=blended&datasetId=${descriptor.id}`,
      },
    });
  }
  throw new Error(`Unexpected fetch: ${url}`);
}) as typeof fetch;

const terrainSource = await import("../terrain/terrainSource.ts");

async function testSessionRestore() {
  storage.clear();
  fetchCalls.length = 0;
  storage.setItem("terrain-source-selection-v1", JSON.stringify({ mode: "blended", selectedDatasetId: "dsm-1" }));
  terrainSource.__resetTerrainSourceForTests({ clearStorage: false });

  await terrainSource.initializeTerrainSourceState();

  const state = terrainSource.getTerrainSourceState();
  assert.equal(state.source.mode, "blended");
  assert.equal(state.descriptor?.id, "dsm-1");
  assert.equal(state.datasets.length, 2);
  assert.match(terrainSource.getTerrainDemUrlTemplateForCurrentSource() ?? "", /datasetId=dsm-1/);
  assert.ok(fetchCalls.includes(`${backendBaseUrl}/v1/dsm/datasets`));
}

async function testSavedDsmSelectionAndPersistence() {
  storage.clear();
  fetchCalls.length = 0;
  terrainSource.__resetTerrainSourceForTests();

  await terrainSource.initializeTerrainSourceState();
  let state = terrainSource.getTerrainSourceState();
  assert.equal(state.source.mode, "mapbox");
  assert.equal(state.descriptor, null);

  const selected = await terrainSource.selectTerrainSourceDataset("dsm-2");
  assert.equal(selected.id, "dsm-2");
  state = terrainSource.getTerrainSourceState();
  assert.equal(state.source.mode, "blended");
  assert.equal(state.descriptor?.id, "dsm-2");
  assert.equal(
    storage.getItem("terrain-source-selection-v1"),
    JSON.stringify({ mode: "blended", selectedDatasetId: "dsm-2" }),
  );

  terrainSource.clearTerrainSourceSelection();
  state = terrainSource.getTerrainSourceState();
  assert.equal(state.source.mode, "mapbox");
  assert.equal(state.descriptor, null);
  assert.equal(
    storage.getItem("terrain-source-selection-v1"),
    JSON.stringify({ mode: "mapbox", selectedDatasetId: null }),
  );
}

await testSessionRestore();
await testSavedDsmSelectionAndPersistence();

console.log("terrain_source.test.ts passed");
