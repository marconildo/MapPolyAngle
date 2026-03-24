import assert from "node:assert/strict";
import { writeArrayBuffer } from "geotiff";

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
const uploadedDescriptor = makeDescriptor("uploaded-dsm", "Uploaded DSM");

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

async function makeSingleBandGeoTiffFile(name: string): Promise<File> {
  const arrayBuffer = await writeArrayBuffer(
    new Float32Array([1, 4, 9, 16]),
    {
      width: 2,
      height: 2,
      SampleFormat: [3],
      BitsPerSample: [32],
      GeographicTypeGeoKey: 4326,
      ModelPixelScale: [0.01, 0.01, 0],
      ModelTiepoint: [0, 0, 0, 7, 47, 0],
    },
  );
  const blob = new Blob([arrayBuffer], { type: "image/tiff" }) as Blob & { name: string; lastModified: number };
  Object.defineProperty(blob, "name", { value: name, configurable: true });
  Object.defineProperty(blob, "lastModified", { value: 0, configurable: true });
  return blob as unknown as File;
}

async function makeRgbaGeoTiffFile(name: string): Promise<File> {
  const arrayBuffer = await writeArrayBuffer(
    [
      [[120, 120], [120, 120]],
      [[80, 80], [80, 80]],
      [[40, 40], [40, 40]],
      [[255, 255], [255, 255]],
    ],
    {
      width: 2,
      height: 2,
      SamplesPerPixel: 4,
      BitsPerSample: [8, 8, 8, 8],
      PhotometricInterpretation: 2,
      ExtraSamples: [2],
      GeographicTypeGeoKey: 4326,
      ModelPixelScale: [0.01, 0.01, 0],
      ModelTiepoint: [0, 0, 0, 7, 47, 0],
    },
  );
  const blob = new Blob([arrayBuffer], { type: "image/tiff" }) as Blob & { name: string; lastModified: number };
  Object.defineProperty(blob, "name", { value: name, configurable: true });
  Object.defineProperty(blob, "lastModified", { value: 0, configurable: true });
  return blob as unknown as File;
}

const fetchCalls: Array<{ url: string; method: string }> = [];

globalThis.window = { localStorage: storage } as typeof window;
globalThis.__TERRAIN_BACKEND_URL_FOR_TESTS__ = backendBaseUrl;
globalThis.fetch = (async (input: string | URL, init?: RequestInit) => {
  const url = String(input);
  const method = init?.method ?? "GET";
  fetchCalls.push({ url, method });
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
  if (url === `${backendBaseUrl}/v1/dsm/prepare-upload` && method === "POST") {
    return mockResponse({
      jsonBody: {
        status: "upload-required",
        uploadId: "terrain-source-upload",
        uploadTarget: {
          url: "https://upload-target.test/terrain-source",
          method: "PUT",
          headers: { "Content-Type": "image/tiff" },
          expiresAtIso: "2026-03-24T13:00:00Z",
        },
      },
    });
  }
  if (url === "https://upload-target.test/terrain-source" && method === "PUT") {
    return mockResponse({ status: 200, textBody: "" });
  }
  if (url === `${backendBaseUrl}/v1/dsm/finalize-upload` && method === "POST") {
    return mockResponse({
      jsonBody: {
        datasetId: uploadedDescriptor.id,
        descriptor: uploadedDescriptor,
        processingStatus: "ready",
        reusedExisting: false,
        terrainTileUrlTemplate: `${backendBaseUrl}/v1/terrain-rgb/{z}/{x}/{y}.png?mode=blended&datasetId=${uploadedDescriptor.id}`,
      },
    });
  }
  throw new Error(`Unexpected fetch: ${method} ${url}`);
}) as typeof fetch;

const terrainSource = await import("../terrain/terrainSource.ts");

async function testSessionRestore() {
  storage.clear();
  fetchCalls.length = 0;
  storage.setItem(
    "terrain-source-selection-v1",
    JSON.stringify({ mode: "blended", selectedDatasetId: "dsm-1", rememberedDescriptor: datasets[0] }),
  );
  terrainSource.__resetTerrainSourceForTests({ clearStorage: false });

  await terrainSource.initializeTerrainSourceState();

  const state = terrainSource.getTerrainSourceState();
  assert.equal(state.source.mode, "mapbox");
  assert.equal(state.descriptor, null);
  assert.equal(state.rememberedDescriptor?.id, "dsm-1");
  assert.equal(terrainSource.getTerrainDemUrlTemplateForCurrentSource(), null);
  assert.equal(fetchCalls.length, 0);
}

async function testApplyingRememberedDsmLoadsBackendOnlyOnDemand() {
  storage.clear();
  fetchCalls.length = 0;
  storage.setItem(
    "terrain-source-selection-v1",
    JSON.stringify({ mode: "blended", selectedDatasetId: "dsm-1", rememberedDescriptor: datasets[0] }),
  );
  terrainSource.__resetTerrainSourceForTests({ clearStorage: false });

  await terrainSource.initializeTerrainSourceState();
  const descriptor = await terrainSource.activateRememberedTerrainSource();

  assert.equal(descriptor.id, "dsm-1");
  const state = terrainSource.getTerrainSourceState();
  assert.equal(state.source.mode, "blended");
  assert.equal(state.source.datasetId, "dsm-1");
  assert.equal(state.descriptor?.id, "dsm-1");
  assert.equal(state.rememberedDescriptor?.id, "dsm-1");
  assert.match(terrainSource.getTerrainDemUrlTemplateForCurrentSource() ?? "", /datasetId=dsm-1/);
  assert.deepEqual(
    fetchCalls.map((call) => [call.method, call.url]),
    [["GET", `${backendBaseUrl}/v1/dsm/datasets/dsm-1`]],
  );
}

async function testUploadingDsmUpdatesTerrainSourceState() {
  storage.clear();
  fetchCalls.length = 0;
  terrainSource.__resetTerrainSourceForTests();

  await terrainSource.initializeTerrainSourceState();
  const descriptor = await terrainSource.loadTerrainSourceFromFile(await makeSingleBandGeoTiffFile("uploaded.tiff"));

  assert.equal(descriptor.id, uploadedDescriptor.id);
  const state = terrainSource.getTerrainSourceState();
  assert.equal(state.source.mode, "blended");
  assert.equal(state.source.datasetId, uploadedDescriptor.id);
  assert.equal(state.descriptor?.id, uploadedDescriptor.id);
  assert.deepEqual(JSON.parse(storage.getItem("terrain-source-selection-v1") ?? "null"), {
    mode: "blended",
    selectedDatasetId: uploadedDescriptor.id,
    rememberedDescriptor: uploadedDescriptor,
  });
  assert.deepEqual(
    fetchCalls
      .filter((call) => call.method !== "GET")
      .map((call) => [call.method, call.url]),
    [
      ["POST", `${backendBaseUrl}/v1/dsm/prepare-upload`],
      ["PUT", "https://upload-target.test/terrain-source"],
      ["POST", `${backendBaseUrl}/v1/dsm/finalize-upload`],
    ],
  );
}

async function testInvalidRgbaGeoTiffIsRejectedBeforeBackendUpload() {
  storage.clear();
  fetchCalls.length = 0;
  terrainSource.__resetTerrainSourceForTests();

  await assert.rejects(
    terrainSource.loadTerrainSourceFromFile(await makeRgbaGeoTiffFile("ortho-rgba.tiff")),
    /single-band elevation geotiff|rgb\/rgba ortho imagery/i,
  );
  assert.equal(fetchCalls.length, 0);
  assert.equal(terrainSource.getTerrainSourceState().source.mode, "mapbox");
  assert.equal(terrainSource.getTerrainSourceState().descriptor, null);
}

await testSessionRestore();
await testApplyingRememberedDsmLoadsBackendOnlyOnDemand();
await testUploadingDsmUpdatesTerrainSourceState();
await testInvalidRgbaGeoTiffIsRejectedBeforeBackendUpload();

console.log("terrain_source.test.ts passed");
