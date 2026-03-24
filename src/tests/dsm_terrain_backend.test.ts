import assert from 'node:assert/strict';

import type { DsmSourceDescriptor } from '../terrain/types.ts';

const backendBaseUrl = 'http://terrain-backend.test';

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
    sourceCrsCode: 'EPSG:3857',
    sourceCrsLabel: 'EPSG:3857',
    sourceProj4: 'EPSG:3857',
    horizontalUnits: 'metre',
    verticalScaleToMeters: 1,
    noDataValue: null,
    nativeResolutionXM: 1,
    nativeResolutionYM: 1,
    validCoverageRatio: 0.75,
    loadedAtIso: '2026-03-24T12:00:00Z',
  };
}

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
      return init.textBody ?? JSON.stringify(init.jsonBody ?? '');
    },
  };
}

const fetchCalls: Array<{ url: string; method: string; body?: unknown; headers?: HeadersInit }> = [];
const existingDataset = {
  datasetId: 'existing-dsm',
  descriptor: makeDescriptor('existing-dsm', 'Existing DSM'),
  processingStatus: 'ready' as const,
  reusedExisting: true,
  terrainTileUrlTemplate: `${backendBaseUrl}/v1/terrain-rgb/{z}/{x}/{y}.png?mode=blended&datasetId=existing-dsm`,
};
const finalizedDataset = {
  datasetId: 'new-dsm',
  descriptor: makeDescriptor('new-dsm', 'Uploaded DSM'),
  processingStatus: 'ready' as const,
  reusedExisting: false,
  terrainTileUrlTemplate: `${backendBaseUrl}/v1/terrain-rgb/{z}/{x}/{y}.png?mode=blended&datasetId=new-dsm`,
};

function makeFile(bytes: number[], name: string, type: string): File {
  const blob = new Blob([new Uint8Array(bytes)], { type }) as Blob & { name: string; lastModified: number };
  Object.defineProperty(blob, 'name', { value: name, configurable: true });
  Object.defineProperty(blob, 'lastModified', { value: 0, configurable: true });
  return blob as unknown as File;
}

globalThis.__TERRAIN_BACKEND_URL_FOR_TESTS__ = backendBaseUrl;
globalThis.fetch = (async (input: string | URL, init?: RequestInit) => {
  const url = String(input);
  const method = init?.method ?? 'GET';
  fetchCalls.push({ url, method, body: init?.body, headers: init?.headers });

  if (url === `${backendBaseUrl}/v1/dsm/prepare-upload` && method === 'POST') {
    const rawBody = typeof init?.body === 'string' ? JSON.parse(init.body) : null;
    if (rawBody?.originalName === 'existing.tiff') {
      return mockResponse({ jsonBody: { status: 'existing', dataset: existingDataset } });
    }
    if (rawBody?.originalName === 'upload-fail.tiff') {
      return mockResponse({
        jsonBody: {
          status: 'upload-required',
          uploadId: 'upload-fail-session',
          uploadTarget: {
            url: 'https://upload-target.test/upload-fail',
            method: 'PUT',
            headers: { 'Content-Type': 'image/tiff' },
            expiresAtIso: '2026-03-24T13:00:00Z',
          },
        },
      });
    }
    return mockResponse({
      jsonBody: {
        status: 'upload-required',
        uploadId: 'upload-session',
        uploadTarget: {
          url: 'https://upload-target.test/staged',
          method: 'PUT',
          headers: { 'Content-Type': 'image/tiff' },
          expiresAtIso: '2026-03-24T13:00:00Z',
        },
      },
    });
  }

  if (url === 'https://upload-target.test/staged' && method === 'PUT') {
    return mockResponse({ status: 200, textBody: '' });
  }
  if (url === 'https://upload-target.test/upload-fail' && method === 'PUT') {
    return mockResponse({ ok: false, status: 500, textBody: 'presigned upload failed' });
  }
  if (url === `${backendBaseUrl}/v1/dsm/finalize-upload` && method === 'POST') {
    return mockResponse({ jsonBody: finalizedDataset });
  }

  throw new Error(`Unexpected fetch: ${method} ${url}`);
}) as typeof fetch;

const service = await import('../services/dsmTerrainBackend.ts');

async function testExistingDsmSkipsUploadAndFinalize() {
  fetchCalls.length = 0;
  const file = makeFile([1, 2, 3, 4], 'existing.tiff', 'image/tiff');
  const dataset = await service.uploadDsmToTerrainBackend(file);

  assert.equal(dataset.datasetId, existingDataset.datasetId);
  assert.equal(fetchCalls.length, 1);
  assert.equal(fetchCalls[0].url, `${backendBaseUrl}/v1/dsm/prepare-upload`);
}

async function testNewDsmRunsPreparePutAndFinalize() {
  fetchCalls.length = 0;
  const file = makeFile([10, 20, 30, 40], 'new-upload.tiff', 'image/tiff');
  const dataset = await service.uploadDsmToTerrainBackend(file);

  assert.equal(dataset.datasetId, finalizedDataset.datasetId);
  assert.equal(fetchCalls.length, 3);
  assert.deepEqual(
    fetchCalls.map((call) => [call.method, call.url]),
    [
      ['POST', `${backendBaseUrl}/v1/dsm/prepare-upload`],
      ['PUT', 'https://upload-target.test/staged'],
      ['POST', `${backendBaseUrl}/v1/dsm/finalize-upload`],
    ],
  );
}

async function testUploadErrorsSurfaceToCaller() {
  fetchCalls.length = 0;
  const file = makeFile([9, 8, 7, 6], 'upload-fail.tiff', 'image/tiff');
  await assert.rejects(
    service.uploadDsmToTerrainBackend(file),
    /presigned upload failed/,
  );
  assert.equal(fetchCalls.length, 2);
}

await testExistingDsmSkipsUploadAndFinalize();
await testNewDsmRunsPreparePutAndFinalize();
await testUploadErrorsSurfaceToCaller();

console.log('dsm_terrain_backend.test.ts passed');
