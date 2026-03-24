/***********************************************************************
 * utils/terrain.ts
 *
 * Functions for fetching terrain data.
 *
 * © 2025 <your-name>. MIT License.
 ***********************************************************************/

import { Polygon, TerrainTile } from '@/utils/terrainAspectHybrid';
import { getPolygonBounds } from './geometry';
import { applyActiveDsmToTerrainRgbTile } from '@/terrain/dsmSource';
import { getTerrainTileUrlForCurrentSource } from '@/terrain/terrainSource';

async function getTileData(
  z: number,
  x: number,
  y: number,
  token: string,
  signal: AbortSignal
): Promise<TerrainTile | null> {
  try {
    const terrainUrl = getTerrainTileUrlForCurrentSource(z, x, y);
    const response = terrainUrl
      ? await fetch(terrainUrl, { signal })
      : await fetch(`https://api.mapbox.com/v4/mapbox.terrain-rgb/${z}/${x}/${y}.pngraw?access_token=${token}`, { signal });
    if (!response.ok) {
      throw new Error(`Failed to fetch tile: ${response.statusText}`);
    }
    const blob = await response.blob();
    const image = await createImageBitmap(blob);
    const canvas = new OffscreenCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context');
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.width, image.height);
    const data = new Uint8ClampedArray(imageData.data);
    if (!terrainUrl) {
      await applyActiveDsmToTerrainRgbTile(z, x, y, image.width, data);
    }

    return {
      x,
      y,
      z,
      width: image.width,
      height: image.height,
      data,
      format: 'terrain-rgb' as const,
    } as TerrainTile;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      console.log('Tile fetch aborted');
    } else {
      console.error('Error fetching tile data:', error);
    }
    return null;
  }
}

function long2tile(lon: number, zoom: number) {
  return Math.floor(((lon + 180) / 360) * Math.pow(2, zoom));
}

function lat2tile(lat: number, zoom: number) {
  return Math.floor(
    ((1 - Math.log(Math.tan((lat * Math.PI) / 180) + 1 / Math.cos((lat * Math.PI) / 180)) / Math.PI) / 2) *
      Math.pow(2, zoom)
  );
}

export async function fetchTilesForPolygon(
  polygon: Polygon,
  zoom: number,
  token: string,
  signal: AbortSignal
): Promise<TerrainTile[]> {
  const bounds = getPolygonBounds(polygon.coordinates);
  const minTileX = long2tile(bounds.minLng, zoom);
  const maxTileX = long2tile(bounds.maxLng, zoom);
  const minTileY = lat2tile(bounds.maxLat, zoom);
  const maxTileY = lat2tile(bounds.minLat, zoom);

  const promises: Promise<TerrainTile | null>[] = [];
  for (let x = minTileX; x <= maxTileX; x++) {
    for (let y = minTileY; y <= maxTileY; y++) {
      promises.push(getTileData(zoom, x, y, token, signal));
    }
  }

  const results = await Promise.all(promises);
  return results.filter((t): t is TerrainTile => t !== null);
}
