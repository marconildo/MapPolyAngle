import type { PolygonLngLat } from "./types";
import { lngLatToTile } from "./mercator";

export function tilesCoveringPolygon(polygon: PolygonLngLat, z: number, pad: number = 0) {
  const lons = polygon.ring.map((point) => point[0]);
  const lats = polygon.ring.map((point) => point[1]);
  const min = { lon: Math.min(...lons), lat: Math.min(...lats) };
  const max = { lon: Math.max(...lons), lat: Math.max(...lats) };
  const tMin = lngLatToTile(min.lon, max.lat, z);
  const tMax = lngLatToTile(max.lon, min.lat, z);
  const tiles: { x: number; y: number }[] = [];
  for (let x = tMin.x - pad; x <= tMax.x + pad; x++) {
    for (let y = tMin.y - pad; y <= tMax.y + pad; y++) {
      tiles.push({ x, y });
    }
  }
  return tiles;
}

export function tilesCoveringRing(ring: [number, number][], z: number, pad: number = 0) {
  return tilesCoveringPolygon({ ring }, z, pad);
}
