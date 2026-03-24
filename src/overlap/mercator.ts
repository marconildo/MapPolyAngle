const R = 6378137;
const WORLD = Math.PI * R * 2;

export function lngLatToMeters(lng: number, lat: number): [number, number] {
  const λ = (lng * Math.PI) / 180;
  const clamped = Math.max(-85.05112878, Math.min(85.05112878, lat)) * Math.PI/180;
  return [R * λ, R * Math.log(Math.tan(Math.PI / 4 + clamped / 2))];
}

export function tileMetersBounds(z: number, x: number, y: number) {
  const tiles = 1 << z;
  const tileMeters = WORLD / tiles;
  const minX = -WORLD / 2 + x * tileMeters;
  const maxY =  WORLD / 2 - y * tileMeters;
  const maxX = minX + tileMeters;
  const minY = maxY - tileMeters;
  return { minX, minY, maxX, maxY, pixelSize: tileMeters / 512 }; // default 512; override if needed
}

export function pixelToWorld(tx: {minX:number;maxY:number;pixelSize:number}, col: number, row: number) {
  const x = tx.minX + (col + 0.5) * tx.pixelSize;
  const y = tx.maxY - (row + 0.5) * tx.pixelSize;
  return [x, y] as const;
}

export function worldToPixel(tx: {minX:number;maxY:number;pixelSize:number}, x: number, y: number) {
  const col = (x - tx.minX) / tx.pixelSize - 0.5;
  const row = (tx.maxY - y) / tx.pixelSize - 0.5;
  return [col, row] as const;
}

export function tileCornersLngLat(z: number, x: number, y: number): [number,number][] {
  const bounds = tileMetersBounds(z, x, y);
  const m2ll = (mx: number, my: number) => {
    const lng = (mx / R) * 180/Math.PI;
    const lat = (Math.atan(Math.sinh(my / R)))*180/Math.PI;
    return [lng, lat] as [number,number];
  };
  // Mapbox expects TL, TR, BR, BL for image source coordinates
  return [
    m2ll(bounds.minX, bounds.maxY),
    m2ll(bounds.maxX, bounds.maxY),
    m2ll(bounds.maxX, bounds.minY),
    m2ll(bounds.minX, bounds.minY),
  ];
}

export function lngLatToTile(lng: number, lat: number, z: number) {
  const n = 2 ** z;
  const x = Math.floor(((lng + 180) / 360) * n);
  const y = Math.floor( (1 - Math.log(Math.tan((lat*Math.PI)/180) + 1/Math.cos((lat*Math.PI)/180))/Math.PI) / 2 * n );
  return {x, y};
}
