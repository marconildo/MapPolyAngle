/**
 * Projection utilities for converting between geographic and metric coordinates.
 * Provides tested, consistent implementations used throughout the application.
 */

/**
 * Convert longitude/latitude to Web Mercator meters (EPSG:3857).
 */
export function lngLatToMeters(lng: number, lat: number): [number, number] {
  const R = 6378137; // WGS84 equatorial radius in meters
  const x = lng * Math.PI / 180 * R;
  const y = Math.log(Math.tan((90 + lat) * Math.PI / 360)) * R;
  return [x, y];
}

/**
 * Convert Web Mercator meters (EPSG:3857) back to longitude/latitude.
 */
export function metersToLngLat(x: number, y: number): [number, number] {
  const R = 6378137; // WGS84 equatorial radius in meters
  const lng = (x / R) * 180 / Math.PI;
  const lat = (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) * 180 / Math.PI;
  return [lng, lat];
}

/**
 * Calculate the haversine distance between two points in meters.
 */
export function haversineMeters(a: [number, number], b: [number, number]): number {
  const R = 6371000; // Earth's radius in meters
  const [lng1, lat1] = a;
  const [lng2, lat2] = b;

  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;

  const x = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLng / 2) * Math.sin(dLng / 2);

  const c = 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));

  return R * c;
}
