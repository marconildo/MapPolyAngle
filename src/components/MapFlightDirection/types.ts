/***********************************************************************
 * types.ts
 *
 * Type definitions for the MapFlightDirection component.
 *
 * © 2025 <your-name>. MIT License.
 ***********************************************************************/

import { Polygon as AspectPolygon, AspectResult } from '../../utils/terrainAspectHybrid';
import type { FlightParams } from '@/domain/types';

/** Enhanced result interface for multiple polygons */
export interface PolygonAnalysisResult {
  polygonId: string;
  result: AspectResult;
  polygon: AspectPolygon;
  terrainZoom: number; // Track which zoom level was used
}

/** Per‑polygon flight planning parameters set by the user. */
export type PolygonParams = FlightParams;
