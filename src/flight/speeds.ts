import type { FlightParams } from '@/domain/types';
import { getLidarModel } from '@/domain/lidar';

export const DEFAULT_CAMERA_SPEED_MPS = 12;

export function getCruiseSpeedMps(params: FlightParams): number {
  if ((params.payloadKind ?? 'camera') === 'lidar') {
    return params.speedMps ?? getLidarModel(params.lidarKey).defaultSpeedMps;
  }
  return params.speedMps ?? DEFAULT_CAMERA_SPEED_MPS;
}
