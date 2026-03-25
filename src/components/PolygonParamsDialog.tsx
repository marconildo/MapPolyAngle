import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectTrigger, SelectContent, SelectItem, SelectValue } from "@/components/ui/select";
import { SONY_RX1R2, SONY_RX1R3, DJI_ZENMUSE_P1_24MM, ILX_LR1_INSPECT_85MM, MAP61_17MM, RGB61_24MM } from "@/domain/camera";
import { DEFAULT_LIDAR_MAX_RANGE_M, WINGTRA_LIDAR_XT32M2X } from "@/domain/lidar";
import type { PolygonParams } from "@/components/MapFlightDirection/types";

type Props = {
  open: boolean;
  polygonId: string | null;
  onClose: () => void;
  onSubmit: (params: PolygonParams) => void;
  onSubmitAll?: (params: PolygonParams) => void; // bulk apply
  defaults?: PolygonParams;
};

export default function PolygonParamsDialog({
  open,
  polygonId,
  onClose,
  onSubmit,
  onSubmitAll,
  defaults
}: Props) {
  const clampNumber = React.useCallback((value: number, min: number, max: number, fallback: number) => {
    if (!Number.isFinite(value)) return fallback;
    return Math.min(max, Math.max(min, value));
  }, []);

  const [payloadKind, setPayloadKind] = React.useState<"camera" | "lidar">(defaults?.payloadKind ?? "camera");
  const [altitudeAGL, setAltitudeAGL] = React.useState<number>(defaults?.altitudeAGL ?? 100);
  const [frontOverlap, setFrontOverlap] = React.useState<number>(defaults?.frontOverlap ?? 70);
  const [sideOverlap, setSideOverlap] = React.useState<number>(defaults?.sideOverlap ?? 70);
  const [cameraKey, setCameraKey] = React.useState<string>(defaults?.cameraKey ?? "MAP61_17MM");
  const [lidarKey, setLidarKey] = React.useState<string>(defaults?.lidarKey ?? WINGTRA_LIDAR_XT32M2X.key);
  const [speedMps, setSpeedMps] = React.useState<number>(defaults?.speedMps ?? WINGTRA_LIDAR_XT32M2X.defaultSpeedMps);
  const [lidarReturnMode, setLidarReturnMode] = React.useState<"single" | "dual" | "triple">(defaults?.lidarReturnMode ?? "single");
  const [mappingFovDeg, setMappingFovDeg] = React.useState<number>(defaults?.mappingFovDeg ?? WINGTRA_LIDAR_XT32M2X.effectiveHorizontalFovDeg);
  const [maxLidarRangeM, setMaxLidarRangeM] = React.useState<number>(defaults?.maxLidarRangeM ?? DEFAULT_LIDAR_MAX_RANGE_M);
  const [showAdvanced, setShowAdvanced] = React.useState<boolean>(false);
  const [useCustomBearing, setUseCustomBearing] = React.useState<boolean>(defaults?.useCustomBearing ?? false);
  const [customBearingDeg, setCustomBearingDeg] = React.useState<number>(defaults?.customBearingDeg ?? 0);
  const [rotateCamera90, setRotateCamera90] = React.useState<boolean>(
    Math.round((((defaults?.cameraYawOffsetDeg ?? 0) % 180) + 180) % 180) === 90
  );

  // map keys to models (could be lifted up later if needed)
  const cameraOptions: Array<{ key:string; model:any; label:string }> = [
    { key:'SONY_RX1R2', model: SONY_RX1R2, label: SONY_RX1R2.names?.[0] || 'RX1RII 35mm' },
    { key:'SONY_RX1R3', model: SONY_RX1R3, label: SONY_RX1R3.names?.[0] || 'SURVEY61' },
    { key:'DJI_ZENMUSE_P1_24MM', model: DJI_ZENMUSE_P1_24MM, label: DJI_ZENMUSE_P1_24MM.names?.[0] || 'DJI Zenmuse P1 24mm' },
    { key:'ILX_LR1_INSPECT_85MM', model: ILX_LR1_INSPECT_85MM, label: ILX_LR1_INSPECT_85MM.names?.[0] || 'INSPECT 85mm' },
    { key:'MAP61_17MM', model: MAP61_17MM, label: MAP61_17MM.names?.[0] || 'MAP61 17mm' },
    { key:'RGB61_24MM', model: RGB61_24MM, label: RGB61_24MM.names?.[0] || 'RGB61 24mm' },
  ];
  const lidarOptions = [
    { key: WINGTRA_LIDAR_XT32M2X.key, label: WINGTRA_LIDAR_XT32M2X.names?.[0] || 'Wingtra Lidar' },
  ];

  React.useEffect(() => {
    if (open) {
      setPayloadKind(defaults?.payloadKind ?? "camera");
      setAltitudeAGL(defaults?.altitudeAGL ?? 100);
      setFrontOverlap(defaults?.frontOverlap ?? ((defaults?.payloadKind ?? "camera") === "lidar" ? 0 : 70));
      setSideOverlap(defaults?.sideOverlap ?? 70);
      setCameraKey(defaults?.cameraKey ?? "MAP61_17MM");
      setLidarKey(defaults?.lidarKey ?? WINGTRA_LIDAR_XT32M2X.key);
      setSpeedMps(defaults?.speedMps ?? WINGTRA_LIDAR_XT32M2X.defaultSpeedMps);
      setLidarReturnMode(defaults?.lidarReturnMode ?? "single");
      setMappingFovDeg(defaults?.mappingFovDeg ?? WINGTRA_LIDAR_XT32M2X.effectiveHorizontalFovDeg);
      setMaxLidarRangeM(defaults?.maxLidarRangeM ?? DEFAULT_LIDAR_MAX_RANGE_M);
      setUseCustomBearing(defaults?.useCustomBearing ?? false);
      setCustomBearingDeg(defaults?.customBearingDeg ?? 0);
      const rotate = Math.round((((defaults?.cameraYawOffsetDeg ?? 0) % 180) + 180) % 180) === 90;
      setRotateCamera90(rotate);
      setShowAdvanced(!!(defaults?.useCustomBearing) || rotate);
    }
  }, [open, defaults?.payloadKind, defaults?.altitudeAGL, defaults?.frontOverlap, defaults?.sideOverlap, defaults?.cameraKey, defaults?.lidarKey, defaults?.speedMps, defaults?.lidarReturnMode, defaults?.mappingFovDeg, defaults?.maxLidarRangeM, defaults?.useCustomBearing, defaults?.customBearingDeg, defaults?.cameraYawOffsetDeg]);

  if (!open || !polygonId) return null;

  return (
    <div className="absolute top-2 left-2 z-50 w-80">
      <Card className="shadow-lg">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Flight setup for <span className="font-mono">#{polygonId.slice(0,8)}</span></CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <label className="text-xs text-gray-600 block">
            Payload
            <Select value={payloadKind} onValueChange={(value) => setPayloadKind(value as "camera" | "lidar")}>
              <SelectTrigger className="h-8 text-xs mt-1">
                <SelectValue placeholder="Select payload" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="camera" className="text-xs">Camera</SelectItem>
                <SelectItem value="lidar" className="text-xs">Lidar</SelectItem>
              </SelectContent>
            </Select>
          </label>
          {payloadKind === "camera" ? (
            <label className="text-xs text-gray-600 block">
              Camera
              <Select value={cameraKey} onValueChange={setCameraKey}>
                <SelectTrigger className="h-8 text-xs mt-1">
                  <SelectValue placeholder="Select camera" />
                </SelectTrigger>
                <SelectContent>
                  {cameraOptions.map(c => (
                    <SelectItem value={c.key} key={c.key} className="text-xs">
                      {c.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </label>
          ) : (
            <label className="text-xs text-gray-600 block">
              Lidar
              <Select value={lidarKey} onValueChange={setLidarKey}>
                <SelectTrigger className="h-8 text-xs mt-1">
                  <SelectValue placeholder="Select lidar" />
                </SelectTrigger>
                <SelectContent>
                  {lidarOptions.map(l => (
                    <SelectItem value={l.key} key={l.key} className="text-xs">
                      {l.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </label>
          )}
          <label className="text-xs text-gray-600 block">
            Altitude AGL (m)
            <input className="w-full border rounded px-2 py-1 text-xs" type="number"
                   value={altitudeAGL}
                   onChange={(e)=>setAltitudeAGL(Math.max(1, parseInt(e.target.value || "100")))} />
          </label>
          {payloadKind === "camera" && (
            <label className="text-xs text-gray-600 block">
              Front overlap (%)
              <input className="w-full border rounded px-2 py-1 text-xs" type="number" min={0} max={95}
                     value={frontOverlap}
                     onChange={(e)=>setFrontOverlap(clampNumber(parseInt(e.target.value || "70"), 0, 95, 70))} />
            </label>
          )}
          <label className="text-xs text-gray-600 block">
            Side overlap (%)
            <input className="w-full border rounded px-2 py-1 text-xs" type="number" min={0} max={95}
                   value={sideOverlap}
                   onChange={(e)=>setSideOverlap(clampNumber(parseInt(e.target.value || "70"), 0, 95, 70))} />
          </label>
          {payloadKind === "lidar" && (
            <>
              <label className="text-xs text-gray-600 block">
                Speed (m/s)
                <input className="w-full border rounded px-2 py-1 text-xs" type="number" min={1} step={0.1}
                       value={speedMps}
                       onChange={(e)=>setSpeedMps(Math.max(0.1, parseFloat(e.target.value || `${WINGTRA_LIDAR_XT32M2X.defaultSpeedMps}`)))} />
              </label>
              <label className="text-xs text-gray-600 block">
                Return mode
                <Select value={lidarReturnMode} onValueChange={(value) => setLidarReturnMode(value as "single" | "dual" | "triple")}>
                  <SelectTrigger className="h-8 text-xs mt-1">
                    <SelectValue placeholder="Select return mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="single" className="text-xs">Single return</SelectItem>
                    <SelectItem value="dual" className="text-xs">Dual return</SelectItem>
                    <SelectItem value="triple" className="text-xs">Triple return</SelectItem>
                  </SelectContent>
                </Select>
              </label>
              <label className="text-xs text-gray-600 block">
                Mapping FOV (deg)
                <input className="w-full border rounded px-2 py-1 text-xs" type="number" min={1} max={180}
                       value={mappingFovDeg}
                       onChange={(e)=>setMappingFovDeg(clampNumber(parseFloat(e.target.value || `${WINGTRA_LIDAR_XT32M2X.effectiveHorizontalFovDeg}`), 1, 180, WINGTRA_LIDAR_XT32M2X.effectiveHorizontalFovDeg))} />
              </label>
              <label className="text-xs text-gray-600 block">
                Max lidar range (m)
                <input className="w-full border rounded px-2 py-1 text-xs" type="number" min={1} step={1}
                       value={maxLidarRangeM}
                       onChange={(e)=>setMaxLidarRangeM(Math.max(1, parseFloat(e.target.value || `${DEFAULT_LIDAR_MAX_RANGE_M}`)))} />
              </label>
            </>
          )}

          <div className="pt-1">
            <button
              type="button"
              className="text-[11px] text-blue-600 hover:underline"
              onClick={() => setShowAdvanced((prev) => !prev)}
            >
              {showAdvanced ? 'Hide advanced options' : 'Show advanced options'}
            </button>
          </div>

          {showAdvanced && (
            <div className="border rounded-md p-2 space-y-2 bg-slate-50">
              <label className="flex items-center gap-2 text-xs text-gray-600">
                <input
                  type="checkbox"
                  checked={useCustomBearing}
                  onChange={(e) => {
                    setUseCustomBearing(e.target.checked);
                    if (e.target.checked) setShowAdvanced(true);
                  }}
                />
                Use custom flight direction
              </label>
              {payloadKind === "camera" && (
                <label className="flex items-center gap-2 text-xs text-gray-600">
                  <input
                    type="checkbox"
                    checked={rotateCamera90}
                    onChange={(e) => {
                      setRotateCamera90(e.target.checked);
                      if (e.target.checked) setShowAdvanced(true);
                    }}
                  />
                  Rotate camera 90° (swap width/height)
                </label>
              )}
              <label className="text-xs text-gray-600 block">
                Flight direction (° clockwise from North)
                <input
                  className="w-full border rounded px-2 py-1 text-xs mt-1"
                  type="number"
                  min={0}
                  max={359.9}
                  step={0.1}
                  value={customBearingDeg}
                  disabled={!useCustomBearing}
                  onChange={(e) => {
                    const raw = parseFloat(e.target.value || '0');
                    if (Number.isFinite(raw)) setCustomBearingDeg(raw);
                  }}
                />
              </label>
            </div>
          )}

          <div className="flex gap-2 pt-1">
            <Button
              size="sm"
              className="flex-1 min-w-0 h-8 px-2 text-xs"
              onClick={() => {
                const normalizedBearing = ((customBearingDeg % 360) + 360) % 360;
                const payload: PolygonParams = {
                  payloadKind,
                  altitudeAGL: Math.max(1, Number.isFinite(altitudeAGL) ? altitudeAGL : 100),
                  frontOverlap: payloadKind === "lidar" ? 0 : clampNumber(frontOverlap, 0, 95, 70),
                  sideOverlap: clampNumber(sideOverlap, 0, 95, 70),
                  cameraKey: payloadKind === "camera" ? cameraKey : undefined,
                  lidarKey: payloadKind === "lidar" ? lidarKey : undefined,
                  cameraYawOffsetDeg: payloadKind === "camera" && rotateCamera90 ? 90 : 0,
                  speedMps: payloadKind === "lidar" ? Math.max(0.1, Number.isFinite(speedMps) ? speedMps : WINGTRA_LIDAR_XT32M2X.defaultSpeedMps) : undefined,
                  lidarReturnMode: payloadKind === "lidar" ? lidarReturnMode : undefined,
                  mappingFovDeg: payloadKind === "lidar" ? clampNumber(mappingFovDeg, 1, 180, WINGTRA_LIDAR_XT32M2X.effectiveHorizontalFovDeg) : undefined,
                  maxLidarRangeM: payloadKind === "lidar" ? Math.max(1, Number.isFinite(maxLidarRangeM) ? maxLidarRangeM : DEFAULT_LIDAR_MAX_RANGE_M) : undefined,
                  useCustomBearing,
                  customBearingDeg: useCustomBearing ? normalizedBearing : undefined,
                };
                onSubmit(payload);
              }}>
              Apply
            </Button>
            {onSubmitAll && (
              <Button
                size="sm"
                variant="secondary"
                className="h-8 px-2 text-xs whitespace-nowrap"
                onClick={() => {
                  const normalizedBearing = ((customBearingDeg % 360) + 360) % 360;
                  const payload: PolygonParams = {
                    payloadKind,
                    altitudeAGL: Math.max(1, Number.isFinite(altitudeAGL) ? altitudeAGL : 100),
                    frontOverlap: payloadKind === "lidar" ? 0 : clampNumber(frontOverlap, 0, 95, 70),
                    sideOverlap: clampNumber(sideOverlap, 0, 95, 70),
                    cameraKey: payloadKind === "camera" ? cameraKey : undefined,
                    lidarKey: payloadKind === "lidar" ? lidarKey : undefined,
                    cameraYawOffsetDeg: payloadKind === "camera" && rotateCamera90 ? 90 : 0,
                    speedMps: payloadKind === "lidar" ? Math.max(0.1, Number.isFinite(speedMps) ? speedMps : WINGTRA_LIDAR_XT32M2X.defaultSpeedMps) : undefined,
                    lidarReturnMode: payloadKind === "lidar" ? lidarReturnMode : undefined,
                    mappingFovDeg: payloadKind === "lidar" ? clampNumber(mappingFovDeg, 1, 180, WINGTRA_LIDAR_XT32M2X.effectiveHorizontalFovDeg) : undefined,
                    maxLidarRangeM: payloadKind === "lidar" ? Math.max(1, Number.isFinite(maxLidarRangeM) ? maxLidarRangeM : DEFAULT_LIDAR_MAX_RANGE_M) : undefined,
                    useCustomBearing,
                    customBearingDeg: useCustomBearing ? normalizedBearing : undefined,
                  };
                  onSubmitAll(payload);
                }}
                title="Apply these parameters to all remaining polygons awaiting setup"
              >
                Apply All
              </Button>
            )}
            <Button
              size="sm"
              variant="outline"
              className="h-8 px-2 text-xs whitespace-nowrap"
              onClick={onClose}
            >
              Cancel
            </Button>
          </div>
          <p className="text-[11px] text-gray-500">
            {onSubmitAll ? 'Use Apply All to apply these parameters to every remaining polygon in this import batch.' : 'After applying, flight lines and GSD will run for this polygon only.'}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
