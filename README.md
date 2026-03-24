# MapPolyAngle

Terrain-aware flight planning for camera and lidar mapping missions.

The app lets you draw or import areas, analyze terrain, generate flight lines, inspect overlap or lidar-density rasters, and split large areas into terrain-aligned sub-areas with a backend solver.

- Live app: [map-poly-angle.vercel.app](https://map-poly-angle.vercel.app)
- Frontend: React + Vite + Mapbox GL + Deck.gl
- Backend: FastAPI terrain-splitting service, runnable locally or on AWS Lambda

## What it does

- Import and export Wingtra `.flightplan` files
- Import KML polygons and DJI/Wingtra pose JSON
- Optimize flight direction from terrain analysis
- Visualize camera overlap / GSD and lidar point-density rasters
- Auto-split polygons into terrain-aligned faces using the backend solver
- Support per-polygon camera or lidar settings

## Frontend quick start

Requirements:

- Node.js 18+
- A Mapbox token

Setup:

```bash
git clone git@github.com:wingtra/MapPolyAngle.git
cd MapPolyAngle
npm install
echo "VITE_MAPBOX_TOKEN=your_mapbox_token_here" > .env
npm run dev
```

Open:

```text
http://localhost:5173
```

## Backend quick start

The terrain-splitting backend is optional but recommended for `Auto split`.

Local backend:

```bash
python3 -m venv backend/terrain_splitter/.venv
source backend/terrain_splitter/.venv/bin/activate
pip install -e "backend/terrain_splitter[test]"
export MAPBOX_TOKEN=your_mapbox_token_here
npm run backend:terrain-splitter
```

Then point the frontend at it:

```bash
export VITE_TERRAIN_PARTITION_BACKEND_URL=http://127.0.0.1:8090
npm run dev
```

Backend deployment details live in [backend/terrain_splitter/README.md](/Users/bharat/Documents/src/MapPolyAngle/backend/terrain_splitter/README.md).

## Useful commands

```bash
npm run dev
npm run build
npm run lint
npm run typecheck
npm run test:terrain-split
npm run test:terrain-objective
npm run test:terrain-graph
```

Backend:

```bash
backend/terrain_splitter/.venv/bin/python -m pytest -q backend/terrain_splitter/tests
sam validate --template-file backend/terrain_splitter/template.yaml
```

Optional pre-commit setup:

```bash
python3 -m pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

## Project layout

```text
src/
  components/
    MapFlightDirection/
    OverlapGSDPanel.tsx
  domain/
  interop/
  overlap/
  services/
  tests/
  utils/

backend/terrain_splitter/
  terrain_splitter/
  tests/
  template.yaml
```

## Notes

- The frontend can run without the backend, but `Auto split` will fall back to the older local implementation.
- The AWS deployment uses a Lambda Function URL and exact fan-out across split branches for better solve latency.
