# Terrain Splitter Backend

Local FastAPI service for terrain-aware polygon partitioning.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e "backend/terrain_splitter[test]"
export MAPBOX_TOKEN=...
npm run backend:terrain-splitter
```

The frontend will call this service when:

```bash
export VITE_TERRAIN_PARTITION_BACKEND_URL=http://127.0.0.1:8090
```

## Debug artifacts

When `debug=true` in the solve request, artifacts are written under:

- `backend/terrain_splitter/.debug/`

Fetched terrain tiles are cached under:

- `backend/terrain_splitter/.cache/`

## AWS SAM deployment

This backend can also be deployed as an AWS Lambda Function URL in a similar style to
[DetectAnythinBackend-main](/Users/bharat/Documents/src/MapPolyAngle/DetectAnythinBackend-main).

Files:

- [template.yaml](/Users/bharat/Documents/src/MapPolyAngle/backend/terrain_splitter/template.yaml)
- [samconfig.toml](/Users/bharat/Documents/src/MapPolyAngle/backend/terrain_splitter/samconfig.toml)
- [main.py](/Users/bharat/Documents/src/MapPolyAngle/backend/terrain_splitter/main.py)
- [requirements.txt](/Users/bharat/Documents/src/MapPolyAngle/backend/terrain_splitter/requirements.txt)

The SAM deployment uses:

- Lambda Function URL
- `main.lambda_handler` via `Mangum`
- `/tmp/terrain-splitter/cache` for terrain tile cache
- `/tmp/terrain-splitter/debug` for optional debug artifacts

### Build and deploy

From [backend/terrain_splitter](/Users/bharat/Documents/src/MapPolyAngle/backend/terrain_splitter):

```bash
sam build --use-container
sam deploy --guided
```

Or with explicit parameter overrides:

```bash
sam build --use-container
sam deploy \
  --guided \
  --parameter-overrides \
    MapboxToken=YOUR_MAPBOX_TOKEN \
    AllowedOrigin=https://your-frontend.example.com
```

Or use the helper script:

```bash
export MAPBOX_TOKEN=YOUR_MAPBOX_TOKEN
./deploy_aws.sh
```

To deploy a separate staging stack and Function URL without touching the current production stack:

```bash
export MAPBOX_TOKEN=YOUR_MAPBOX_TOKEN
./deploy_aws_staging.sh
```

The staging deploy uses the `staging` SAM config environment from
[samconfig.toml](/Users/bharat/Documents/src/MapPolyAngle-main-backend-sync-20260320/backend/terrain_splitter/samconfig.toml),
which points at a separate stack name (`terrain-splitter-staging`) and S3 prefix
(`terrain-splitter-staging`).

After deploy, set the frontend to the Function URL:

```bash
export VITE_TERRAIN_PARTITION_BACKEND_URL=https://<function-id>.lambda-url.<region>.on.aws
```

Notes:

- The Lambda package is stateless. Cache/debug data is ephemeral in `/tmp`.
- On macOS, `sam build` without `--use-container` can fail for `numpy`/`pillow` because Lambda needs Linux wheels.
- Memory is set to the Lambda maximum (`10240 MB`) because the solver is CPU-heavy and benefits from extra CPU allocation.
- AWS deploys now enable exact root-branch fan-out via Lambda self-invocation:
  - `TERRAIN_SPLITTER_ROOT_PARALLEL_MODE=lambda`
  - `TERRAIN_SPLITTER_ROOT_PARALLEL_WORKERS`
  - `TERRAIN_SPLITTER_ROOT_PARALLEL_GRANULARITY=branch|subtree`
  - `TERRAIN_SPLITTER_ROOT_PARALLEL_MAX_INFLIGHT`
  - `TERRAIN_SPLITTER_LAMBDA_INVOKE_READ_TIMEOUT_SEC`
  - `TERRAIN_SPLITTER_NESTED_LAMBDA_MIN_CELLS`
  - `TERRAIN_SPLITTER_NESTED_LAMBDA_MAX_INFLIGHT`
- Subtree workers can now re-fan out one level deeper when a child solve is still large enough, using a smaller nested inflight cap to keep the invocation tree bounded.
- Local runs still default to exact process-based root fan-out when `TERRAIN_SPLITTER_ROOT_PARALLEL_WORKERS > 1`.
- Timeout is `900s`, matching Lambda's maximum.
