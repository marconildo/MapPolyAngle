#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_ENV="${1:-${SAM_CONFIG_ENV:-default}}"

if [ -z "${MAPBOX_TOKEN:-}" ]; then
  echo "MAPBOX_TOKEN must be set before deploying."
  exit 1
fi

DOCKER_SOCKET="${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock}"
export DOCKER_HOST="$DOCKER_SOCKET"

CONFIG_PARAMETER_OVERRIDES="$(
python3 - "$CONFIG_ENV" <<'PY'
from __future__ import annotations

import pathlib
import sys
import tomllib

config_env = sys.argv[1]
config_path = pathlib.Path("samconfig.toml")
with config_path.open("rb") as fh:
    config = tomllib.load(fh)

section = config.get(config_env, {})
deploy = section.get("deploy", {})
parameters = deploy.get("parameters", {})
print(parameters.get("parameter_overrides", ""))
PY
)"

if [ -n "$CONFIG_PARAMETER_OVERRIDES" ]; then
  PARAMETER_OVERRIDES="$CONFIG_PARAMETER_OVERRIDES MapboxToken=$MAPBOX_TOKEN"
else
  PARAMETER_OVERRIDES="MapboxToken=$MAPBOX_TOKEN"
fi

echo "Building terrain splitter Lambda with SAM (config env: $CONFIG_ENV)..."
# Force a fresh containerized build for the Python Lambda so SAM never reuses
# stale host-built deps (which can omit or mismatch compiled Linux wheels).
sam build --config-env "$CONFIG_ENV" --use-container --no-cached

echo "Deploying terrain splitter Lambda (config env: $CONFIG_ENV)..."
sam deploy --config-env "$CONFIG_ENV" --parameter-overrides "$PARAMETER_OVERRIDES"
