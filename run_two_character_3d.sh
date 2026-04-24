#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${TWOCHAR_ENV_NAME:-twochar_rtmpose}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found. Install Conda first, then create the environment from environment.macos-arm64.yml."
    exit 1
fi

exec conda run -n "${ENV_NAME}" python "${ROOT_DIR}/two_character_3d_reconstruction.py" "$@"
