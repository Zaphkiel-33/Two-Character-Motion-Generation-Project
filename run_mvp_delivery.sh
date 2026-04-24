#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${TWOCHAR_ENV_NAME:-twochar_rtmpose}"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda was not found. Install Conda first, then create the environment from environment.macos-arm64.yml."
    exit 1
fi

VIDEO_PATH="${ROOT_DIR}/video2.mp4"
TWO_D_OUTPUT="${ROOT_DIR}/outputs/video_rtmpose"
IMPROVED_3D_OUTPUT="${ROOT_DIR}/outputs_improved/video_3d"
DELIVERY_OUTPUT="${ROOT_DIR}/mvp_delivery/video2"
SOURCE_LABEL="video2_mvp"
FORCE_2D=0
FORCE_3D=0
FORCE_MVP=0
EXPORT_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video-path)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --two-d-output)
            TWO_D_OUTPUT="$2"
            shift 2
            ;;
        --improved-3d-output)
            IMPROVED_3D_OUTPUT="$2"
            shift 2
            ;;
        --delivery-output)
            DELIVERY_OUTPUT="$2"
            shift 2
            ;;
        --source-label)
            SOURCE_LABEL="$2"
            shift 2
            ;;
        --force-2d)
            FORCE_2D=1
            shift
            ;;
        --force-3d)
            FORCE_3D=1
            shift
            ;;
        --force-mvp)
            FORCE_MVP=1
            shift
            ;;
        *)
            EXPORT_ARGS+=("$1")
            shift
            ;;
    esac
done

SOURCE_SEQUENCE="${TWO_D_OUTPUT}/pose_sequences.json"
SOURCE_METADATA="${TWO_D_OUTPUT}/metadata.json"
SOURCE_MOTION="${IMPROVED_3D_OUTPUT}/motion3d_sequences.json"

if [[ ${FORCE_2D} -eq 1 ]]; then
    rm -rf "${TWO_D_OUTPUT}"
fi

if [[ ! -f "${SOURCE_SEQUENCE}" || ! -f "${SOURCE_METADATA}" ]]; then
    conda run -n "${ENV_NAME}" python "${ROOT_DIR}/two_character_rtmpose_pipeline.py" \
        --video-path "${VIDEO_PATH}" \
        --output-root "${TWO_D_OUTPUT}" \
        --save-video
fi

if [[ ${FORCE_3D} -eq 1 ]]; then
    rm -rf "${IMPROVED_3D_OUTPUT}"
fi

if [[ ! -f "${SOURCE_MOTION}" ]]; then
    conda run -n "${ENV_NAME}" python "${ROOT_DIR}/two_character_3d_reconstruction.py" \
        --sequence-json "${SOURCE_SEQUENCE}" \
        --metadata-json "${SOURCE_METADATA}" \
        --output-root "${IMPROVED_3D_OUTPUT}" \
        --use-smoothed
fi

if [[ ${FORCE_MVP} -eq 1 ]]; then
    rm -rf "${DELIVERY_OUTPUT}"
fi

CMD=(
    conda run -n "${ENV_NAME}" python "${ROOT_DIR}/export_dual_interaction_mvp.py"
    --motion-json "${SOURCE_MOTION}"
    --output-root "${DELIVERY_OUTPUT}"
    --source-label "${SOURCE_LABEL}"
)

if [[ ${#EXPORT_ARGS[@]} -gt 0 ]]; then
    CMD+=("${EXPORT_ARGS[@]}")
fi

exec "${CMD[@]}"
