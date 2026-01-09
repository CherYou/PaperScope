#!/bin/bash

# Load environment variables from .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please copy .env.example to .env and configure your settings:"
    echo "  cp .env.example .env"
    exit 1
fi

echo "Loading environment variables from .env file..."
set -a  # automatically export all variables
source "$ENV_FILE"
set +a  # stop automatically exporting

# Validate critical variables
if [ "$MODEL_Name" = "/your/model/path" ] || [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME not configured in .env file"
    exit 1
fi

echo "==== Starting inference using external API... ===="
echo "Model Name: $MODEL_NAME"

cd "$( dirname -- "${BASH_SOURCE[0]}" )"
python -u run_multi_react_api.py \
    --dataset "$DATASET" \
    --output "$OUTPUT_PATH" \
    --max_workers "$MAX_WORKERS" \
    --model "$MODEL_NAME" \
    --temperature "$TEMPERATURE" \
    --presence_penalty "$PRESENCE_PENALTY" \
    --total_splits "${WORLD_SIZE:-1}" \
    --worker_split "$((${RANK:-0} + 1))" \
    --roll_out_count "$ROLLOUT_COUNT"


echo "Inference finished."