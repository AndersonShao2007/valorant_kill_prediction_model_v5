#!/bin/zsh
SCRIPT_PATH="${(%):-%x}"
MODEL_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
cd "$MODEL_DIR" || exit 1
PYTHON_BIN="./.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Run setup_mac.command first."
  read -k 1 "?Press any key to close..."
  exit 1
fi

"$PYTHON_BIN" apify_lines.py score \
  --ledger ./line_history/prediction_ledger.csv \
  --matches ./api_data/matches.csv \
  --player-maps ./api_data/player_maps.csv \
  --output ./line_history/scored_predictions.csv \
  --summary ./line_history/accuracy_summary.json || exit 1

open -R ./line_history/accuracy_summary.json
echo "Accuracy report finished. Only completed, matched VLR results are scored."
read -k 1 "?Press any key to close..."
