#!/bin/zsh
SCRIPT_PATH="${(%):-%x}"
MODEL_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
if [[ ! -f "$MODEL_DIR/model.py" ]]; then
  MODEL_FILE="$(find "$HOME/Desktop" "$HOME/Downloads" "$HOME/Documents" -type f -path '*/valorant_role_model_v*/model.py' -print -quit 2>/dev/null)"
  if [[ -n "$MODEL_FILE" ]]; then
    MODEL_DIR="$(dirname -- "$MODEL_FILE")"
  fi
fi
cd "$MODEL_DIR" || exit 1
PYTHON_BIN="python3"
if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
fi
"$PYTHON_BIN" model.py backtest --predictions ./model_output/walk_forward_predictions.csv --lines ./historical_lines.csv --output ./historical_line_results.csv --summary ./historical_line_summary.json || exit 1
open -R ./historical_line_summary.json
