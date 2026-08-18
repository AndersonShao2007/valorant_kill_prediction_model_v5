#!/bin/zsh
SCRIPT_PATH="${(%):-%x}"
MODEL_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
if [[ ! -f "$MODEL_DIR/model.py" || ! -f "$MODEL_DIR/future_lines.csv" ]]; then
  MODEL_FILE="$(find "$HOME/Desktop" "$HOME/Downloads" "$HOME/Documents" -type f -path '*/valorant_role_model_v*/model.py' -print -quit 2>/dev/null)"
  if [[ -n "$MODEL_FILE" ]]; then
    MODEL_DIR="$(dirname -- "$MODEL_FILE")"
  fi
fi
cd "$MODEL_DIR" || {
  echo "Model folder not found: $MODEL_DIR"
  read -k 1 "?Press any key to close..."
  exit 1
}
if [[ ! -f ./model.py || ! -f ./future_lines.csv ]]; then
  echo "The V5 model folder could not be found."
  echo "Expected model.py and future_lines.csv in: $MODEL_DIR"
  read -k 1 "?Press any key to close..."
  exit 1
fi
echo "Reading these saved lines from future_lines.csv:"
sed -n '1,20p' ./future_lines.csv
echo ""
PYTHON_BIN="python3"
if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
fi
"$PYTHON_BIN" model.py predict --model ./model_output/production_model.json --input ./future_lines.csv --output ./future_predictions.csv || {
  echo "Prediction stopped. Fix future_lines.csv, save it, and run again."
  read -k 1 "?Press any key to close..."
  exit 1
}
"$PYTHON_BIN" rank_lines.py --input ./future_predictions.csv --output ./ranked_predictions.csv --minimum-probability 0.50 || exit 1
if command -v code >/dev/null 2>&1; then
  code ./ranked_predictions.csv
else
  open -R ./ranked_predictions.csv
  echo "Prediction finished. ranked_predictions.csv is selected in Finder."
fi
