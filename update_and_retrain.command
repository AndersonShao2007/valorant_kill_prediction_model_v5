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

ARCHIVE_DIR=""
if [[ -f ./archive_path.txt ]]; then
  ARCHIVE_DIR="$(head -n 1 ./archive_path.txt)"
fi
if [[ -z "$ARCHIVE_DIR" || ! -d "$ARCHIVE_DIR/all_ids" ]]; then
  ARCHIVE_DIR="$(find "$HOME/Downloads" "$HOME/Documents" "$HOME/Desktop" -type d -name all_ids -print -quit 2>/dev/null)"
  ARCHIVE_DIR="${ARCHIVE_DIR%/all_ids}"
fi
if [[ -z "$ARCHIVE_DIR" || ! -d "$ARCHIVE_DIR/all_ids" ]]; then
  echo "Could not find the original VCT archive."
  echo "Create archive_path.txt in this folder and put the archive's full path on line 1."
  read -k 1 "?Press any key to close..."
  exit 1
fi

echo "Fetching recent completed VLR matches..."
"$PYTHON_BIN" update_vlr.py fetch --output ./api_data --pages 2 --max-new 25 --competition-scope relevant || exit 1

if [[ -f ./model_output/production_model.json ]]; then
  cp ./model_output/production_model.json ./model_output/production_model_before_api_update.json
fi
echo "Retraining with the original archive plus api_data..."
"$PYTHON_BIN" model.py train \
  --archive "$ARCHIVE_DIR" \
  --incremental-data ./api_data \
  --output ./model_output || exit 1

echo "Update and retraining finished."
open -R ./model_output/production_model.json
read -k 1 "?Press any key to close..."
