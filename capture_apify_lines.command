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

if [[ ! -s ./.apify_token ]]; then
  echo "Paste your Apify API token. It will be saved only in this folder and ignored by Git."
  read -s "APIFY_TOKEN_VALUE?Apify token: "
  echo ""
  if [[ -z "$APIFY_TOKEN_VALUE" ]]; then
    echo "No token entered."
    exit 1
  fi
  print -rn -- "$APIFY_TOKEN_VALUE" > ./.apify_token
  chmod 600 ./.apify_token
  unset APIFY_TOKEN_VALUE
fi

"$PYTHON_BIN" apify_lines.py pull \
  --actor-id 4AmgQeem8dEgMEiRF \
  --output ./apify_future_lines.csv \
  --history-dir ./line_history || exit 1

if [[ ! -f ./apify_future_lines.csv || $(wc -l < ./apify_future_lines.csv) -le 1 ]]; then
  echo "No eligible VCT kill lines were found. Review line_history/rejected_lines.csv."
  read -k 1 "?Press any key to close..."
  exit 1
fi

"$PYTHON_BIN" model.py predict \
  --model ./model_output/production_model.json \
  --input ./apify_future_lines.csv \
  --output ./apify_predictions_latest.csv || exit 1

"$PYTHON_BIN" rank_lines.py \
  --input ./apify_predictions_latest.csv \
  --output ./apify_ranked_predictions.csv \
  --minimum-probability 0.50 || exit 1

"$PYTHON_BIN" apify_lines.py lock \
  --predictions ./apify_predictions_latest.csv \
  --model ./model_output/production_model.json \
  --ledger ./line_history/prediction_ledger.csv || exit 1

open -R ./apify_ranked_predictions.csv
echo "Lines captured, predictions generated, and the snapshot locked."
read -k 1 "?Press any key to close..."
