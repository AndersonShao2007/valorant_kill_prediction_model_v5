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

echo "Creating a clean Python environment for this Mac..."
python3 -m venv .venv || {
  echo "Python could not create the environment. Install Python 3, then try again."
  read -k 1 "?Press any key to close..."
  exit 1
}

./.venv/bin/python -m pip install --upgrade pip || exit 1
./.venv/bin/python -m pip install --upgrade -r requirements.txt || {
  echo "The Python packages could not be installed. Check your internet connection and try again."
  read -k 1 "?Press any key to close..."
  exit 1
}

echo "Setup finished. You can now double-click run_predictions.command."
read -k 1 "?Press any key to close..."
