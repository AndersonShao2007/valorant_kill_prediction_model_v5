#!/bin/zsh
MODEL_DIR="/Users/andersonshao/Documents/Codex/2026-07-31/i/outputs/valorant_role_model_v2"
cd "$MODEL_DIR" || exit 1
open -R ./future_lines.csv
echo "The exact input file is selected in Finder. Open and edit this file, then save it before running predictions."
