# Valorant Role Model V5 (same folder)

This folder now supports two separate PrizePicks markets:

- `map_1`: kills on Map 1 only.
- `maps_1_2`: combined kills from Maps 1 and 2.

The production model uses VCT 2021–2026, exponentially weighted player form, current-team form, map-specific KPR, kill share, role proportions, recency-weighted training, Elo, optional no-vig market odds, and team-specific map tendencies.

## Update from VLR automatically

This folder integrates the authorized [`vlrdevapi`](https://github.com/Vanshbordia/vlrdevapi) Python client. It imports completed matches only. Automatic updates use the `relevant` competition filter: VCT/Champions/Masters plus Challengers, Ascension, and China Evolution Series. Game Changers, collegiate, academy, Premier, and unrelated small events are excluded. Raw API responses are cached in `api_data/raw`, then validated and deduplicated into `api_data/matches.csv` and `api_data/player_maps.csv`. The original archive is never edited.

First, double-click `setup_mac.command` once. To fetch up to 25 recent completed series and retrain, double-click `update_and_retrain.command`. The script finds the original VCT archive in Downloads, Documents, or Desktop. If it cannot find it, create `archive_path.txt` beside the model and put the full archive folder path on its first line, for example:

```text
/Users/yourname/Downloads/archive (2)
```

For a specific VLR match, copy the numeric series ID from its URL and run:

```bash
./.venv/bin/python update_vlr.py fetch --series-id 123456 --output ./api_data
./.venv/bin/python model.py train --archive "/full/path/to/archive (2)" --incremental-data ./api_data --output ./model_output
```

Running the updater again is safe: imported series and player-map rows are keyed by VLR IDs and replaced rather than duplicated. A backup of the previous production model is saved as `model_output/production_model_before_api_update.json` before retraining.

Review `api_data/selection_log.csv` to see each completed match considered, its event and teams, and the reason it was included, excluded, skipped, or deferred. Use `--competition-scope tier1` for VCT-level events only or `--competition-scope all` to disable automatic competition filtering. A manually supplied `--series-id` always bypasses the filter.

## Capture Apify lines and measure accuracy

`capture_apify_lines.command` downloads the latest successful dataset from Apify actor `4AmgQeem8dEgMEiRF`, filters it, runs predictions, and permanently locks that snapshot. On the first run it asks for your Apify API token and saves it locally as `.apify_token`; that file is excluded by `.gitignore` and must never be committed or shared.

The automatic line filter accepts only Valorant kill markets where both teams match teams observed in recent `VCT ...` events in `api_data/matches.csv`. `vct_team_aliases.csv` maps abbreviations such as SEN, PRX, DRX, EDG, RRQ, and BLG to canonical VLR team names. Edit that CSV when a VCT team or sponsor name changes.

Each capture produces:

- `line_history/raw/`: untouched Apify dataset JSON.
- `line_history/line_snapshots.csv`: deduplicated timestamped lines.
- `line_history/rejected_lines.csv`: excluded records and exact reasons.
- `line_history/apify_schema_fields.txt`: field names observed in the latest actor dataset.
- `apify_future_lines.csv`: latest eligible line per player/match/market.
- `apify_predictions_latest.csv`: complete model output.
- `apify_ranked_predictions.csv`: confidence ranking.
- `line_history/prediction_ledger.csv`: immutable prediction snapshot plus model hash.

After the matches finish, run `update_and_retrain.command` to import their VLR results, then double-click `score_apify_accuracy.command`. It matches predictions to completed series by date and team pair and writes `line_history/scored_predictions.csv` and `line_history/accuracy_summary.json`. The report includes projection MAE, direction accuracy, recommendation accuracy, Brier score, and accuracy at multiple confidence thresholds. Unfinished or ambiguous matches remain unscored instead of being guessed.

If the actor output schema changes, export one run as JSON and test it without downloading:

```bash
./.venv/bin/python apify_lines.py import-file /path/to/export.json
```

Review `line_history/rejected_lines.csv` after the first real run. It provides the fields needed to adjust aliases without discarding the raw dataset.

V4 added leakage-safe opponent defense and teammate competition. Opponent features include recent deaths allowed per round, kills per round, first-death rate, total fight pace, and map-specific deaths allowed. Teammate features summarize the other expected players' average, maximum, and combined recent KPR plus the number of high-usage teammates. During historical training, all of these values are calculated before the match being predicted.

V5 adds role-specific opponent defense. It measures the KPR an opponent allows to duelists, smokes, initiators, sentinels, and flex players, then weights those rates by each player's historical role mix. It also includes map-specific role defense and a strength-adjusted residual that compares actual KPR allowed with the opposing player's pre-match expected KPR. Small samples shrink toward league role averages using a 150-round prior.

## Predict future lines

Edit `future_lines.csv`. The columns are:

```csv
match_date,team,opponent,player,line,team_odds,opponent_odds,market,map1,map2,role_override
```

Examples:

```csv
2026-08-03,FULL SENSE,Rex Regum Qeon,primmie,38.5,,,maps_1_2,,,
2026-08-03,MIBR,Evil Geniuses,zekken,17.5,4.44,1.18,map_1,,,
```

Maps and role may be blank. Unknown maps are weighted using the two teams' recent map histories plus the current global map pool.

Double-click `run_predictions.command`. It creates:

- `future_predictions.csv`: complete model output.
- `ranked_predictions.csv`: lines sorted by confidence and standardized edge.

The model always provides a directional `more` or `less` lean. `recommendation` becomes actionable only when the estimated probability is at least 57%; otherwise it is `pass`.

The prediction output includes opponent-defense, role-matchup, and teammate-competition diagnostics so you can verify what context the model used. `opponent_role_residual` above zero means the opponent has allowed that player's role more kills than expected; below zero means fewer. Rosters are inferred from each player's latest recorded team; retrain after adding new match data when rosters change.

## Probability and uncertainty

The point projection is converted into a More/Less probability with a negative-binomial count distribution. Its dispersion is estimated separately for Map 1 and Maps 1–2 using historical residuals. `uncertainty_sd` shows the estimated kill volatility.

## Walk-forward validation

`model_output/evaluation.json` contains future-like tests for 2023, 2024, 2025, and 2026. For every test year, only earlier years trained that fold. `model_output/walk_forward_predictions.csv` contains the out-of-fold predictions used for honest historical-line backtests.

Current V5 walk-forward MAE is 4.095 kills for Map 1 and 5.945 kills for Maps 1–2. This improves on V4's 4.108 and 5.971 respectively. The previous V4 production weights remain available as `model_output/production_model_v4_backup.json`.

## Capture and backtest lines

Append every pre-match line snapshot to `historical_lines.csv`. Never add a line after seeing the result. Required join fields are `market`, `match_id`, and the exact player handle. Preserve `captured_at`, the line, and available decimal odds.

After adding historical lines, double-click `run_backtest.command`. It creates:

- `historical_line_results.csv`
- `historical_line_summary.json`

The summary compares model MAE with the PrizePicks line's MAE and reports selection accuracy at probability thresholds from 50% to 70%. Use those results to choose the recommendation threshold instead of guessing.

## Important limits

- Exact historical timestamps are unavailable; year and official match order are used for chronology.
- The 75% market / 25% Elo blend cannot be optimized until timestamped historical odds are collected.
- A projection is not a guarantee. Do not increase risk merely because the model produces more selections.
