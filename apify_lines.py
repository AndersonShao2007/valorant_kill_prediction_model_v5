#!/usr/bin/env python3
"""Import authorized Apify line datasets, lock predictions, and score results.

The importer is intentionally tolerant of common actor output schemas. It keeps
the untouched JSON and writes rejected rows with reasons, so a new actor schema
can be mapped without losing data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

FUTURE_FIELDS = [
    "match_date", "team", "opponent", "player", "line", "team_odds",
    "opponent_odds", "market", "map1", "map2", "role_override",
    "snapshot_id", "captured_at", "source", "source_record_id", "league",
    "event", "start_time", "projection_type",
]
SNAPSHOT_FIELDS = FUTURE_FIELDS + ["raw_file"]
REJECT_FIELDS = [
    "captured_at", "source_record_id", "player", "team", "opponent", "line",
    "league", "event", "stat_type", "reason", "raw_file",
]
LEDGER_EXTRA_FIELDS = [
    "model_sha256", "locked_at", "matched_player", "predicted_kills", "edge",
    "probability_more", "probability_less", "direction", "recommendation",
    "uncertainty_sd", "warning",
]

STATIC_TEAM_ALIASES = {
    "100t": "100 Thieves", "blg": "Guangzhou Huadu Bilibili Gaming(Bilibili Gaming)",
    "bilibili gaming": "Guangzhou Huadu Bilibili Gaming(Bilibili Gaming)",
    "c9": "Cloud9", "dfm": "DetonatioN FocusMe", "drg": "Dragon Ranger Gaming",
    "drx": "KIWOOM DRX", "edg": "EDward Gaming", "eg": "Evil Geniuses",
    "fnc": "FNATIC", "fs": "FULL SENSE", "fur": "FURIA", "fut": "FUT Esports",
    "g2": "G2 Esports", "gx": "GIANTX", "ge": "Global Esports",
    "jdg": "JD Gaming", "kru": "KRÜ Esports", "krü": "KRÜ Esports",
    "lev": "Leviatán", "navi": "Natus Vincere", "ns": "Nongshim RedForce",
    "nrg": "NRG", "prx": "Paper Rex", "rrq": "Rex Regum Qeon",
    "sen": "Sentinels", "th": "Team Heretics", "ts": "Team Secret",
    "te": "Trace Esports", "tec": "Wuxi Titan Esports Club(Titan Esports Club)",
    "titan esports club": "Wuxi Titan Esports Club(Titan Esports Club)",
    "varrel": "VARREL", "wolv": "Wolves Esports", "wolves": "Wolves Esports",
    "xlg": "Xi Lai Gaming", "zeta": "ZETA DIVISION",
}


def clean(value: Any) -> str:
    return str(value or "").strip()


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean(value).casefold())


def number(value: Any) -> float | None:
    try:
        return float(clean(value))
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def append_unique(path: Path, rows: Iterable[dict[str, Any]], fields: list[str], keys: tuple[str, ...]) -> int:
    existing = read_csv(path)
    merged = {tuple(clean(row.get(k)) for k in keys): row for row in existing}
    before = len(merged)
    for row in rows:
        normalized = {field: row.get(field, "") for field in fields}
        merged[tuple(clean(normalized.get(k)) for k in keys)] = normalized
    write_csv(path, merged.values(), fields)
    return len(merged) - before


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict):
                result.update(flatten(child, path))
            elif not isinstance(child, list):
                result[path.casefold()] = child
    return result


def first(flat: dict[str, Any], paths: tuple[str, ...]) -> Any:
    for path in paths:
        if path.casefold() in flat and clean(flat[path.casefold()]):
            return flat[path.casefold()]
    for path in paths:
        suffix = "." + path.casefold()
        matches = [v for k, v in flat.items() if k.endswith(suffix) and clean(v)]
        if len(matches) == 1:
            return matches[0]
    return ""


def parse_matchup(value: str) -> tuple[str, str]:
    text = clean(value)
    for token in (" vs. ", " vs ", " v ", " @ "):
        if token in text.casefold():
            index = text.casefold().index(token)
            return text[:index].strip(), text[index + len(token):].strip()
    return "", ""


def parse_match_date(start_time: str, fallback: str = "") -> str:
    text = clean(start_time)
    if text:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
        except ValueError:
            pass
        match = re.search(r"20\d{2}-\d{2}-\d{2}", text)
        if match:
            return match.group(0)
    return clean(fallback)


def load_team_lookup(matches_path: Path, aliases_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    # Only teams observed in actual VCT events are admitted automatically.
    for row in read_csv(matches_path):
        if not clean(row.get("tournament")).casefold().startswith("vct "):
            continue
        for field in ("team_a", "team_b"):
            canonical = clean(row.get(field))
            if canonical:
                lookup[norm(canonical)] = canonical
    for alias, canonical in STATIC_TEAM_ALIASES.items():
        if norm(canonical) in lookup:
            lookup[norm(alias)] = lookup[norm(canonical)]
    for row in read_csv(aliases_path):
        if clean(row.get("active", "1")).casefold() in ("0", "false", "no"):
            continue
        canonical = clean(row.get("canonical"))
        if not canonical:
            continue
        lookup[norm(canonical)] = canonical
        for alias in clean(row.get("aliases")).split("|"):
            if clean(alias):
                lookup[norm(alias)] = canonical
    return lookup


def normalize_item(item: dict[str, Any], context: dict[str, str], team_lookup: dict[str, str], default_market: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    flat = flatten(item)
    player = clean(first(flat, (
        "player_name", "new_player.name", "player.name", "athlete.name",
        "attributes.player_name", "attributes.name", "name",
    )))
    line_value = first(flat, ("line", "line_score", "attributes.line_score", "value", "projection"))
    line = number(line_value)
    team_raw = clean(first(flat, (
        "team", "team_name", "player_team", "new_player.team", "player.team",
        "attributes.team", "team_abbreviation", "attributes.description", "description",
    )))
    opponent_raw = clean(first(flat, (
        "opponent", "opponent_name", "attributes.opponent", "game.opponent",
        "match.opponent", "opponent_abbreviation",
    )))
    matchup = clean(first(flat, ("matchup", "game", "game_name", "event_name", "attributes.matchup")))
    left, right = parse_matchup(matchup)
    if not team_raw and left:
        team_raw = left
    if not opponent_raw and right:
        opponent_raw = right
    league = clean(first(flat, ("league_name", "league.name", "league", "sport", "attributes.league")))
    event = clean(first(flat, ("event", "tournament", "competition", "event.name", "attributes.event")))
    stat_type = clean(first(flat, ("stat_type", "stat", "market", "category", "attributes.stat_type")))
    projection_type = clean(first(flat, ("projection_type", "attributes.projection_type", "odds_type")))
    start_time = clean(first(flat, ("start_time", "game_time", "scheduled_at", "date", "attributes.start_time")))
    source_record_id = clean(first(flat, ("id", "projection_id", "attributes.id"))) or hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16]
    raw_reject = {
        "captured_at": context["captured_at"], "source_record_id": source_record_id,
        "player": player, "team": team_raw, "opponent": opponent_raw,
        "line": "" if line is None else line, "league": league, "event": event,
        "stat_type": stat_type, "raw_file": context["raw_file"],
    }
    league_key = norm(league + " " + event)
    if "valorant" not in league_key and not re.search(r"(^|[^a-z])val([^a-z]|$)", (league + " " + event).casefold()) and "vct" not in league_key:
        return None, {**raw_reject, "reason": "not identified as Valorant/VCT"}
    if "kill" not in stat_type.casefold():
        return None, {**raw_reject, "reason": "not a kill market"}
    if not player or line is None:
        return None, {**raw_reject, "reason": "missing player or numeric line"}
    team = team_lookup.get(norm(team_raw), "")
    opponent = team_lookup.get(norm(opponent_raw), "")
    if not team or not opponent:
        missing = []
        if not team: missing.append(f"team '{team_raw}'")
        if not opponent: missing.append(f"opponent '{opponent_raw}'")
        return None, {**raw_reject, "reason": "not in current VCT team allowlist: " + ", ".join(missing)}
    stat_key = stat_type.casefold().replace("–", "-")
    if "map 1" in stat_key and not any(token in stat_key for token in ("1-2", "1 & 2", "1 and 2", "maps 1 2")):
        market = "map_1"
    elif any(token in stat_key for token in ("1-2", "1 & 2", "1 and 2", "maps 1 2", "maps 1 and 2")):
        market = "maps_1_2"
    else:
        market = default_market
    match_date = parse_match_date(start_time, clean(first(flat, ("match_date",))))
    if not match_date:
        return None, {**raw_reject, "reason": "missing match/start date"}
    snapshot_seed = "|".join((context["source_run_id"], source_record_id, clean(line), market))
    snapshot_id = hashlib.sha256(snapshot_seed.encode()).hexdigest()[:24]
    normalized = {
        "match_date": match_date, "team": team, "opponent": opponent,
        "player": player, "line": line, "team_odds": "", "opponent_odds": "",
        "market": market, "map1": "", "map2": "", "role_override": "",
        "snapshot_id": snapshot_id, "captured_at": context["captured_at"],
        "source": "Apify", "source_record_id": source_record_id, "league": league,
        "event": event, "start_time": start_time, "projection_type": projection_type,
        "raw_file": context["raw_file"],
    }
    return normalized, raw_reject


def token_value(path: Path) -> str:
    value = clean(os.environ.get("APIFY_TOKEN"))
    if value:
        return value
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def apify_json(url: str, token: str) -> Any:
    separator = "&" if "?" in url else "?"
    request = urllib.request.Request(url + separator + urllib.parse.urlencode({"token": token}), headers={"User-Agent": "valorant-role-model/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_latest(actor_id: str, dataset_id: str, token: str) -> tuple[list[dict[str, Any]], str, str]:
    run_id = "dataset-" + dataset_id if dataset_id else ""
    if not dataset_id:
        actor = urllib.parse.quote(actor_id, safe="~")
        payload = apify_json(f"https://api.apify.com/v2/acts/{actor}/runs/last?status=SUCCEEDED", token)
        run = payload.get("data") or {}
        dataset_id = clean(run.get("defaultDatasetId"))
        run_id = clean(run.get("id"))
        if not dataset_id:
            raise RuntimeError("latest successful actor run has no default dataset")
    items = apify_json(f"https://api.apify.com/v2/datasets/{urllib.parse.quote(dataset_id)}/items?clean=true&format=json", token)
    if not isinstance(items, list):
        raise RuntimeError("Apify dataset response was not a JSON list")
    return [item for item in items if isinstance(item, dict)], run_id, dataset_id


def ingest(items: list[dict[str, Any]], source_run_id: str, raw_path: Path, args: argparse.Namespace) -> tuple[int, int]:
    captured_at = datetime.now(timezone.utc).isoformat()
    context = {"captured_at": captured_at, "source_run_id": source_run_id, "raw_file": str(raw_path)}
    team_lookup = load_team_lookup(Path(args.vct_matches), Path(args.team_aliases))
    accepted, rejected = [], []
    for item in items:
        row, reject = normalize_item(item, context, team_lookup, args.default_market)
        (accepted if row else rejected).append(row or reject)
    history = Path(args.history_dir)
    observed_fields = sorted({field for item in items for field in flatten(item)})
    history.mkdir(parents=True, exist_ok=True)
    (history / "apify_schema_fields.txt").write_text("\n".join(observed_fields) + "\n", encoding="utf-8")
    added = append_unique(history / "line_snapshots.csv", accepted, SNAPSHOT_FIELDS, ("snapshot_id",))
    append_unique(history / "rejected_lines.csv", rejected, REJECT_FIELDS, ("captured_at", "source_record_id", "reason"))
    # The latest accepted snapshot for each player/match/market becomes prediction input.
    latest: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in read_csv(history / "line_snapshots.csv"):
        key = tuple(clean(row.get(k)).casefold() for k in ("match_date", "team", "opponent", "player", "market"))
        if key not in latest or clean(row.get("captured_at")) > clean(latest[key].get("captured_at")):
            latest[key] = row
    write_csv(Path(args.output), latest.values(), FUTURE_FIELDS)
    print(f"accepted {len(accepted)} VCT kill lines ({added} new snapshots); rejected {len(rejected)} rows")
    print(f"wrote {len(latest)} latest lines to {args.output}")
    print(f"observed Apify fields written to {history / 'apify_schema_fields.txt'}")
    return len(accepted), len(rejected)


def pull_command(args: argparse.Namespace) -> None:
    token = token_value(Path(args.token_file))
    if not token:
        raise SystemExit(f"Missing Apify token. Put it in {args.token_file} or set APIFY_TOKEN.")
    items, run_id, dataset_id = fetch_latest(args.actor_id, args.dataset_id, token)
    raw_dir = Path(args.history_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"apify_{run_id or dataset_id}.json"
    raw_path.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    ingest(items, run_id or dataset_id, raw_path, args)


def import_file_command(args: argparse.Namespace) -> None:
    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise SystemExit("Input JSON must be a list or an object containing an items list.")
    run_id = clean(args.run_id) or f"file-{hashlib.sha256(source.read_bytes()).hexdigest()[:16]}"
    ingest([x for x in items if isinstance(x, dict)], run_id, source, args)


def lock_command(args: argparse.Namespace) -> None:
    predictions = read_csv(Path(args.predictions))
    model_hash = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    locked_at = datetime.now(timezone.utc).isoformat()
    fields = list(dict.fromkeys(FUTURE_FIELDS + LEDGER_EXTRA_FIELDS))
    rows = []
    for row in predictions:
        rows.append({**row, "model_sha256": model_hash, "locked_at": locked_at})
    added = append_unique(Path(args.ledger), rows, fields, ("snapshot_id", "model_sha256"))
    print(f"locked {added} new predictions in {args.ledger}")


def team_pair(row: dict[str, str]) -> frozenset[str]:
    return frozenset((norm(row.get("team_a")), norm(row.get("team_b"))))


def score_command(args: argparse.Namespace) -> None:
    ledger = read_csv(Path(args.ledger))
    matches = read_csv(Path(args.matches))
    player_maps = read_csv(Path(args.player_maps))
    match_by_id = {clean(row.get("series_id")): row for row in matches}
    match_candidates: dict[frozenset[str], list[dict[str, str]]] = defaultdict(list)
    for row in matches:
        match_candidates[team_pair(row)].append(row)
    stats: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in player_maps:
        stats[(clean(row.get("series_id")), norm(row.get("player")))].append(row)
    scored = []
    for prediction in ledger:
        match = match_by_id.get(clean(prediction.get("match_id"))) if clean(prediction.get("match_id")) else None
        if not match:
            pair = frozenset((norm(prediction.get("team")), norm(prediction.get("opponent"))))
            candidates = match_candidates.get(pair, [])
            target_date = clean(prediction.get("match_date"))
            exact = [row for row in candidates if clean(row.get("match_date")) == target_date]
            match = exact[0] if len(exact) == 1 else None
        if not match:
            continue
        maps = stats.get((clean(match.get("series_id")), norm(prediction.get("matched_player") or prediction.get("player"))), [])
        wanted = {1} if clean(prediction.get("market")) == "map_1" else {1, 2}
        chosen = [row for row in maps if int(float(clean(row.get("map_order")) or 0)) in wanted]
        if len({int(float(row["map_order"])) for row in chosen}) != len(wanted):
            continue
        actual = sum(float(row["kills"]) for row in chosen)
        line = float(prediction["line"]); predicted = float(prediction["predicted_kills"])
        outcome = "push" if actual == line else ("more" if actual > line else "less")
        direction = clean(prediction.get("direction"))
        confidence = max(float(prediction.get("probability_more") or 0), float(prediction.get("probability_less") or 0))
        correct = "" if outcome == "push" else int(direction == outcome)
        recommended = clean(prediction.get("recommendation")) in ("more", "less")
        scored.append({
            **prediction, "match_id": match.get("series_id", ""), "actual_kills": actual,
            "outcome": outcome, "correct": correct, "recommended_pick": int(recommended),
            "recommended_correct": correct if recommended else "", "confidence": confidence,
            "absolute_error": abs(actual - predicted),
            "brier": "" if outcome == "push" else (float(prediction.get("probability_more") or 0) - int(outcome == "more")) ** 2,
        })
    score_fields = list(dict.fromkeys((list(scored[0]) if scored else FUTURE_FIELDS + LEDGER_EXTRA_FIELDS) + [
        "match_id", "actual_kills", "outcome", "correct", "recommended_pick",
        "recommended_correct", "confidence", "absolute_error", "brier",
    ]))
    write_csv(Path(args.output), scored, score_fields)
    non_push = [row for row in scored if row["outcome"] != "push"]
    recommendations = [row for row in non_push if row["recommended_pick"]]
    summary = {
        "scored_predictions": len(scored), "non_push_predictions": len(non_push),
        "direction_accuracy": sum(int(row["correct"]) for row in non_push) / len(non_push) if non_push else None,
        "recommended_picks": len(recommendations),
        "recommendation_accuracy": sum(int(row["recommended_correct"]) for row in recommendations) / len(recommendations) if recommendations else None,
        "mae": sum(float(row["absolute_error"]) for row in scored) / len(scored) if scored else None,
        "brier_score": sum(float(row["brier"]) for row in non_push) / len(non_push) if non_push else None,
        "thresholds": [],
    }
    for threshold in (.50, .52, .55, .57, .60, .65, .70):
        subset = [row for row in non_push if float(row["confidence"]) >= threshold]
        summary["thresholds"].append({
            "minimum_probability": threshold, "picks": len(subset),
            "accuracy": sum(int(row["correct"]) for row in subset) / len(subset) if subset else None,
        })
    Path(args.summary).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def shared_ingest_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--history-dir", default="./line_history")
    parser.add_argument("--output", default="./apify_future_lines.csv")
    parser.add_argument("--vct-matches", default="./api_data/matches.csv")
    parser.add_argument("--team-aliases", default="./vct_team_aliases.csv")
    parser.add_argument("--default-market", choices=("map_1", "maps_1_2"), default="maps_1_2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive Apify PrizePicks lines and maintain a locked accuracy ledger")
    sub = parser.add_subparsers(dest="command", required=True)
    pull = sub.add_parser("pull", help="download the latest successful Apify actor dataset")
    pull.add_argument("--actor-id", default="4AmgQeem8dEgMEiRF")
    pull.add_argument("--dataset-id", default="")
    pull.add_argument("--token-file", default="./.apify_token")
    shared_ingest_args(pull);pull.set_defaults(func=pull_command)
    imp = sub.add_parser("import-file", help="import an exported Apify JSON file")
    imp.add_argument("input");imp.add_argument("--run-id", default="")
    shared_ingest_args(imp);imp.set_defaults(func=import_file_command)
    lock = sub.add_parser("lock", help="append model predictions to the immutable prediction ledger")
    lock.add_argument("--predictions", default="./apify_predictions_latest.csv")
    lock.add_argument("--model", default="./model_output/production_model.json")
    lock.add_argument("--ledger", default="./line_history/prediction_ledger.csv")
    lock.set_defaults(func=lock_command)
    score = sub.add_parser("score", help="attach completed VLR results and calculate accuracy")
    score.add_argument("--ledger", default="./line_history/prediction_ledger.csv")
    score.add_argument("--matches", default="./api_data/matches.csv")
    score.add_argument("--player-maps", default="./api_data/player_maps.csv")
    score.add_argument("--output", default="./line_history/scored_predictions.csv")
    score.add_argument("--summary", default="./line_history/accuracy_summary.json")
    score.set_defaults(func=score_command)
    args = parser.parse_args();args.func(args)


if __name__ == "__main__":
    main()
