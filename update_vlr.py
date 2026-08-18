#!/usr/bin/env python3
"""Incrementally import completed VLR series through the authorized vlrdevapi client.

Raw responses are cached for reproducibility. Validated, deduplicated rows are stored
in two small CSV files that model.py can layer on top of the original Kaggle archive.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MATCH_FIELDS = [
    "series_id", "match_date", "year", "event_id", "tournament", "stage",
    "match_type", "match_name", "team_a_id", "team_a", "team_b_id", "team_b",
    "team_a_score", "team_b_score", "best_of", "status", "source_url",
]
PLAYER_MAP_FIELDS = [
    "series_id", "game_id", "map_order", "map", "rounds", "player_id",
    "player", "team_id", "team", "agents", "rating", "acs", "kills", "deaths",
    "assists", "adr", "headshot_percent", "first_kills", "first_deaths",
]
SELECTION_FIELDS = ["last_seen", "match_id", "event", "team1", "team2", "scope", "decision", "reason"]

TIER1_EVENT_KEYWORDS = (
    "valorant champions tour", "champions tour", "vct ", "valorant masters",
    "masters ", "valorant champions", "vct masters", "vct champions",
)
TIER2_EVENT_KEYWORDS = (
    "challengers", "ascension", "china evolution series", "evolution series",
)
EXCLUDED_EVENT_KEYWORDS = (
    "game changers", "collegiate", "college", "university", "academy", "premier",
)


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def completed_series_ids(payload: Any) -> list[int]:
    """Find series IDs without assuming a particular pagination wrapper."""
    found: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            # Series detail responses call this series_id; completed-list
            # entries in vlrdevapi 2.3 call the same VLR identifier match_id.
            for id_field in ("series_id", "match_id"):
                if id_field in value:
                    try:
                        found.add(int(value[id_field]))
                    except (TypeError, ValueError):
                        pass
            # Completed-match models may call the match identifier simply `id`.
            if "id" in value and any(k in value for k in ("team1", "team2", "team1_name", "team2_name")):
                try:
                    found.add(int(value["id"]))
                except (TypeError, ValueError):
                    pass
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(plain(payload))
    return sorted(i for i in found if i > 0)


def completed_match_entries(payload: Any) -> list[dict[str, Any]]:
    """Return the match rows from a vlrdevapi completed-page model."""
    data = plain(payload)
    if isinstance(data, dict) and isinstance(data.get("matches"), list):
        return [row for row in data["matches"] if isinstance(row, dict)]
    return []


def competition_allowed(event: str, scope: str) -> tuple[bool, str]:
    """Filter automatic imports to betting-relevant professional events."""
    normalized = " ".join(str(event or "").casefold().split())
    excluded = next((word for word in EXCLUDED_EVENT_KEYWORDS if word in normalized), None)
    if scope == "all":
        return True, "all competitions requested"
    if excluded:
        return False, f"excluded category: {excluded}"
    if any(word in normalized for word in TIER1_EVENT_KEYWORDS):
        return True, "tier-1 VCT event"
    if scope == "relevant" and any(word in normalized for word in TIER2_EVENT_KEYWORDS):
        return True, "approved tier-2 event"
    expected = "VCT/Champions/Masters" if scope == "tier1" else "VCT or approved Challengers/Ascension"
    return False, f"event did not match {expected}"


def listed_team_name(entry: dict[str, Any], field: str) -> str:
    team = entry.get(field) or {}
    return str(team.get("name", "")) if isinstance(team, dict) else str(team)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def merge_rows(path: Path, rows: Iterable[dict[str, Any]], fields: list[str], key_fields: tuple[str, ...]) -> int:
    merged = {tuple(str(row.get(k, "")) for k in key_fields): row for row in read_rows(path)}
    before = len(merged)
    for row in rows:
        normalized = {field: row.get(field, "") for field in fields}
        merged[tuple(str(normalized.get(k, "")) for k in key_fields)] = normalized
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(merged.values(), key=lambda r: tuple(str(r.get(k, "")) for k in key_fields)))
    temporary.replace(path)
    return len(merged) - before


def parse_date(value: Any) -> tuple[str, int]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("series has no match datetime")
    normalized = text.replace("Z", "+00:00")
    try:
        moment = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"unrecognized match datetime: {text}") from exc
    return moment.date().isoformat(), moment.year


def normalize_bundle(bundle: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info = bundle.get("info") or {}
    if str(info.get("status", "")).casefold() != "completed":
        raise ValueError("series is not completed")
    series_id = int(info.get("series_id") or bundle.get("series_id") or 0)
    if not series_id:
        raise ValueError("series_id is missing")
    match_date, year = parse_date(info.get("datetime"))
    team_a, team_b = info.get("team1") or {}, info.get("team2") or {}
    if not team_a.get("name") or not team_b.get("name"):
        raise ValueError("both team names are required")
    match_name = f"{team_a['name']} vs {team_b['name']}"
    match = {
        "series_id": series_id, "match_date": match_date, "year": year,
        "event_id": info.get("event_id", ""), "tournament": info.get("event_name") or "VLR Event",
        "stage": info.get("stage") or "Unknown Stage", "match_type": info.get("bracket") or "Match",
        "match_name": match_name, "team_a_id": team_a.get("id", ""), "team_a": team_a["name"],
        "team_b_id": team_b.get("id", ""), "team_b": team_b["name"],
        "team_a_score": info.get("score1", 0), "team_b_score": info.get("score2", 0),
        "best_of": info.get("best_of", ""), "status": info.get("status", ""),
        "source_url": f"https://www.vlr.gg/{series_id}/",
    }
    player_payloads = {str(p.get("game_id")): p for p in bundle.get("players", [])}
    round_payloads = {str(p.get("game_id")): p for p in bundle.get("rounds", [])}
    player_maps: list[dict[str, Any]] = []
    for game in info.get("games", []):
        if not game.get("played", True) or not game.get("game_id"):
            continue
        game_id = str(game["game_id"])
        stats = player_payloads.get(game_id)
        if not stats:
            raise ValueError(f"missing player stats for game {game_id}")
        rounds_data = round_payloads.get(game_id, {}).get("rounds", [])
        rounds = len(rounds_data) or int(game.get("team1_score") or 0) + int(game.get("team2_score") or 0)
        if rounds <= 0:
            raise ValueError(f"missing round count for game {game_id}")
        for side in ("team1", "team2"):
            team = stats.get(side) or {}
            if not team.get("team_name"):
                raise ValueError(f"missing team assignment for game {game_id}")
            for player in team.get("players", []):
                overall = ((player.get("stats") or {}).get("overall") or {})
                if overall.get("kills") is None:
                    raise ValueError(f"missing kills for {player.get('name') or 'player'} in game {game_id}")
                player_maps.append({
                    "series_id": series_id, "game_id": game_id, "map_order": game.get("order", 0),
                    "map": game.get("map_name") or stats.get("map_name") or "Unknown",
                    "rounds": rounds, "player_id": player.get("player_id", ""), "player": player.get("name", ""),
                    "team_id": team.get("team_id", ""), "team": team.get("team_name", ""),
                    "agents": "/".join(player.get("agents") or []), "rating": overall.get("rating", ""),
                    "acs": overall.get("acs", ""), "kills": overall.get("kills", ""),
                    "deaths": overall.get("deaths", ""), "assists": overall.get("assists", ""),
                    "adr": overall.get("adr", ""), "headshot_percent": overall.get("hs_percent", ""),
                    "first_kills": overall.get("first_kills", ""), "first_deaths": overall.get("first_deaths", ""),
                })
    if not player_maps:
        raise ValueError("series contains no played-map player rows")
    return match, player_maps


def import_bundle(bundle: dict[str, Any], output: Path) -> tuple[int, int]:
    match, players = normalize_bundle(bundle)
    matches_added = merge_rows(output / "matches.csv", [match], MATCH_FIELDS, ("series_id",))
    players_added = merge_rows(
        output / "player_maps.csv", players, PLAYER_MAP_FIELDS,
        ("series_id", "game_id", "player_id", "player", "team_id"),
    )
    return matches_added, players_added


def fetch_bundle(client: Any, series_id: int) -> dict[str, Any]:
    series = client.series(series_id)
    info = plain(series.info())
    players, rounds = [], []
    for game in info.get("games", []):
        if game.get("played", True) and game.get("game_id"):
            game_id = game["game_id"]
            players.append(plain(series.players(game_id=game_id)))
            rounds.append(plain(series.rounds(game_id=game_id)))
    return {
        "source": "vlrdevapi", "fetched_at": datetime.now(timezone.utc).isoformat(),
        "series_id": series_id, "info": info, "players": players, "rounds": rounds,
    }


def fetch_command(args: argparse.Namespace) -> None:
    try:
        from vlrdevapi import VLRClient
    except ImportError as exc:
        raise SystemExit("vlrdevapi is not installed. Double-click setup_mac.command first.") from exc
    output = Path(args.output)
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    requested = list(dict.fromkeys(args.series_id or []))
    automatic = not requested
    selection_rows: list[dict[str, Any]] = []
    seen_at = datetime.now(timezone.utc).isoformat()
    with VLRClient(timeout=args.timeout, requests_per_second=args.requests_per_second) as client:
        if automatic:
            for page in range(1, args.pages + 1):
                try:
                    completed_page = client.matches.completed(page=page)
                except Exception as exc:
                    raise SystemExit(f"Could not download completed VLR page {page}: {exc}") from exc
                entries = completed_match_entries(completed_page)
                page_ids: list[int] = []
                for entry in entries:
                    try:
                        match_id = int(entry.get("match_id") or entry.get("series_id") or 0)
                    except (TypeError, ValueError):
                        continue
                    if not match_id:
                        continue
                    event = str(entry.get("event", ""))
                    allowed, reason = competition_allowed(event, args.competition_scope)
                    selection_rows.append({
                        "last_seen": seen_at, "match_id": match_id, "event": event,
                        "team1": listed_team_name(entry, "team1"), "team2": listed_team_name(entry, "team2"),
                        "scope": args.competition_scope, "decision": "include" if allowed else "exclude", "reason": reason,
                    })
                    if allowed:
                        page_ids.append(match_id)
                print(
                    f"completed page {page}: considered {len(entries)} matches; "
                    f"{len(page_ids)} passed the {args.competition_scope} competition filter"
                )
                requested.extend(page_ids)
            requested = list(dict.fromkeys(requested))
            if not requested:
                raise SystemExit(
                    "No completed series IDs were found. The VLR listing format may have changed; "
                    "try --series-id with the number from a VLR match URL."
                )
        known = {row.get("series_id") for row in read_rows(output / "matches.csv")}
        selected = [sid for sid in requested if args.refresh or str(sid) not in known]
        if args.max_new:
            selected = selected[: args.max_new]
        selected_set = set(selected)
        if automatic:
            for row in selection_rows:
                if row["decision"] != "include":
                    continue
                match_id = int(row["match_id"])
                if not args.refresh and str(match_id) in known:
                    row["decision"], row["reason"] = "skip", "already imported"
                elif match_id not in selected_set:
                    row["decision"], row["reason"] = "defer", f"beyond max-new limit ({args.max_new})"
            merge_rows(output / "selection_log.csv", selection_rows, SELECTION_FIELDS, ("match_id",))
            print(f"selection details written to {output / 'selection_log.csv'}")
        else:
            print("manual --series-id supplied: competition filtering bypassed")
        print(f"selected {len(selected)} new series after filtering, deduplication, and limit")
        added_matches = added_players = 0
        failures: list[str] = []
        for series_id in selected:
            try:
                bundle = fetch_bundle(client, series_id)
                raw_path = raw_dir / f"series_{series_id}.json"
                raw_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
                ma, pa = import_bundle(bundle, output)
                added_matches += ma
                added_players += pa
                print(f"imported series {series_id}: {pa} player-map rows")
            except Exception as exc:  # keep one malformed series from stopping the batch
                failures.append(f"{series_id}: {exc}")
                print(f"skipped series {series_id}: {exc}")
    print(f"update complete: {added_matches} matches and {added_players} player-map rows added")
    if failures:
        (output / "last_failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"{len(failures)} series failed validation; see {output / 'last_failures.txt'}")
        if selected and added_matches == 0:
            raise SystemExit("No selected series could be imported. Review last_failures.txt.")
    elif not selected:
        print("No new matches were needed; every completed series on the requested pages is already stored.")


def import_file_command(args: argparse.Namespace) -> None:
    bundle = json.loads(Path(args.input).read_text(encoding="utf-8"))
    ma, pa = import_bundle(bundle, Path(args.output))
    print(f"imported {ma} match and {pa} player-map rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the V5 model dataset from completed VLR series")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="fetch recent completed matches or specified series IDs")
    fetch.add_argument("--output", default="./api_data")
    fetch.add_argument("--pages", type=int, default=1)
    fetch.add_argument("--max-new", type=int, default=25)
    fetch.add_argument("--series-id", type=int, action="append")
    fetch.add_argument(
        "--competition-scope", choices=("relevant", "tier1", "all"), default="relevant",
        help="automatic match filter: VCT plus approved tier 2 (default), tier 1 only, or all VLR events",
    )
    fetch.add_argument("--refresh", action="store_true")
    fetch.add_argument("--timeout", type=int, default=30)
    fetch.add_argument("--requests-per-second", type=float, default=2.0)
    fetch.set_defaults(func=fetch_command)
    imp = sub.add_parser("import-file", help="validate/import a cached raw series JSON")
    imp.add_argument("input")
    imp.add_argument("--output", default="./api_data")
    imp.set_defaults(func=import_file_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
