import csv
import json
import os
from datetime import datetime

SCORES_PATH = "transcript_scores.json"
POSITIONS_PATH = "positions.json"
PERFORMANCE_LOG_PATH = "performance_log.csv"
_PERF_HEADERS = ["date", "portfolio_value", "num_positions", "benchmark_price"]

def load_scores(path: str = SCORES_PATH) -> list:
    # first run
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = json.load(f) # parses JSON data into python dict
    # pulls "scored_transcripts" list from dict, returns [] if missing
    return data.get("scored_transcripts", [])

def save_scores(records: list, path: str = SCORES_PATH) -> None:
    payload = {
        "scored_transcripts": records,
        "last_updated": str(datetime.today().date()),
    }

    # overwrites existing content and dumps payload to SCORES_PATH
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def get_scored_keys(records: list) -> set:
    # set of tuples containing (ticker, date) per record
    # r["date"][:10] slices the date string to YYYY-MM-DD
    return {(r["ticker"], r["date"][:10]) for r in records}

def load_positions(path: str = POSITIONS_PATH) -> dict:
    # returns {ticker: {"entry_date": str, "entry_price": float}}
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def save_positions(positions: dict, path: str = POSITIONS_PATH) -> None:
    with open(path, "w") as f:
        json.dump(positions, f, indent=2)

def log_performance(
    date: str,
    portfolio_value: float,
    num_positions: int,
    benchmark_price: float,
    path: str = PERFORMANCE_LOG_PATH,
) -> None:
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PERF_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "date": date,
            "portfolio_value": portfolio_value,
            "num_positions": num_positions,
            "benchmark_price": benchmark_price, # using SPY as benchmark
        })