import json
import os
from datetime import datetime

SCORES_PATH = "transcript_scores.json"

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