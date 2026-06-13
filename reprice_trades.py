"""
reprice_trades.py — fix the 7-day entry-lag bug WITHOUT re-running the backtest.

THE BUG: backtest.py fetched one 9-day window of 1min bars per candidate. FMP
caps rows per request and serves the most-recent end of the range first, so the
early sessions were silently missing and `entry_day` (first session with bars)
landed ~T+5 instead of the intended next-session open. Every return column in
backtest_trades.csv therefore measures the wrong trade.

THE FIX: signal columns (composite, level_z, drift_z, signal_blend, n_priors)
don't depend on pricing — they carry over untouched, so no FinBERT, no
transcript fetches. This script re-prices ONLY the entry + return columns,
fetching bars ONE SESSION AT A TIME (≈390 rows of 1min, far under any cap):

  decision = 22:00 ET on the transcript date (date-only timestamps -> same day)
  entry    = 09:30 open of the FIRST session strictly after the decision date,
             walking forward day-by-day until a session with RTH bars is found
  exits    = open+30/60/120m, same-day close, T+1 close (next session w/ bars)

Output: backtest_trades_repriced.csv — same schema, feed it straight to the
analyzer:

    export FMP_API_KEY=...
    python reprice_trades.py                  # resumes if interrupted
    python backtest_analyze.py backtest_trades_repriced.csv

Runtime: ~2-4 requests/row x 4,959 rows ≈ 10-20k requests. Serial HTTP at
~0.15-0.25s/request -> roughly 40-80 min. Checkpoints every 200 rows; re-running
skips completed rows. Known remaining caveats (unchanged by this script):
  * report_timing is still degenerate (all 'BMO'); genuine before-open reports
    are entered one session late. The AMC slice question stays open until the
    timing relabel lands.
  * Foreign-suffixed listings (.KS/.TO/.L/...) keep local-currency caps and
    non-ET sessions; use --us-only to drop them (recommended).
"""

import argparse
import os
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from fmp_client import FMPClient

IN_PATH = "backtest_trades.csv"
OUT_PATH = "backtest_trades_repriced.csv"
CHECKPOINT_EVERY = 200
MAX_FORWARD_DAYS = 10          # give up if no session found within this many days
RTH_OPEN, RTH_CLOSE = time(9, 30), time(16, 0)
HORIZON_MINUTES = {"ret_open+30m": 30, "ret_open+60m": 60, "ret_open+120m": 120}
RUN_TIME_ET = time(22, 0)

PRICING_COLS = ["entry_date", "entry_price", "ret_open+30m", "ret_open+60m",
                "ret_open+120m", "ret_close", "ret_t1_close"]


def decision_date(transcript_dt: datetime) -> date:
    """Nightly run that first catches the transcript (same semantics as backtest.py)."""
    d = transcript_dt.date()
    if transcript_dt.time() >= RUN_TIME_ET:
        d += timedelta(days=1)
    return d


def rth(bars):
    return [b for b in bars if RTH_OPEN <= b["dt"].time() <= RTH_CLOSE]


def bar_at_or_after(day_bars, target: time):
    for b in day_bars:
        if b["dt"].time() >= target:
            return b
    return None


def next_session(client, symbol, after: date, limit=MAX_FORWARD_DAYS):
    """First date strictly after `after` with RTH bars, fetched ONE DAY at a time."""
    d = after + timedelta(days=1)
    for _ in range(limit):
        if d.weekday() < 5:                      # skip weekends without spending a request
            day_bars = rth(client.intraday_bars(symbol, "1min", d, d))
            if day_bars:
                return d, day_bars
        d += timedelta(days=1)
    return None, None


def price_row(client, symbol, transcript_dt):
    """Correct next-session-open pricing. Returns dict of PRICING_COLS or None."""
    dec = decision_date(transcript_dt)

    entry_day, eday = next_session(client, symbol, dec)
    if entry_day is None:
        return None
    entry_bar = bar_at_or_after(eday, RTH_OPEN)
    if not entry_bar or not entry_bar.get("open"):
        return None
    entry_price = entry_bar["open"]
    open_dt = datetime.combine(entry_day, RTH_OPEN)

    out = {"entry_date": entry_day.isoformat(), "entry_price": round(entry_price, 4)}
    for col, mins in HORIZON_MINUTES.items():
        b = bar_at_or_after(eday, (open_dt + timedelta(minutes=mins)).time())
        px = b["close"] if b else eday[-1]["close"]
        out[col] = round(px / entry_price - 1.0, 6) if px else None
    out["ret_close"] = (round(eday[-1]["close"] / entry_price - 1.0, 6)
                        if eday[-1]["close"] else None)

    t1_day, t1bars = next_session(client, symbol, entry_day)
    out["ret_t1_close"] = (round(t1bars[-1]["close"] / entry_price - 1.0, 6)
                           if t1bars and t1bars[-1]["close"] else None)
    return out


def main():
    ap = argparse.ArgumentParser(description="Re-price backtest candidates with "
                                 "correct next-session-open entries.")
    ap.add_argument("--us-only", action="store_true",
                    help="Drop foreign-suffixed symbols (recommended).")
    ap.add_argument("--max-rows", type=int, default=None, help="Pilot cap.")
    args = ap.parse_args()

    df = pd.read_csv(IN_PATH)
    df["transcript_dt"] = pd.to_datetime(df["transcript_dt"])
    if args.us_only:
        before = len(df)
        df = df[~df.symbol.str.contains(r"\.", regex=True)].reset_index(drop=True)
        print(f"--us-only: dropped {before - len(df)} foreign-suffixed rows; "
              f"{len(df)} remain.")

    # resume support: rows already in the output are skipped
    done = set()
    if os.path.exists(OUT_PATH):
        prev = pd.read_csv(OUT_PATH)
        done = set(zip(prev.symbol, prev.year, prev.quarter))
        print(f"Resuming: {len(done)} rows already re-priced.")
        rows_out = prev.to_dict("records")
    else:
        rows_out = []

    client = FMPClient()
    todo = df[~df.apply(lambda r: (r.symbol, r.year, r.quarter) in done, axis=1)]
    if args.max_rows:
        todo = todo.head(args.max_rows)

    failed = 0
    for i, (_, r) in enumerate(tqdm(todo.iterrows(), total=len(todo), unit="row"), 1):
        try:
            priced = price_row(client, r.symbol, r.transcript_dt)
        except Exception as e:
            tqdm.write(f"  > {r.symbol} {r.year}Q{r.quarter}: {e}")
            priced = None
        if priced is None:
            failed += 1
            continue
        rec = r.to_dict()
        rec["transcript_dt"] = r.transcript_dt.isoformat()
        rec.update(priced)
        rows_out.append(rec)

        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows_out).to_csv(OUT_PATH, index=False)

    out = pd.DataFrame(rows_out)
    out.to_csv(OUT_PATH, index=False)

    # --- sanity report: the lag distribution is the proof the fix worked ---
    lag = (pd.to_datetime(out.entry_date) -
           pd.to_datetime(out.transcript_dt).dt.normalize()).dt.days
    print(f"\nWrote {len(out)} re-priced rows -> {OUT_PATH}  ({failed} unpriceable)")
    print("Entry-lag distribution (calendar days):",
          lag.value_counts().sort_index().to_dict())
    print("Expected: mass at 1 (weekdays) and 3 (Fri->Mon); if 7 still dominates,"
          " stop and re-check the bar fetch.")
    print(f"ret_t1_close missing: {out['ret_t1_close'].isna().mean()*100:.1f}% "
          "(should fall to ~0 from 18.3%)")
    print(f"\nNext: python backtest_analyze.py {OUT_PATH}")


if __name__ == "__main__":
    main()