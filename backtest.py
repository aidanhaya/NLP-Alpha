"""
backtest.py — generate candidate-level trade data for the NLP earnings-sentiment backtest.

This is the EXPENSIVE, run-once step. For every in-window earnings transcript in the
>$500M (point-in-time) universe it:
  1. scores sentiment with FinBERT (reusing sentiment_scoring.py), with on-disk caching,
  2. computes three point-in-time trigger quantities (level-z, drift-z, signal blend),
  3. determines the next trading day's open as ENTRY (report time-of-day doesn't matter —
     only the calendar day the report landed on),
  4. prices forward GROSS returns at 1 / 3 / 5 trading days using daily close prices.

It does NOT apply thresholds, direction, or cost — those are cheap and live in
backtest_analyze.py, so you can sweep them without re-scoring. Output: backtest_trades.csv,
one row per priced candidate.

Run:
    export FMP_API_KEY=...
    python backtest.py --months 12                 # full run
    python backtest.py --months 12 --max-tickers 50  # pilot
"""

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
import numpy as np
from tqdm import tqdm

import preprocessing as pp          # reused: split_transcript, sentence_tokenize
import signal_constructor as sc     # reused: SentimentSignal (point-in-time drift/z)
import sentiment_scoring as scr     # reused: avoids importing torch unless we score
from fmp_client import FMPClient

# --- config ---

MIN_MARKET_CAP = 500_000_000        # point-in-time universe filter
MIN_PRIOR_TRANSCRIPTS = 4           # need a stable z baseline before trading a name
MAX_PRIOR_TRANSCRIPTS = 12          # cap priors used (bounds scoring; ~3yr quarterly)
SCORE_CACHE_PATH = "backtest_scores.json"   # {f"{SYM}:{year}:{q}": composite}
TRADES_PATH = "backtest_trades.csv"

# forward horizons measured in trading days from the entry day
HORIZON_DAYS = {"ret_1d": 1, "ret_3d": 3, "ret_5d": 5}
PRICE_WINDOW_DAYS = 21      # calendar-day buffer covering the 5-trading-day horizon

TRADE_FIELDS = [
    "symbol", "year", "quarter", "transcript_dt",
    "market_cap", "composite", "n_priors",
    "level_z", "drift_z", "signal_blend",
    "entry_date", "entry_price",
    "ret_1d", "ret_3d", "ret_5d",
    # per-trade market (SPY) matched returns
    "spy_ret_1d", "spy_ret_3d", "spy_ret_5d",
]

SPY_SYMBOL = "SPY"


@dataclass
class Candidate:
    symbol: str
    year: int
    quarter: int
    transcript_dt: str
    market_cap: float
    composite: float
    n_priors: int
    level_z: float
    drift_z: float
    signal_blend: float
    entry_date: str
    entry_price: float
    ret_1d: float
    ret_3d: float
    ret_5d: float
    spy_ret_1d: float
    spy_ret_3d: float
    spy_ret_5d: float

    def row(self) -> dict:
        return {
            "symbol": self.symbol, "year": self.year, "quarter": self.quarter,
            "transcript_dt": self.transcript_dt,
            "market_cap": self.market_cap, "composite": round(self.composite, 6),
            "n_priors": self.n_priors,
            "level_z": _r(self.level_z), "drift_z": _r(self.drift_z),
            "signal_blend": _r(self.signal_blend),
            "entry_date": self.entry_date, "entry_price": _r(self.entry_price, 4),
            "ret_1d": _r(self.ret_1d, 6), "ret_3d": _r(self.ret_3d, 6),
            "ret_5d": _r(self.ret_5d, 6),
            "spy_ret_1d": _r(self.spy_ret_1d, 6), "spy_ret_3d": _r(self.spy_ret_3d, 6),
            "spy_ret_5d": _r(self.spy_ret_5d, 6),
        }


def _r(v, nd=4):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


# --- scoring cache ---

def load_score_cache(path=SCORE_CACHE_PATH) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_score_cache(cache: dict, path=SCORE_CACHE_PATH):
    with open(path, "w") as f:
        json.dump(cache, f)


def composite_for(client, scorer, cache, symbol, year, quarter) -> float | None:
    """FinBERT composite for one transcript, cached. None if unavailable/empty."""
    key = f"{symbol}:{year}:{quarter}"
    if key in cache:
        return cache[key]
    t = client.get_transcript(symbol, year, quarter)
    if not t or not t["content"]:
        cache[key] = None
        return None
    split = pp.split_transcript(t["content"])
    tokenized = {
        "prepared": pp.sentence_tokenize(pp.clean_fmp_text(split["prepared"])),
        "qa": pp.sentence_tokenize(pp.clean_fmp_text(split["qa"])),
    }
    scored = scr.score_transcript(tokenized, symbol, str(year), scorer)
    cache[key] = scored["composite"]
    return scored["composite"]


# --- triggers ---

def point_in_time_z(prior_composites: list[float], current: float) -> float:
    """Level z-score of current composite vs PRIOR composites (priors only, no leak)."""
    if len(prior_composites) < 2:
        return 0.0
    arr = np.asarray(prior_composites, dtype=float)
    mu, sd = arr.mean(), arr.std()
    return float((current - mu) / sd) if sd > 0 else 0.0


def drift_and_signal(records_through_i: list[dict]) -> tuple[float, float]:
    """Reuse SentimentSignal for the blended drift-z and full signal, as of transcript i."""
    sig = sc.SentimentSignal()
    for rec in records_through_i:
        sig.add_score(rec)
    out = sig.get_signal(records_through_i[-1]["ticker"])
    return float(out.get("drift", 0.0)), float(out.get("signal", 0.0))


# --- entry timing ---

def price_trade(client, symbol, transcript_dt):
    """
    Returns (entry_date, entry_price, gross_returns_dict) or None.
    Entry is always the next trading day's open after the report date — only the calendar
    day the report landed on matters, not the time of day. Gross returns are LONG-signed;
    direction/cost applied later.
    """
    report_date = transcript_dt.date()
    win_start = report_date + timedelta(days=1)
    win_end = report_date + timedelta(days=PRICE_WINDOW_DAYS)
    bars = client.daily_bars(symbol, win_start, win_end)
    if not bars:
        return None

    entry_bar = bars[0]
    if not entry_bar["open"]:
        return None
    entry_day = entry_bar["date"]
    entry_price = entry_bar["open"]

    rets = {}
    for label, n in HORIZON_DAYS.items():
        px = bars[n]["close"] if len(bars) > n else None
        rets[label] = (px / entry_price - 1.0) if px else None

    return entry_day, entry_price, rets


# --- SPY matched-window pricing (per-trade market adjustment / beta context) ---

def fetch_spy_index(client, win_start: date, win_end: date) -> tuple[dict[date, dict], list[date]]:
    """Fetch SPY daily bars ONCE for the whole backtest window and return a
    ({date -> bar}, sorted_dates) index, so each trade's matched-window market return is
    priced from cache rather than a per-trade API call."""
    bars = client.daily_bars(SPY_SYMBOL, win_start, win_end)
    by_date = {b["date"]: b for b in bars}
    return by_date, sorted(by_date)


def spy_matched_returns(spy_by_date: dict[date, dict], spy_dates: list[date],
                        entry_day: date) -> dict | None:
    """SPY return over the SAME entry day and SAME horizons as the trade, priced with the
    identical open-anchored logic used in price_trade. Returns a dict keyed by the trade's
    horizon labels, or None if SPY bars for entry_day are missing (fail soft)."""
    if entry_day not in spy_by_date:
        return None
    entry_price = spy_by_date[entry_day]["open"]
    if not entry_price:
        return None
    idx = spy_dates.index(entry_day)

    rets = {}
    for label, n in HORIZON_DAYS.items():
        if idx + n < len(spy_dates):
            px = spy_by_date[spy_dates[idx + n]]["close"]
            rets[label] = (px / entry_price - 1.0) if px else None
        else:
            rets[label] = None
    return rets


def mcap_at(series: list[tuple[date, float]], d: date) -> float | None:
    """Most recent market cap on or before date d."""
    val = None
    for dd, mc in series:
        if dd <= d:
            val = mc
        else:
            break
    return val


# --- main loop ---

def run(months: int, max_tickers: int | None):
    client = FMPClient()
    end = date.today()
    start = end - timedelta(days=int(months * 30.44))
    # priors need history well before the window:
    prior_lookback_start = start - timedelta(days=365 * 4)

    print(f"Enumerating transcripts {start} -> {end} ...")
    window = client.list_transcripts_in_window(start, end)
    by_symbol: dict[str, list] = {}
    for w in window:
        by_symbol.setdefault(w["symbol"], []).append(w)
    symbols = sorted(by_symbol)
    if max_tickers:
        symbols = symbols[:max_tickers]
    print(f"{len(window)} window transcripts across {len(by_symbol)} symbols; "
          f"processing {len(symbols)}.")

    scorer = scr.FinBERTScorer()
    cache = load_score_cache()
    candidates: list[Candidate] = []
    skips = {"mcap": 0, "priors": 0, "no_bars": 0, "no_composite": 0}

    # Fetch SPY ONCE for the whole window and index by date, so each trade's matched-window
    # market return is priced from cache rather than a per-trade API call. Buffer covers
    # entries near the window edges and the 5-trading-day exit horizon past the end.
    spy_win_start = start - timedelta(days=5)
    spy_win_end = end + timedelta(days=PRICE_WINDOW_DAYS)
    print(f"Fetching SPY daily bars {spy_win_start} -> {spy_win_end} ...")
    spy_by_date, spy_dates = fetch_spy_index(client, spy_win_start, spy_win_end)
    print(f"  SPY index: {len(spy_by_date)} trading days cached.")

    for n, symbol in enumerate(tqdm(symbols, desc="tickers", unit="sym"), 1):
        try:
            all_dates = client.transcript_dates(symbol)            # full history
            mc_series = client.historical_market_cap(symbol, prior_lookback_start, end)
        except Exception as e:
            tqdm.write(f"  > metadata error {symbol}: {e}")
            continue

        window_keys = {(w["year"], w["quarter"]): w for w in by_symbol[symbol]}

        # PRE-SCORING mcap gate. A symbol can only produce a trade if it clears
        # MIN_MARKET_CAP at one of its in-window transcript dates (the same test
        # applied per-transcript below at the `mc < MIN_MARKET_CAP` skip). If it
        # never clears the floor, every candidate would be skipped anyway — so cut
        # the whole symbol here, BEFORE FinBERT-scoring its full history or fetching
        # any bars. Previously the cap was only checked after scoring, so sub-cap
        # US names and foreign listings (e.g. *.HK/*.TO/*.L, which have no US mcap
        # series) were fully scored and then discarded. Output is unchanged: these
        # symbols contributed zero candidates either way.
        if all((mcap_at(mc_series, w["date"]) or 0.0) < MIN_MARKET_CAP
               for w in by_symbol[symbol]):
            skips["mcap"] += len(by_symbol[symbol])
            continue

        history: list[dict] = []   # chronological scored {ticker,date,composite}

        for rec in tqdm(all_dates, desc=symbol, leave=False, unit="qtr"):  # oldest -> newest
            yr, q, tdt = rec["year"], rec["quarter"], rec["dt"]
            comp = None
            try:
                comp = composite_for(client, scorer, cache, symbol, yr, q)
            except Exception as e:
                tqdm.write(f"  > score error {symbol} {yr}Q{q}: {e}")
            if comp is None:
                continue
            ddate = (tdt.date() if tdt else date(yr, 1, 1))
            history.append({"ticker": symbol, "date": ddate.isoformat(), "composite": comp})

            # only generate a trade for transcripts inside the window
            if (yr, q) not in window_keys:
                continue
            w = window_keys[(yr, q)]
            priors = [h["composite"] for h in history[:-1]][-MAX_PRIOR_TRANSCRIPTS:]
            if len(priors) < MIN_PRIOR_TRANSCRIPTS:
                skips["priors"] += 1
                continue
            mc = mcap_at(mc_series, w["date"])
            if mc is None or mc < MIN_MARKET_CAP:
                skips["mcap"] += 1
                continue

            lvl_z = point_in_time_z(priors, comp)
            recs_i = history[-(MAX_PRIOR_TRANSCRIPTS + 1):]   # bounded window incl. current
            drift_z, sig_blend = drift_and_signal(recs_i)

            try:
                priced = price_trade(client, symbol, w["dt"])
            except Exception as e:
                tqdm.write(f"  > price error {symbol}: {e}")
                priced = None
            if priced is None:
                skips["no_bars"] += 1
                continue
            entry_day, entry_px, rets = priced

            # matched SPY return over the same entry day/horizons (fail soft -> None cols)
            spy = spy_matched_returns(spy_by_date, spy_dates, entry_day) or {}

            candidates.append(Candidate(
                symbol=symbol, year=yr, quarter=q,
                transcript_dt=(w["dt"].isoformat() if w["dt"] else ""),
                market_cap=mc, composite=comp, n_priors=len(priors),
                level_z=lvl_z, drift_z=drift_z, signal_blend=sig_blend,
                entry_date=entry_day.isoformat(), entry_price=entry_px,
                ret_1d=rets["ret_1d"], ret_3d=rets["ret_3d"], ret_5d=rets["ret_5d"],
                spy_ret_1d=spy.get("ret_1d"), spy_ret_3d=spy.get("ret_3d"),
                spy_ret_5d=spy.get("ret_5d"),
            ))

        if n % 25 == 0:
            save_score_cache(cache)     # checkpoint
    save_score_cache(cache)

    with open(TRADES_PATH, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        wr.writeheader()
        for c in candidates:
            wr.writerow(c.row())

    print(f"\nWrote {len(candidates)} candidates -> {TRADES_PATH}")
    print(f"Skipped: {skips}")
    print("Next: python backtest_analyze.py")


def main():
    ap = argparse.ArgumentParser(description="Generate NLP earnings-sentiment backtest candidates.")
    ap.add_argument("--months", type=int, default=12, help="Trade window length (default 12).")
    ap.add_argument("--max-tickers", type=int, default=None, help="Cap symbols (pilot runs).")
    args = ap.parse_args()
    run(args.months, args.max_tickers)


if __name__ == "__main__":
    main()