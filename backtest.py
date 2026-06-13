"""
backtest.py — generate candidate-level trade data for the NLP retail-liquidity backtest.

This is the EXPENSIVE, run-once step. For every in-window earnings transcript in the
>$500M (point-in-time) universe it:
  1. scores sentiment with FinBERT (reusing sentiment_scoring.py), with on-disk caching,
  2. computes three point-in-time trigger quantities (level-z, drift-z, signal blend),
  3. determines the run-time-anchored next-session-open ENTRY (AMC fair / BMO late),
  4. prices forward GROSS returns at open+30m / +60m / +120m / same-day-close / T+1-close.

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
from datetime import date, datetime, time, timedelta
import numpy as np
from tqdm import tqdm

import preprocessing as pp          # reused: split_transcript, sentence_tokenize
import signal_constructor as sc     # reused: SentimentSignal (point-in-time drift/z)
import sentiment_scoring as scr     # reused: avoids importing torch unless we score
from fmp_client import FMPClient

# --- config ---

RUN_TIME_ET = time(22, 0)
MIN_MARKET_CAP = 500_000_000        # point-in-time universe filter
MIN_PRIOR_TRANSCRIPTS = 4           # need a stable z baseline before trading a name
MAX_PRIOR_TRANSCRIPTS = 12          # cap priors used (bounds scoring; ~3yr quarterly)
INTERVAL = "1min"
SCORE_CACHE_PATH = "backtest_scores.json"   # {f"{SYM}:{year}:{q}": composite}
TRADES_PATH = "backtest_trades.csv"

# exit horizons measured from the 09:30 ET open (minutes), plus session-relative exits
HORIZON_MINUTES = {"open+30m": 30, "open+60m": 60, "open+120m": 120}
RTH_OPEN, RTH_CLOSE = time(9, 30), time(16, 0)

TRADE_FIELDS = [
    "symbol", "year", "quarter", "transcript_dt", "report_timing",
    "market_cap", "composite", "n_priors",
    "level_z", "drift_z", "signal_blend",
    "entry_date", "entry_price",
    "ret_open+30m", "ret_open+60m", "ret_open+120m",
    "ret_close", "ret_t1_close",
]


@dataclass
class Candidate:
    symbol: str
    year: int
    quarter: int
    transcript_dt: str
    report_timing: str
    market_cap: float
    composite: float
    n_priors: int
    level_z: float
    drift_z: float
    signal_blend: float
    entry_date: str
    entry_price: float
    ret_30: float
    ret_60: float
    ret_120: float
    ret_close: float
    ret_t1_close: float

    def row(self) -> dict:
        return {
            "symbol": self.symbol, "year": self.year, "quarter": self.quarter,
            "transcript_dt": self.transcript_dt, "report_timing": self.report_timing,
            "market_cap": self.market_cap, "composite": round(self.composite, 6),
            "n_priors": self.n_priors,
            "level_z": _r(self.level_z), "drift_z": _r(self.drift_z),
            "signal_blend": _r(self.signal_blend),
            "entry_date": self.entry_date, "entry_price": _r(self.entry_price, 4),
            "ret_open+30m": _r(self.ret_30, 6), "ret_open+60m": _r(self.ret_60, 6),
            "ret_open+120m": _r(self.ret_120, 6),
            "ret_close": _r(self.ret_close, 6), "ret_t1_close": _r(self.ret_t1_close, 6),
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

def classify_report_timing(timing_label) -> str:
    """Return the normalized BMO/AMC/during label from the earnings calendar.

    We no longer infer timing from the transcript datetime: FMP transcript records are
    date-only (every dt parses to midnight), which previously collapsed the whole universe
    into BMO and left the AMC bucket empty. The label now comes from client.earnings_timing().
    A missing label becomes 'unknown' — a real third bucket — NEVER a silent BMO default.
    """
    return timing_label if timing_label in ("BMO", "AMC", "during") else "unknown"


def decision_datetime(transcript_dt: datetime) -> datetime:
    """The nightly run that first catches this transcript (22:00 ET same day, else next day).
    NOTE: no longer used by price_trade — entry is now timing-aware (see price_trade).
    Kept for reference / the live nightly-batch path."""
    d = transcript_dt.date()
    if transcript_dt.time() >= RUN_TIME_ET:
        d = d + timedelta(days=1)
    return datetime.combine(d, RUN_TIME_ET)


def _rth(bars: list[dict], d: date) -> list[dict]:
    return [b for b in bars if b["dt"].date() == d
            and RTH_OPEN <= b["dt"].time() <= RTH_CLOSE]


def _bar_at_or_after(day_bars: list[dict], target: time) -> dict | None:
    for b in day_bars:
        if b["dt"].time() >= target:
            return b
    return None


def price_trade(client, symbol, transcript_dt, timing="unknown", interval=INTERVAL):
    """
    Returns (entry_date, entry_price, gross_returns_dict) or None.
    Entry session depends on report timing:
      BMO                      -> that SAME morning's 09:30 open (report landed pre-open),
      AMC / during / unknown   -> the NEXT session's 09:30 open (conservative).
    This fixes the prior bug where every report entered next-open: BMO names were a full
    session late. Gross returns are LONG-signed; direction/cost applied later.
    """
    report_date = transcript_dt.date()
    # window must reach back to the report date itself so BMO can enter the same session
    win_start = report_date - timedelta(days=1)
    win_end = report_date + timedelta(days=9)
    bars = client.intraday_bars(symbol, interval, win_start, win_end)
    if not bars:
        return None

    trading_days = sorted({b["dt"].date() for b in bars})
    if timing == "BMO":
        # report is pre-open -> first session on or after the report date
        entry_day = next((d for d in trading_days if d >= report_date), None)
    else:
        # AMC / during / unknown -> first session strictly after the report date
        entry_day = next((d for d in trading_days if d > report_date), None)
    if entry_day is None:
        return None

    eday = _rth(bars, entry_day)
    if not eday:
        return None
    entry_bar = _bar_at_or_after(eday, RTH_OPEN)
    if not entry_bar or not entry_bar["open"]:
        return None
    entry_price = entry_bar["open"]
    entry_open_dt = datetime.combine(entry_day, RTH_OPEN)

    rets = {}
    for label, mins in HORIZON_MINUTES.items():
        target = (entry_open_dt + timedelta(minutes=mins)).time()
        b = _bar_at_or_after(eday, target)
        px = (b["close"] if b else eday[-1]["close"])   # fall back to close if past EOD
        rets[label] = (px / entry_price - 1.0) if px else None

    rets["ret_close"] = (eday[-1]["close"] / entry_price - 1.0) if eday[-1]["close"] else None

    # T+1 close: last RTH bar of the next trading day with data
    later = sorted({b["dt"].date() for b in bars if b["dt"].date() > entry_day})
    t1 = next((d for d in later if _rth(bars, d)), None)
    if t1:
        t1bars = _rth(bars, t1)
        rets["ret_t1_close"] = (t1bars[-1]["close"] / entry_price - 1.0) if t1bars[-1]["close"] else None
    else:
        rets["ret_t1_close"] = None

    return entry_day, entry_price, rets


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

    for n, symbol in enumerate(tqdm(symbols, desc="tickers", unit="sym"), 1):
        try:
            all_dates = client.transcript_dates(symbol)            # full history
            mc_series = client.historical_market_cap(symbol, prior_lookback_start, end)
        except Exception as e:
            tqdm.write(f"  > metadata error {symbol}: {e}")
            continue

        # BMO/AMC label per earnings date (date-only join). Non-fatal: a calendar miss
        # leaves timing 'unknown' rather than dropping the symbol.
        try:
            timing_map = client.earnings_timing(symbol)
        except Exception as e:
            tqdm.write(f"  > timing error {symbol}: {e}")
            timing_map = {}

        window_keys = {(w["year"], w["quarter"]): w for w in by_symbol[symbol]}
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

            # resolve report timing from the calendar (join on the transcript date)
            label = classify_report_timing(timing_map.get(w["date"]))

            try:
                priced = price_trade(client, symbol, w["dt"], timing=label)
            except Exception as e:
                tqdm.write(f"  > price error {symbol}: {e}")
                priced = None
            if priced is None:
                skips["no_bars"] += 1
                continue
            entry_day, entry_px, rets = priced

            candidates.append(Candidate(
                symbol=symbol, year=yr, quarter=q,
                transcript_dt=(w["dt"].isoformat() if w["dt"] else ""),
                report_timing=label,
                market_cap=mc, composite=comp, n_priors=len(priors),
                level_z=lvl_z, drift_z=drift_z, signal_blend=sig_blend,
                entry_date=entry_day.isoformat(), entry_price=entry_px,
                ret_30=rets["open+30m"], ret_60=rets["open+60m"], ret_120=rets["open+120m"],
                ret_close=rets["ret_close"], ret_t1_close=rets["ret_t1_close"],
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
    ap = argparse.ArgumentParser(description="Generate NLP retail-liquidity backtest candidates.")
    ap.add_argument("--months", type=int, default=12, help="Trade window length (default 12).")
    ap.add_argument("--max-tickers", type=int, default=None, help="Cap symbols (pilot runs).")
    args = ap.parse_args()
    run(args.months, args.max_tickers)


if __name__ == "__main__":
    main()