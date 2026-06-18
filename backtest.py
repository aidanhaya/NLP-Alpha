"""
backtest.py — generate candidate-level trade data for the NLP earnings-sentiment backtest.

This is the EXPENSIVE, run-once step. For every in-window earnings transcript in the
>$500M (point-in-time) universe it:
  1. scores sentiment with FinBERT (reusing sentiment_scoring.py), with on-disk caching,
  2. computes three point-in-time trigger quantities (level-z, drift-z, signal blend),
  3. determines the next trading day's open as ENTRY (report time-of-day doesn't matter —
     only the calendar day the report landed on),
  4. prices forward GROSS returns at 30 / 90 / 180 trading days using daily close prices.

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
import subjectivity_scoring as subj # reused: SubjectivityScorer + its own on-disk cache
from fmp_client import FMPClient

# --- config ---

MIN_MARKET_CAP = 500_000_000        # point-in-time universe filter
MIN_PRIOR_TRANSCRIPTS = 4           # need a stable z baseline before trading a name
MAX_PRIOR_TRANSCRIPTS = 12          # cap priors used (bounds scoring; ~3yr quarterly)
SCORE_CACHE_PATH = "backtest_scores.json"   # {f"{SYM}:{year}:{q}": composite}
TRADES_PATH = "backtest_trades.csv"

# forward horizons measured in trading days from the entry day
HORIZON_DAYS = {"ret_30d": 30, "ret_90d": 90, "ret_180d": 180}
# calendar-day buffer covering the 180-trading-day horizon: ~252 calendar days span
# 180 NYSE sessions counting weekends only, plus slack for the ~7 holidays in that
# span and run-time edge cases.
PRICE_WINDOW_DAYS = 290

# canonical lowercase order of the six SubjECTive-QA dimensions, matching
# train_subjectivity.DIMENSIONS lowercased. Hardcoded (not read off the loaded
# checkpoint) so the trade schema is fixed at import time; run() validates the loaded
# SubjectivityScorer's dimensions against this list and fails loudly on a mismatch.
SUBJ_DIMS = ["assertive", "cautious", "optimistic", "specific", "clear", "relevant"]
# per-dimension signals: drift_z is the primary trigger (tone shift vs. the firm's own
# trend), level_z is a control feature left for the ElasticNet to keep or zero out, and
# frac_low_z is the lightly z-scored evasiveness tail (most informative for clear/relevant,
# but emitted for all six so Phase 4 can decide which carry the edge).
SUBJ_DIM_METRICS = ["level_z", "drift_z", "frac_low_z"]
SUBJ_DIM_FIELDS = [f"{m}_{dl}" for dl in SUBJ_DIMS for m in SUBJ_DIM_METRICS]

# numerical_density is a 7th "pseudo-dimension" from subjectivity_scoring.py: a text-derived
# regex count (no model) stored as numerical_density_mean + frac_low_numerical_density in the
# subjectivity cache. It gets the same level_z/drift_z treatment as the six model dims, but
# NOT frac_low_z (the frac_low tail is already its own feature; z-scoring it again is
# redundant given drift_z already captures the shift).
NUMDEN_FIELDS = ["level_z_numerical_density", "drift_z_numerical_density"]

TRADE_FIELDS = [
    "symbol", "year", "quarter", "transcript_dt",
    "market_cap", "composite", "n_priors",
    "level_z", "drift_z", "signal_blend",
    *SUBJ_DIM_FIELDS,
    *NUMDEN_FIELDS,
    "entry_date", "entry_price",
    "ret_30d", "ret_90d", "ret_180d",
    # per-trade market (SPY) matched returns
    "spy_ret_30d", "spy_ret_90d", "spy_ret_180d",
]

SPY_SYMBOL = "SPY"


def _empty_dim_fields() -> dict:
    """All per-dimension z-fields set to None (subjectivity data missing/insufficient
    for this candidate). None-not-fake-zero, matching subjectivity_scoring.py's convention."""
    return {f: None for f in [*SUBJ_DIM_FIELDS, *NUMDEN_FIELDS]}


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
    dim_signals: dict          # the 18 SUBJ_DIM_FIELDS -> float|None
    entry_date: str
    entry_price: float
    ret_30d: float
    ret_90d: float
    ret_180d: float
    spy_ret_30d: float
    spy_ret_90d: float
    spy_ret_180d: float

    def row(self) -> dict:
        out = {
            "symbol": self.symbol, "year": self.year, "quarter": self.quarter,
            "transcript_dt": self.transcript_dt,
            "market_cap": self.market_cap, "composite": round(self.composite, 6),
            "n_priors": self.n_priors,
            "level_z": _r(self.level_z), "drift_z": _r(self.drift_z),
            "signal_blend": _r(self.signal_blend),
            "entry_date": self.entry_date, "entry_price": _r(self.entry_price, 4),
            "ret_30d": _r(self.ret_30d, 6), "ret_90d": _r(self.ret_90d, 6),
            "ret_180d": _r(self.ret_180d, 6),
            "spy_ret_30d": _r(self.spy_ret_30d, 6), "spy_ret_90d": _r(self.spy_ret_90d, 6),
            "spy_ret_180d": _r(self.spy_ret_180d, 6),
        }
        out.update({k: _r(v) for k, v in self.dim_signals.items()})
        return out


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

def point_in_time_z(prior_values: list[float], current: float) -> float:
    """Level z-score of `current` vs PRIOR values (priors only, no leak). Field-agnostic:
    the caller extracts whichever field's values (composite, a {dim}_mean, ...) into the list."""
    if len(prior_values) < 2:
        return 0.0
    arr = np.asarray(prior_values, dtype=float)
    mu, sd = arr.mean(), arr.std()
    return float((current - mu) / sd) if sd > 0 else 0.0


def drift_and_signal(records_through_i: list[dict],
                     field: str = "composite") -> tuple[float, float]:
    """Reuse SentimentSignal for the blended drift-z and full signal, as of transcript i.
    `field` selects the record key to operate on (defaults to FinBERT's "composite")."""
    sig = sc.SentimentSignal(field=field)
    for rec in records_through_i:
        sig.add_score(rec)
    out = sig.get_signal(records_through_i[-1]["ticker"])
    return float(out.get("drift", 0.0)), float(out.get("signal", 0.0))


def dim_signals_for(subj_history: list[dict]) -> dict:
    """level_z_{dim}/drift_z_{dim}/frac_low_z_{dim} for each of the six subjectivity
    dimensions, plus level_z/drift_z for numerical_density, as of the LAST entry in
    subj_history (the current transcript). Exactly mirrors composite's
    level_z/drift_and_signal pattern, looped over SUBJ_DIMS and over the {dim}_mean /
    frac_low_{dim} fields — same point_in_time_z and drift_and_signal calls, no new math.
    numerical_density gets level_z + drift_z only (no frac_low_z — frac_low_numerical_density
    is already its own cached feature; z-scoring it again given drift_z captures the shift
    is redundant). None-filled if this ticker doesn't yet have a full
    MIN_PRIOR_TRANSCRIPTS subjectivity baseline.

    Caller contract: only call this when subj_history's last entry IS the current
    transcript (i.e. this transcript actually produced subjectivity features) — otherwise
    "current" would silently be a stale prior quarter.
    """
    out = _empty_dim_fields()
    priors_window = subj_history[:-1][-MAX_PRIOR_TRANSCRIPTS:]
    if len(priors_window) < MIN_PRIOR_TRANSCRIPTS:
        return out
    recs_i = subj_history[-(MAX_PRIOR_TRANSCRIPTS + 1):]   # bounded window incl. current
    current = subj_history[-1]
    for dl in SUBJ_DIMS:
        mean_key, frac_key = f"{dl}_mean", f"frac_low_{dl}"
        level_vals = [h[mean_key] for h in priors_window]
        frac_vals = [h[frac_key] for h in priors_window]
        out[f"level_z_{dl}"] = point_in_time_z(level_vals, current[mean_key])
        out[f"drift_z_{dl}"], _ = drift_and_signal(recs_i, field=mean_key)
        out[f"frac_low_z_{dl}"] = point_in_time_z(frac_vals, current[frac_key])
    # numerical_density: same level_z/drift_z treatment, no frac_low_z (see docstring).
    nd_key = "numerical_density_mean"
    nd_level_vals = [h[nd_key] for h in priors_window]
    out["level_z_numerical_density"] = point_in_time_z(nd_level_vals, current[nd_key])
    out["drift_z_numerical_density"], _ = drift_and_signal(recs_i, field=nd_key)
    return out


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

    # NOTE on the window: FMP's "latest" transcript feed (list_transcripts_in_window) has a
    # SERVER-SIDE result cap — it stops returning rows after ~6 months regardless of our
    # max_pages, so it CANNOT bound a multi-year window. We therefore use it ONLY to
    # DISCOVER the active symbol universe. The trade window is derived per-symbol below from
    # each symbol's OWN full transcript_dates history filtered to [start, end], so
    # `--months 36` actually produces ~36 months of trades.
    #
    # CAVEAT (survivorship): symbols that stopped reporting before the discovery feed's
    # ~6-month horizon won't be discovered here. Fixing that needs a point-in-time universe
    # source (e.g. an FMP screener / symbol-list endpoint); this change fixes window DEPTH
    # for the currently-active universe, which is the immediate blocker.
    print(f"Discovering symbols via latest-transcript feed ...")
    discovery = client.list_transcripts_in_window(start, end)
    symbols = sorted({w["symbol"] for w in discovery})
    if max_tickers:
        symbols = symbols[:max_tickers]
    print(f"Trade window {start} -> {end} ({months}mo). "
          f"Discovered {len(symbols)} symbols from feed; processing {len(symbols)}.")

    scorer = scr.FinBERTScorer()
    cache = load_score_cache()

    sub_scorer = subj.SubjectivityScorer()
    loaded_dims = [d.lower() for d in sub_scorer.dimensions]
    if loaded_dims != SUBJ_DIMS:
        raise ValueError(
            f"subjectivity_model dimensions {loaded_dims} don't match the canonical "
            f"SUBJ_DIMS order {SUBJ_DIMS} backtest.py's trade schema is built from. "
            "Update SUBJ_DIMS if the checkpoint's dimension set/order changed."
        )
    subj_cache = subj.load_subjectivity_cache()

    candidates: list[Candidate] = []
    skips = {"mcap": 0, "priors": 0, "no_bars": 0, "no_composite": 0, "no_window": 0}

    # Fetch SPY ONCE for the whole window and index by date, so each trade's matched-window
    # market return is priced from cache rather than a per-trade API call. Buffer covers
    # entries near the window edges and the 180-trading-day exit horizon past the end.
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

        # In-window transcripts now come from the symbol's OWN history (a real datetime,
        # inside [start, end]) — NOT the truncated discovery feed. This is the whole fix.
        in_window = [r for r in all_dates
                     if r["dt"] is not None and start <= r["dt"].date() <= end]
        if not in_window:
            skips["no_window"] += 1
            continue

        # PRE-SCORING mcap gate (same intent as before, now over the symbol's OWN in-window
        # dates). If the name never clears MIN_MARKET_CAP at ANY in-window transcript date,
        # every candidate would be skipped anyway — so cut the whole symbol BEFORE
        # FinBERT-scoring its full history or fetching any bars. Foreign listings with no US
        # mcap series (mcap_at -> None -> treated as 0.0) are still cut here, unchanged.
        if all((mcap_at(mc_series, r["dt"].date()) or 0.0) < MIN_MARKET_CAP
               for r in in_window):
            skips["mcap"] += len(in_window)
            continue

        # (year, quarter) pairs that fall inside the window for this symbol
        window_qkeys = {(r["year"], r["quarter"]) for r in in_window}

        history: list[dict] = []        # chronological scored {ticker,date,composite}
        subj_history: list[dict] = []   # chronological {ticker,date,{dim}_mean,frac_low_{dim}}
                                         # (only transcripts with parseable Q&A get appended)

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

            # subjectivity features, built up every transcript (in/out of window) just like
            # `history` above, so later windows have a full prior baseline. Only appended
            # when this transcript actually segmented Q&A pairs (sub_rec/features non-None) —
            # otherwise subj_history's last entry would silently be a stale prior quarter.
            sub_rec = None
            try:
                sub_rec = subj.subjectivity_features_for(client, sub_scorer, subj_cache,
                                                          symbol, yr, q)
            except Exception as e:
                tqdm.write(f"  > subjectivity score error {symbol} {yr}Q{q}: {e}")
            have_subj_now = False
            if sub_rec is not None:
                have_subj_now = sub_rec["features"].get(f"{SUBJ_DIMS[0]}_mean") is not None
            if have_subj_now:
                subj_history.append({"ticker": symbol, "date": ddate.isoformat(),
                                     **sub_rec["features"]})

            # Only generate a trade for in-window transcripts. Membership is now the
            # transcript's OWN (year, quarter), and we require a real datetime so the entry
            # can be timed/priced. Out-of-window transcripts are still SCORED above (they
            # feed the priors baseline) — only trade-row generation is gated here.
            if (yr, q) not in window_qkeys or tdt is None:
                continue
            priors = [h["composite"] for h in history[:-1]][-MAX_PRIOR_TRANSCRIPTS:]
            if len(priors) < MIN_PRIOR_TRANSCRIPTS:
                skips["priors"] += 1
                continue
            mc = mcap_at(mc_series, ddate)
            if mc is None or mc < MIN_MARKET_CAP:
                skips["mcap"] += 1
                continue

            lvl_z = point_in_time_z(priors, comp)
            recs_i = history[-(MAX_PRIOR_TRANSCRIPTS + 1):]   # bounded window incl. current
            drift_z, sig_blend = drift_and_signal(recs_i)
            dim_fields = dim_signals_for(subj_history) if have_subj_now else _empty_dim_fields()

            try:
                priced = price_trade(client, symbol, tdt)
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
                transcript_dt=tdt.isoformat(),
                market_cap=mc, composite=comp, n_priors=len(priors),
                level_z=lvl_z, drift_z=drift_z, signal_blend=sig_blend,
                dim_signals=dim_fields,
                entry_date=entry_day.isoformat(), entry_price=entry_px,
                ret_30d=rets["ret_30d"], ret_90d=rets["ret_90d"], ret_180d=rets["ret_180d"],
                spy_ret_30d=spy.get("ret_30d"), spy_ret_90d=spy.get("ret_90d"),
                spy_ret_180d=spy.get("ret_180d"),
            ))

        if n % 25 == 0:
            save_score_cache(cache)          # checkpoint
            subj.save_subjectivity_cache(subj_cache)
    save_score_cache(cache)
    subj.save_subjectivity_cache(subj_cache)

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