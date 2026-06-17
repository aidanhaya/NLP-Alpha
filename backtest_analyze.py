"""
backtest_analyze.py — sweep thresholds/z-definitions/directions/horizons over the
candidates in backtest_trades.csv and report whether any configuration shows positive
expectancy AFTER a flat 40bps round-trip cost, validated OUT-OF-SAMPLE.

Cheap and re-runnable (no scoring / no API). Reads backtest_trades.csv, writes
backtest_summary.csv and backtest_figure.png.

Part-1 hardening vs. the original single train/test cut:
  * Grid shrunk to 36 cells (fewer researcher degrees of freedom).
  * Validation is an expanding (anchored) walk-forward that RE-SELECTS the config each
    step, with a 2-trading-day embargo — no single config is cherry-picked with hindsight.
  * Every reported mean carries a 95% confidence interval; with small N the interval is
    the honest story.
"""

import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
# prevents matplotlib from popping up a display upon graph generation
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TRADES_PATH = "backtest_trades.csv"
SUMMARY_PATH = "backtest_summary.csv"
FIGURE_PATH = "backtest_figure.png"

COST_BPS = 40.0                 # flat round-trip, subtracted once per trade
COST_FRAC = COST_BPS / 1e4

# --- sweep grid (shrunk to 36 cells: overfit / multiple-comparisons control) ---
# drop signal_blend (deterministic function of level_z & drift_z).
Z_DEFS = ["level_z", "drift_z"]
MODES = ["fade", "momentum"]    # fade = pos sent => short; momentum = pos sent => long
THRESHOLDS = [1.5, 2.0, 2.5]    # tested z-score cutoffs
HORIZONS = ["ret_30d", "ret_90d", "ret_180d"]   # forward horizons, in trading days
#   2 (z) x 2 (mode) x 3 (thr) x 3 (horizon) = 36 cells.

MIN_N = 30                      # in-sample cells below this are flagged untrustworthy
MIN_N_TEST = 15                 # pooled-OOS confirmation bar

# --- walk-forward validation ---
INITIAL_TRAIN_MONTHS = 12       # anchored start: first 12 months are train-only
EMBARGO_DAYS = 2                # trading days dropped from train just before each test month
# Horizons now reach 180 trading days, so a train trade's label window can extend well past
# the embargo into the test month (leakage). Purge drops train trades entered within the
# longest horizon of the embargo cutoff, sized off HORIZONS so it stays correct if those change.
PURGE_DAYS = max(int(h.rsplit("_", 1)[-1].rstrip("d")) for h in HORIZONS)


# --- trade selection ---

def signed_net(df: pd.DataFrame, z_def: str, mode: str, thr: float, horizon: str) -> pd.Series:
    """
    Return a Series of NET per-trade returns for triggered trades, signed by direction
    and net of round-trip cost. Index preserved so we can sort by date for the equity curve.
    fade:     z>=+thr -> short ;  z<=-thr -> long
    momentum: z>=+thr -> long  ;  z<=-thr -> short
    """
    z = df[z_def]
    gross = df[horizon]
    long_mask = (z <= -thr) if mode == "fade" else (z >= thr)
    short_mask = (z >= thr) if mode == "fade" else (z <= -thr)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    out[long_mask] = gross[long_mask] - COST_FRAC
    out[short_mask] = -gross[short_mask] - COST_FRAC
    return out.dropna().dropna()  # also drops rows where the horizon return was NaN


def cell_stats(net: pd.Series, dates: pd.Series) -> dict:
    n = int(net.notna().sum())
    if n == 0:
        return {"n": 0}
    x = net.dropna().values
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    wins, losses = x[x > 0], x[x < 0]
    pf = (wins.sum() / -losses.sum()) if losses.size and losses.sum() != 0 else np.inf
    tstat = (mean / (sd / np.sqrt(n))) if sd > 0 else 0.0
    # break-even round-trip cost (bps): the gross signed edge before cost
    breakeven_bps = (mean + COST_FRAC) * 1e4
    # 95% CI for the mean (return units). Honest-uncertainty headline for small N.
    if n > 1 and sd > 0:
        lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
        ci_low, ci_high = float(lo), float(hi)
    else:
        ci_low, ci_high = None, None
    # date-ordered equity & max drawdown (additive in return units)
    order = dates.loc[net.dropna().index].sort_values().index
    eq = np.cumsum(net.loc[order].values)
    peak = np.maximum.accumulate(eq)
    mdd = float((peak - eq).max()) if eq.size else 0.0
    return {
        "n": n, "mean": mean, "median": float(np.median(x)), "std": sd,
        "hit_rate": float((x > 0).mean()),
        "avg_win": float(wins.mean()) if wins.size else 0.0,
        "avg_loss": float(losses.mean()) if losses.size else 0.0,
        "profit_factor": float(pf), "sharpe_per_trade": float(mean / sd) if sd > 0 else 0.0,
        "t_stat": float(tstat), "breakeven_bps": float(breakeven_bps),
        "ci_low": ci_low, "ci_high": ci_high,
        "max_drawdown": mdd,
    }


def sweep(df: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    rows = []
    for z_def, mode, thr, horizon in product(Z_DEFS, MODES, THRESHOLDS, HORIZONS):
        net = signed_net(df, z_def, mode, thr, horizon)
        st = cell_stats(net, dates)
        if st["n"] == 0:
            continue
        rows.append({"z_def": z_def, "mode": mode, "threshold": thr,
                     "horizon": horizon, **st})
    return pd.DataFrame(rows)


def apply_config(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Signed-net Series for one config."""
    return signed_net(df, cfg["z_def"], cfg["mode"], cfg["threshold"], cfg["horizon"])


def stats_for(df: pd.DataFrame, dates: pd.Series, cfg: dict) -> dict:
    return cell_stats(apply_config(df, cfg), dates)


# --- expanding walk-forward + embargo ---

def walk_forward(df: pd.DataFrame) -> dict:
    """Anchored (expanding) walk-forward that re-selects the config each test month.

    For each test month m_t (index t >= INITIAL_TRAIN_MONTHS):
      train = all trades in earlier months MINUS those within EMBARGO_DAYS trading days
              before m_t starts; best config = pick_best(sweep(train)).
      test  = trades in m_t; its signed_net(best) is appended to a pooled OOS series.
    Pooled OOS stats are the honest headline. Returns oos series/dates/stats, the per-month
    selection log, and the modal (most-frequently-selected) config.
    """
    work = df.assign(_month=df["entry_date"].dt.to_period("M"))
    uniq_months = sorted(work["_month"].unique())

    oos_net_parts, oos_date_parts, per_month = [], [], []
    config_counter: dict[tuple, int] = {}

    for t, m_t in enumerate(uniq_months):
        if t < INITIAL_TRAIN_MONTHS:
            continue
        test = work[work["_month"] == m_t]
        test_month_start = pd.Timestamp(m_t.start_time)
        embargo_start = test_month_start - pd.tseries.offsets.BDay(EMBARGO_DAYS)

        train = work[work["_month"] < m_t]
        train = train[train["entry_date"] < embargo_start]   # apply the embargo
        # PURGE: horizons top out at 180 trading days, so also drop train trades whose label
        # window could overlap the embargo/test region (entered within PURGE_DAYS of it).
        if PURGE_DAYS > 0:
            purge_start = embargo_start - pd.tseries.offsets.BDay(PURGE_DAYS)
            train = train[train["entry_date"] < purge_start]

        best = pick_best(sweep(train, train["entry_date"]))
        if best is None:
            per_month.append({"month": str(m_t), "test_n": 0, "config": None})
            continue

        net = apply_config(test, best)
        if len(net):
            oos_net_parts.append(net)
            oos_date_parts.append(test["entry_date"].loc[net.index])
        key = (best["z_def"], best["mode"], best["threshold"], best["horizon"])
        config_counter[key] = config_counter.get(key, 0) + 1
        per_month.append({"month": str(m_t), "test_n": int(len(net)), "config": best})

    if oos_net_parts:
        oos_net = pd.concat(oos_net_parts)
        oos_dates = pd.concat(oos_date_parts)
        oos_stats = cell_stats(oos_net, oos_dates)
    else:
        oos_net = pd.Series(dtype=float)
        oos_dates = pd.Series(dtype="datetime64[ns]")
        oos_stats = {"n": 0}

    modal_config = None
    if config_counter:
        k = max(config_counter, key=config_counter.get)
        modal_config = {"z_def": k[0], "mode": k[1], "threshold": k[2], "horizon": k[3],
                        "count": config_counter[k],
                        "n_selected_months": sum(1 for p in per_month if p["config"])}

    return {"oos_net": oos_net, "oos_dates": oos_dates, "oos_stats": oos_stats,
            "per_month": per_month, "modal_config": modal_config}


def pick_best(train_summary: pd.DataFrame) -> dict | None:
    if train_summary.empty:
        return None
    cand = train_summary[train_summary["n"] >= MIN_N]
    if cand.empty:
        return None
    return cand.sort_values("mean", ascending=False).iloc[0].to_dict()


# --- figure ---

def make_figure(df, wf, headline_cfg, path=FIGURE_PATH):
    hc = headline_cfg or {"z_def": "level_z", "mode": "fade", "threshold": 2.0,
                          "horizon": "ret_30d"}
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NLP Earnings-Call Sentiment Backtest — net of 40bps round-trip",
                 fontsize=14, fontweight="bold")

    # (0,0) expectancy vs threshold, by z-def (fade, headline horizon)
    ax = axes[0, 0]
    for z_def in Z_DEFS:
        ys = []
        for thr in THRESHOLDS:
            net = signed_net(df, z_def, "fade", thr, hc["horizon"])
            ys.append(net.mean() * 1e4 if len(net) else np.nan)
        ax.plot(THRESHOLDS, ys, marker="o", label=z_def)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"fade · {hc['horizon']}: net mean (bps) vs z-threshold")
    ax.set_xlabel("|z| threshold"); ax.set_ylabel("net mean return (bps)")
    ax.legend(fontsize=8)

    # (0,1) headline config across horizons (30d/90d/180d)
    ax = axes[0, 1]
    bars = []
    for horizon in HORIZONS:
        net = signed_net(df, hc["z_def"], hc["mode"], hc["threshold"], horizon)
        bars.append(net.mean() * 1e4 if len(net) else 0.0)
    ax.bar(HORIZONS, bars, color="steelblue")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"Headline config across horizons\n({hc['z_def']}·{hc['mode']}·"
                 f"thr{hc['threshold']})")
    ax.set_ylabel("net mean return (bps)")

    # (1,0) net return distribution for the headline config (all)
    ax = axes[1, 0]
    net = signed_net(df, hc["z_def"], hc["mode"], hc["threshold"], hc["horizon"])
    if len(net):
        ax.hist(np.clip(net.values * 100, -15, 15), bins=40,
                color="mediumpurple", alpha=0.8, edgecolor="white")
        ax.axvline(net.mean() * 100, color="black", ls="--",
                   label=f"mean {net.mean()*100:.2f}%")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no triggered trades", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_title("Headline config: net return distribution")
    ax.set_xlabel("net return (%)"); ax.set_ylabel("count")

    # (1,1) pooled out-of-sample equity curve (date-ordered) for the headline config
    ax = axes[1, 1]
    oos_net, oos_dates = wf.get("oos_net"), wf.get("oos_dates")
    if oos_net is not None and len(oos_net):
        order = oos_dates.loc[oos_net.index].sort_values().index
        eq = np.cumsum(oos_net.loc[order].values) * 100
        ax.plot(range(len(eq)), eq, label=f"pooled OOS (n={len(eq)})", color="darkorange")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no pooled OOS trades\n(every step tripped the N guard)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Pooled out-of-sample equity — walk-forward")
    ax.set_xlabel("trade # (date-ordered)"); ax.set_ylabel("cumulative net return (%)")

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    print(f"Figure → {path}")


# --- verdict ---

def _bps(x):
    return x * 1e4


def _ci_text(st: dict) -> str:
    lo, hi = st.get("ci_low"), st.get("ci_high")
    if lo is None:
        return "n/a (n<2 or zero variance)"
    return f"[{_bps(lo):+.1f}, {_bps(hi):+.1f}] bps"


def _wf_call(oos: dict):
    n = oos.get("n", 0)
    if n == 0:
        print("  ✗ NO OUT-OF-SAMPLE EVIDENCE. Nothing qualified to trade walk-forward.")
        return
    mean, ci_lo = oos["mean"], oos.get("ci_low")
    if n < MIN_N_TEST:
        if mean > 0:
            print(f"  ~ PROMISING BUT THIN. Pooled OOS positive but n={n} (< {MIN_N_TEST}) "
                  "— too few to confirm.")
            print("    Extend the window or loosen the threshold before trusting it.")
        else:
            print(f"  ✗ NO EDGE (THIN). Pooled OOS non-positive and n={n} (< {MIN_N_TEST}).")
        return
    if mean > 0 and ci_lo is not None and ci_lo > 0:
        print("  ✓ SIGNAL. Pooled OOS mean positive after cost AND its 95% CI excludes zero.")
        print("    Promote this config to paper trading; keep instrumenting.")
    elif mean > 0:
        print("  ~ INCONCLUSIVE. Pooled OOS mean positive but 95% CI spans zero — "
              "not yet distinguishable from noise.")
    else:
        print("  ✗ NO EDGE after 40bps. Pooled OOS mean flat/negative out-of-sample.")
        print("    Rethink the data source / hypothesis before the streaming rewrite.")


def verdict_wf(label: str, wf: dict, full_best: dict | None, full_stats: dict | None):
    oos = wf["oos_stats"]
    print("\n" + "=" * 72)
    print(f"  UNIVERSE: {label}")
    print("=" * 72)

    n = oos.get("n", 0)
    if n == 0:
        print(f"  HEADLINE — pooled out-of-sample: no trades. Every walk-forward step "
              f"tripped the N>={MIN_N} selection guard.")
    else:
        print(f"  HEADLINE — pooled out-of-sample (expanding walk-forward, "
              f"{EMBARGO_DAYS}d embargo, {INITIAL_TRAIN_MONTHS}mo initial train):")
        print(f"    n={n}  mean={_bps(oos['mean']):+.1f}bps  t={oos['t_stat']:+.2f}  "
              f"hit={oos['hit_rate']*100:.0f}%  maxDD={_bps(oos['max_drawdown']):.0f}bps")
        print(f"    95% CI on mean: {_ci_text(oos)}   <- with small N this interval is the story")

    mc = wf["modal_config"]
    if mc:
        print(f"  Modal selected config ({mc['count']}/{mc['n_selected_months']} selected months): "
              f"{mc['z_def']} · {mc['mode']} · |z|>={mc['threshold']} · {mc['horizon']}")
    else:
        print("  Modal selected config: none (no month selected a qualifying config).")


    print("  Per-month OOS counts:")
    for pm in wf["per_month"]:
        if pm["config"]:
            c = pm["config"]
            tag = f"{c['z_def']}/{c['mode']}/|z|{c['threshold']}/{c['horizon']}"
        else:
            tag = "— (no qualifying config; N guard)"
        print(f"    {pm['month']}: n={pm['test_n']:>3}   {tag}")

    print("-" * 72)
    if full_best and full_stats and full_stats.get("n", 0) > 0:
        print("  (context only — in-sample is optimistic, NOT a trading signal)")
        print(f"    Full in-sample best: {full_best['z_def']} · {full_best['mode']} · "
              f"|z|>={full_best['threshold']} · {full_best['horizon']}")
        print(f"    n={full_stats['n']}  mean={_bps(full_stats['mean']):+.1f}bps  "
              f"t={full_stats['t_stat']:+.2f}  95% CI {_ci_text(full_stats)}")
    else:
        print("  (context only) Full in-sample best: none cleared the N guard.")

    print("-" * 72)
    _wf_call(oos)
    print("=" * 72)


# --- per-universe driver ---

def run_universe(df: pd.DataFrame, label: str, summary_path: str) -> tuple[dict, dict | None]:
    dates = df["entry_date"]
    summary = sweep(df, dates)
    if not summary.empty:
        summary.sort_values("mean", ascending=False).to_csv(summary_path, index=False)
        print(f"  [{label}] in-sample sweep: {len(summary)} populated cells → {summary_path}")
    else:
        print(f"  [{label}] in-sample sweep: no populated cells; {summary_path} not written.")
    full_best = pick_best(summary)
    full_stats = stats_for(df, dates, full_best) if full_best else None
    wf = walk_forward(df)
    verdict_wf(label, wf, full_best, full_stats)
    return wf, full_best


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else TRADES_PATH
    try:
        df = pd.read_csv(path, parse_dates=["entry_date"])
    except FileNotFoundError:
        print(f"No {path}. Run backtest.py first."); return
    if df.empty:
        print("No candidates to analyze."); return

    n_months = df["entry_date"].dt.to_period("M").nunique()
    print(f"{len(df)} candidates | entry {df['entry_date'].min().date()} → "
          f"{df['entry_date'].max().date()} | {n_months} calendar months "
          f"({max(0, n_months - INITIAL_TRAIN_MONTHS)} OOS months after {INITIAL_TRAIN_MONTHS}mo train)")

    wf, best = run_universe(df, "FULL UNIVERSE", SUMMARY_PATH)

    headline = wf["modal_config"] or best
    make_figure(df, wf, headline)


if __name__ == "__main__":
    main()