"""
backtest_analyze.py — sweep thresholds/z-definitions/directions/horizons over the
candidates in backtest_trades.csv and report whether any configuration shows positive
expectancy AFTER a flat 40bps round-trip cost, validated out-of-sample.

Cheap and re-runnable (no scoring / no API). Reads backtest_trades.csv, writes
backtest_summary.csv + backtest_figure.png, prints a GO / MARGINAL / KILL verdict.

    python backtest_analyze.py
"""

import sys
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

TRADES_PATH = "backtest_trades.csv"
SUMMARY_PATH = "backtest_summary.csv"
FIGURE_PATH = "backtest_figure.png"

COST_BPS = 40.0                 # flat round-trip, subtracted once per trade
COST_FRAC = COST_BPS / 1e4
Z_DEFS = ["level_z", "drift_z", "signal_blend"]
MODES = ["fade", "momentum"]    # fade = primary hypothesis; momentum = flipped sign
THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
HORIZONS = ["ret_open+30m", "ret_open+60m", "ret_open+120m", "ret_close", "ret_t1_close"]
TIMINGS = ["ALL", "AMC", "BMO"]
TRAIN_FRAC = 8 / 12             # first 8 of 12 months = train, last 4 = test
MIN_N = 30                      # in-sample cells below this are flagged untrustworthy
MIN_N_TEST = 15                 # test slice is ~1/3 of train; confirm at a lower bar


# ----------------------------- trade selection --------------------------- #
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
        "max_drawdown": mdd,
    }


def sweep(df: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    rows = []
    for z_def, mode, thr, horizon, timing in product(Z_DEFS, MODES, THRESHOLDS, HORIZONS, TIMINGS):
        sub = df if timing == "ALL" else df[df["report_timing"] == timing]
        if sub.empty:
            continue
        net = signed_net(sub, z_def, mode, thr, horizon)
        st = cell_stats(net, dates)
        if st["n"] == 0:
            continue
        rows.append({"z_def": z_def, "mode": mode, "threshold": thr,
                     "horizon": horizon, "timing": timing, **st})
    return pd.DataFrame(rows)


# ----------------------------- figure ------------------------------------ #
def make_figure(df_all, dates_all, df_train, df_test, best, path=FIGURE_PATH):
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NLP Retail-Liquidity Backtest — net of 40bps round-trip",
                 fontsize=14, fontweight="bold")

    # (0,0) expectancy vs threshold, by z-def (fade, AMC, open+60m)
    ax = axes[0, 0]
    for z_def in Z_DEFS:
        ys = []
        for thr in THRESHOLDS:
            sub = df_all[df_all["report_timing"] == "AMC"]
            net = signed_net(sub, z_def, "fade", thr, "ret_open+60m")
            ys.append(net.mean() * 1e4 if len(net) else np.nan)
        ax.plot(THRESHOLDS, ys, marker="o", label=z_def)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("AMC · fade · open+60m: net mean (bps) vs z-threshold")
    ax.set_xlabel("|z| threshold"); ax.set_ylabel("net mean return (bps)")
    ax.legend(fontsize=8)

    # (0,1) AMC vs BMO for the best config (validates the 'late' premise)
    ax = axes[0, 1]
    bars = []
    for timing in ["AMC", "BMO"]:
        sub = df_all[df_all["report_timing"] == timing]
        net = signed_net(sub, best["z_def"], best["mode"], best["threshold"], best["horizon"])
        bars.append(net.mean() * 1e4 if len(net) else 0.0)
    ax.bar(["AMC", "BMO"], bars, color=["steelblue", "tomato"])
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title(f"Best config by timing\n({best['z_def']}·{best['mode']}·"
                 f"thr{best['threshold']}·{best['horizon']})")
    ax.set_ylabel("net mean return (bps)")

    # (1,0) net return distribution for best config (all)
    ax = axes[1, 0]
    net = signed_net(df_all, best["z_def"], best["mode"], best["threshold"], best["horizon"])
    if len(net):
        ax.hist(np.clip(net.values * 100, -15, 15), bins=40,
                color="mediumpurple", alpha=0.8, edgecolor="white")
        ax.axvline(net.mean() * 100, color="black", ls="--",
                   label=f"mean {net.mean()*100:.2f}%")
        ax.legend(fontsize=8)
    ax.set_title("Best config: net return distribution")
    ax.set_xlabel("net return (%)"); ax.set_ylabel("count")

    # (1,1) equity curve: train vs test for best config
    ax = axes[1, 1]
    for label, dd, color in [("train", df_train, "steelblue"), ("test (OOS)", df_test, "darkorange")]:
        net = signed_net(dd, best["z_def"], best["mode"], best["threshold"], best["horizon"])
        if not len(net):
            continue
        order = dates_all.loc[net.index].sort_values().index
        eq = np.cumsum(net.loc[order].values) * 100
        ax.plot(range(len(eq)), eq, label=f"{label} (n={len(eq)})", color=color)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("Cumulative net return — best config")
    ax.set_xlabel("trade # (date-ordered)"); ax.set_ylabel("cumulative net return (%)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    print(f"Figure → {path}")


# ----------------------------- verdict ----------------------------------- #
def pick_best(train_summary: pd.DataFrame) -> dict | None:
    cand = train_summary[(train_summary["n"] >= MIN_N) & (train_summary["timing"] != "BMO")]
    if cand.empty:
        return None
    return cand.sort_values("mean", ascending=False).iloc[0].to_dict()


def verdict(best, train_stats, test_stats):
    print("\n" + "=" * 64)
    if best is None:
        print("  VERDICT: insufficient trades in any cell (N < %d)." % MIN_N)
        print("=" * 64); return
    cfg = (f"{best['z_def']} · {best['mode']} · |z|>={best['threshold']} · "
           f"{best['horizon']} · {best['timing']}")
    print(f"  Best in-sample config: {cfg}")
    print(f"    TRAIN: n={train_stats['n']}  mean={train_stats['mean']*1e4:+.1f}bps  "
          f"t={train_stats['t_stat']:+.2f}  hit={train_stats['hit_rate']*100:.0f}%  "
          f"breakeven={train_stats['breakeven_bps']:.0f}bps")
    if test_stats and test_stats["n"] > 0:
        print(f"    TEST : n={test_stats['n']}  mean={test_stats['mean']*1e4:+.1f}bps  "
              f"t={test_stats['t_stat']:+.2f}  hit={test_stats['hit_rate']*100:.0f}%  "
              f"breakeven={test_stats['breakeven_bps']:.0f}bps")
    print("-" * 64)
    tr_pos = train_stats["mean"] > 0
    te_n = test_stats["n"] if test_stats else 0
    te_confirms = te_n >= MIN_N_TEST and test_stats["mean"] > 0 and test_stats["breakeven_bps"] > COST_BPS
    te_contradicts = te_n >= MIN_N_TEST and test_stats["mean"] <= 0
    if tr_pos and te_confirms:
        print("  ✓ SIGNAL. Positive expectancy survives out-of-sample after cost.")
        print("    Promote this config to paper trading; keep instrumenting.")
    elif tr_pos and te_contradicts:
        print("  ✗ LIKELY OVERFIT. Looks good in-sample, fails to confirm out-of-sample.")
        print("    Do NOT trade it. Treat the in-sample result as noise.")
    elif tr_pos and te_n < MIN_N_TEST:
        print(f"  ~ PROMISING BUT THIN. In-sample positive; test slice only n={te_n} "
              f"(< {MIN_N_TEST}) — too few to confirm.")
        print("    Extend the window or loosen the threshold before trusting it.")
    else:
        print("  ✗ NO EDGE after 40bps. Every parameterization is flat/negative.")
        print("    The most valuable result you can get this month — and it cost nothing.")
        print("    Rethink the data source / hypothesis before the streaming rewrite.")
    print("=" * 64)


def stats_for(df, dates, best):
    net = signed_net(df if best["timing"] == "ALL" else df[df["report_timing"] == best["timing"]],
                     best["z_def"], best["mode"], best["threshold"], best["horizon"])
    return cell_stats(net, dates)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else TRADES_PATH
    try:
        df = pd.read_csv(path, parse_dates=["entry_date"])
    except FileNotFoundError:
        print(f"No {path}. Run backtest.py first."); return
    if df.empty:
        print("No candidates to analyze."); return

    dates = df["entry_date"]
    cutoff = dates.min() + (dates.max() - dates.min()) * TRAIN_FRAC
    df_train, df_test = df[dates <= cutoff], df[dates > cutoff]
    print(f"{len(df)} candidates | train {len(df_train)} (≤{cutoff.date()}) | "
          f"test {len(df_test)} (>{cutoff.date()})")
    print(f"Timing mix: {df['report_timing'].value_counts().to_dict()}")

    full_summary = sweep(df, dates)
    full_summary.sort_values("mean", ascending=False).to_csv(SUMMARY_PATH, index=False)
    print(f"Summary ({len(full_summary)} cells) → {SUMMARY_PATH}")

    train_summary = sweep(df_train, dates)
    best = pick_best(train_summary)

    if best is not None:
        train_stats = stats_for(df_train, dates, best)
        test_stats = stats_for(df_test, dates, best)
        make_figure(df, dates, df_train, df_test, best)
        verdict(best, train_stats, test_stats)
    else:
        make_figure(df, dates, df_train, df_test,
                    {"z_def": "level_z", "mode": "fade", "threshold": 2.0,
                     "horizon": "ret_open+60m", "timing": "AMC"})
        verdict(None, None, None)


if __name__ == "__main__":
    main()
