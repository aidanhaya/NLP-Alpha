"""
ElasticNet-based, out-of-sample evaluation of the NLP earnings-call
signal, with a three-arm baseline comparison.

Independent ARMS over the SAME trades and SAME walk-forward splits, so they are directly
comparable:
  * finbert       — FinBERT composite + its level_z / drift_z            (3 features)
  * lexical       — numerical-density level_z / drift_z                  (2 features)
  * subjectivity  — the 18 SubjECTive-QA dimension fields               (18 features)
  * combined      — everything

Two readouts:
  1. Rank-IC (Spearman between predicted and realized excess) over all test trades each
     month, averaged across folds. Intercept-invariant, so it isolates the features
     from any drift the intercept absorbed.
  2. Monthly dollar-neutral L/S net-of-cost: mean across cohort months, plain and Newey-West
     t-stats, 95% CI, interpretable max drawdown, annualized Sharpe.
"""

import argparse
import warnings
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
# walks an alpha path per CV fold per l1_ratio
from sklearn.linear_model import ElasticNetCV
# z-scores features
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.exceptions import ConvergenceWarning
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# near-degenerate folds emit ConvergenceWarning even at high max_iter
# they flood the terminal, so silence that category
warnings.filterwarnings("ignore", category=ConvergenceWarning)

TRADES_PATH = "backtest_trades.csv"
SUMMARY_PATH = "backtest_summary.csv"
FIGURE_PATH = "backtest_figure.png"

COST_BPS = 40.0
COST_FRAC = COST_BPS / 1e4

# --- data hygiene ---
FOREIGN_SUFFIX_RE = r"\.[A-Z]{2,}$"   # e.g. SAN.MC, BARC.L — local-currency caps
MIN_ENTRY_PRICE = 1.0                 # sub-$1 names make % returns explosive
WINSOR_Q = 0.01                       # clip the regression target at 1st/99th pct per fold

# --- factor neutralization ---
#   log_mcap  — log(market_cap), the SIZE factor (clean).
#   log_price — log(entry_price), a weak junk/price-level proxy (optional).
NEUTRALIZE_CHOICES = {
    "none": [],
    "size": ["log_mcap"], # strips out the effect of size (market cap)
    "size_price": ["log_mcap", "log_price"], # strips out the effect of size and price
}

# --- feature arms (mirrors backtest.py schema) ---
SUBJ_DIMS = ["assertive", "cautious", "optimistic", "specific", "clear", "relevant"]
SUBJ_DIM_METRICS = ["level_z", "drift_z", "frac_low_z"]
SUBJ_DIM_FIELDS = [f"{m}_{dl}" for dl in SUBJ_DIMS for m in SUBJ_DIM_METRICS] # 18
NUMDEN_FIELDS = ["level_z_numerical_density", "drift_z_numerical_density"] # 2
FINBERT_FIELDS = ["composite", "level_z", "drift_z"]

ARMS = {
    "finbert": FINBERT_FIELDS,
    "lexical": NUMDEN_FIELDS,
    "subjectivity": SUBJ_DIM_FIELDS,
}
INCLUDE_COMBINED = True
if INCLUDE_COMBINED:
    ARMS = {**ARMS, "combined": FINBERT_FIELDS + NUMDEN_FIELDS + SUBJ_DIM_FIELDS}

# union of every arm's columns; the analyzable subset requires
# all present so the arms are compared on identical rows
FEATURE_UNION = sorted({c for cols in ARMS.values() for c in cols})

# --- horizons ---
HORIZON_DAYS = [30, 90, 180]
HEADLINE_HORIZON = 90

# --- walk-forward ---
INITIAL_TRAIN_MONTHS = 12
EMBARGO_DAYS = 2
MIN_TRAIN_N = 100
MIN_XS_N = 10 # min test trades in a month to compute L/S spread
MIN_IC_N = 8 # min test trades in a month to compute rank-IC
MIN_OOS_MONTHS = 6

# L1 hyperparameters (optimized by ElasticNetCV)
ENET_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
ENET_CV_SPLITS = 5
ENET_MAX_ITER = 20000 # extremely generous

TRADING_DAYS_PER_MONTH = 21 # trading days / month


# --- factor neutralization ---

def _factor_cols(frame: pd.DataFrame, factors: list[str]) -> np.ndarray | None:
    """Build the cross-sectional factor-exposure matrix for one cohort. Columns are the
    raw exposures (OLS handles scale via the intercept); returns None if no factors."""
    cols = []
    if "log_mcap" in factors:
        mc = pd.to_numeric(frame["market_cap"], errors="coerce").to_numpy(dtype=float)
        cols.append(np.log(np.clip(mc, 1.0, None)))
    if "log_price" in factors:
        px = pd.to_numeric(frame["entry_price"], errors="coerce").to_numpy(dtype=float)
        cols.append(np.log(np.clip(px, 1e-6, None)))
    if not cols:
        return None
    return np.column_stack(cols)


def neutralize_target(work: pd.DataFrame, y_col: str, factors: list[str]) -> pd.DataFrame:
    """Cross-sectionally residualize the target within each entry-month cohort: regress the
    (winsorized) excess return on [1, factor exposures] using that month's cross-section only,
    and replace y with the residual."""
    if not factors:
        return work
    out = work[y_col].to_numpy(dtype=float).copy()
    for _m, idx in work.groupby("_month").indices.items():
        rows = work.iloc[idx]
        y = rows[y_col].to_numpy(dtype=float)
        X = _factor_cols(rows, factors)
        if X is None:
            continue
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y) # drops NaN
        if mask.sum() < X.shape[1] + 2: # too thin -> intercept-only
            if mask.any():
                out[idx] = y - float(np.nanmean(y[mask]))
            continue
        X1 = np.column_stack([np.ones(int(mask.sum())), X[mask]])
        # calculate coefficients of OLS solution
        beta, *_ = np.linalg.lstsq(X1, y[mask], rcond=None)
        Xall1 = np.column_stack([np.ones(len(y)), np.nan_to_num(X)])
        out[idx] = y - Xall1 @ beta
    work[y_col] = out
    return work # same df, but now y_col is the factor-neutral residual (not raw excess)


# --- inference helpers ---

def _newey_west_se(x: np.ndarray, lag: int) -> float | None:
    """Bartlett-kernel (Newey-West) standard error of the MEAN of a serially-correlated
    series. lag = horizon-in-months absorbs the cohort-to-cohort overlap autocorrelation
    that a plain sd/sqrt(n) ignores. Asymptotic — rough at the small month counts here, so
    we report it ALONGSIDE the plain t, not instead of it. Returns None if degenerate."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return None
    e = x - x.mean()
    lrv = float(np.dot(e, e) / n)                 # gamma_0
    for k in range(1, min(lag, n - 1) + 1):
        cov = float(np.dot(e[k:], e[:-k]) / n)    # gamma_k
        lrv += 2.0 * (1.0 - k / (lag + 1)) * cov  # Bartlett weight
    if lrv <= 0:
        return None
    return float(np.sqrt(lrv / n))


def monthly_stats(nets: list[float], dates: list, horizon: int) -> dict:
    """Economic stats over cohort-month observations."""

    x = np.asarray(nets, dtype=float)
    n = int(len(x))
    if n == 0:
        return {"n": 0}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    t_plain = float(mean / (sd / np.sqrt(n))) if sd > 0 else 0.0
    if n > 1 and sd > 0:
        lo, hi = stats.t.interval(0.95, n - 1, loc=mean, scale=sd / np.sqrt(n))
        ci_low, ci_high = float(lo), float(hi)
    else:
        ci_low, ci_high = None, None

    lag = max(1, round(horizon / TRADING_DAYS_PER_MONTH))
    # NW is asymptotic; with few cohort months a lag near n is unreliable. Cap at n//4 so
    # the long-run variance estimate stays sane, and flag when the cap bit.
    lag_capped = min(lag, max(1, n // 4))
    nw_se = _newey_west_se(x, lag_capped)
    nw_t = float(mean / nw_se) if nw_se else None

    # date-ordered, additive equity & max drawdown
    # interpretable because each monthly number is a unit-gross L/S return
    order = np.argsort(np.asarray(dates))
    xo = x[order]
    eq = np.cumsum(xo)
    peak = np.maximum.accumulate(eq)
    mdd = float((peak - eq).max()) if eq.size else 0.0

    wins, losses = x[x > 0], x[x < 0]
    pf = (wins.sum() / -losses.sum()) if losses.size and losses.sum() != 0 else np.inf
    return {
        "n": n, "mean": mean, "median": float(np.median(x)), "std": sd,
        "t_stat": t_plain, "nw_t": nw_t, "nw_lag": lag_capped, "nw_capped": lag_capped < lag,
        "ci_low": ci_low, "ci_high": ci_high,
        "hit_rate": float((x > 0).mean()),
        "profit_factor": float(pf),
        # annualize monthly sharpe => sqrt(12)
        "ann_sharpe": float(mean / sd * np.sqrt(12)) if sd > 0 else 0.0,
        "max_drawdown": mdd,
    }


# --- per-fold model ---

def fit_fold(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], y_col: str):
    """Fit StandardScaler + ElasticNetCV on training data, then make test predictions.
    Returns (pred_array, coef_array_in_standardized_space)."""

    tr = train.sort_values("entry_date") # TimeSeriesSplit needs temporal order
    Xtr = tr[feats].to_numpy(dtype=float)
    ytr = tr[y_col].to_numpy(dtype=float)
    Xte = test[feats].to_numpy(dtype=float)

    scaler = StandardScaler().fit(Xtr) # computes mean, std for each column of feature matrix
    # builds the model with optimal L1-ratio (tries the whole grid),
    # optimal regularization strengths (lambdas, called alpha here)
    model = ElasticNetCV(
        l1_ratio=ENET_L1_RATIOS,
        cv=TimeSeriesSplit(n_splits=ENET_CV_SPLITS),
        max_iter=ENET_MAX_ITER,
    )
    # fits on standardized training features
    model.fit(scaler.transform(Xtr), ytr)
    # applies the fitted scaler to test features, then makes prediction
    pred = model.predict(scaler.transform(Xte))
    return pred, model.coef_.copy()


# --- expanding walk-forward, all arms in one pass ---

def walk_forward_multi(df: pd.DataFrame, horizon: int, long_only: bool = False,
                       neutralize_factors: list[str] | None = None) -> dict:
    """
    Anchored (expanding) walk-forward at one horizon. Every test month, every arm is fit
    on the same purged/embargoed train slice and scored on the same test slice, so the arms
    are directly comparable. Returns, per arm: the monthly dollar-neutral L/S net series +
    its stats, the mean coefficients, and a per-month rank-IC table across arms.
    """

    neutralize_factors = neutralize_factors or []
    ret_col, spy_col = f"ret_{horizon}d", f"spy_ret_{horizon}d"
    y_col = "_y_excess"
    purge_days = horizon

    work = df.dropna(subset=[ret_col, spy_col, "entry_date"]).copy()
    work["_month"] = work["entry_date"].dt.to_period("M") # buckets each trade into its calendar month
    uniq_months = sorted(work["_month"].unique())

    # matched excess return , winsorized so a few extreme survivors can't drag
    # the ElasticNet coefficients (Spearman/rank-IC is unaffected, but the fit is)
    work[y_col] = work[ret_col] - work[spy_col]
    lo, hi = work[y_col].quantile(WINSOR_Q), work[y_col].quantile(1 - WINSOR_Q)
    work[y_col] = work[y_col].clip(lo, hi)

    # optional cross-sectional factor neutralization (strips a stable factor TILT the
    # dollar-neutral L/S would otherwise still ride; see module docstring).
    work = neutralize_target(work, y_col, neutralize_factors)

    arms = {name: {"monthly_net": [], "monthly_dates": [], "coefs": [],
                   "months_fit": 0, "months_traded": 0, "n_pos": 0}
            for name in ARMS}
    ic_rows = []
    skipped = 0

    for t, m_t in enumerate(uniq_months):
        if t < INITIAL_TRAIN_MONTHS:
            continue
        test = work[work["_month"] == m_t] # test set is that month's trades
        test_month_start = pd.Timestamp(m_t.start_time)
        embargo_start = test_month_start - pd.tseries.offsets.BDay(EMBARGO_DAYS)

        train = work[work["_month"] < m_t]
        train = train[train["entry_date"] < embargo_start]
        if purge_days > 0: # 30, 90, 180
            purge_start = embargo_start - pd.tseries.offsets.BDay(purge_days)
            train = train[train["entry_date"] < purge_start]

        if len(train) < MIN_TRAIN_N or len(test) == 0:
            skipped += 1
            continue

        y_test = test[y_col].to_numpy(dtype=float)
        ic_entry = {"month": str(m_t), "n_test": int(len(test))}

        for name, feats in ARMS.items():
            pred, coef = fit_fold(train, test, feats, y_col)
            arms[name]["coefs"].append(coef)
            arms[name]["months_fit"] += 1

            # (1) paired, threshold-free signal quality over ALL test trades. A fold whose
            # model zeroed every coefficient predicts a constant; its rank is undefined, so
            # it's nan and drops out of that arm's IC average.
            ic = np.nan
            # np.ptp(pred) := peak to peak diff of pred (max - min)
            # Spearman correlation functionally requires nonzero range
            # ElasticNet zeroes every coefficient => np.ptp(pred) == 0,
            # so leave ic as np.nan and it drops out later
            if len(test) >= MIN_IC_N and np.ptp(pred) > 0 and np.ptp(y_test) > 0:
                rho = spearmanr(pred, y_test)[0] # correlation coefficient
                ic = float(rho) if np.isfinite(rho) else np.nan
            ic_entry[f"ic_{name}"] = ic

            # drift-neutral cross-sectional long/short over this month's test trades
            # demean predictions -> weights sum to zero -> common drift cancels
            # normalize gross exposure to 1 => monthly results are comparable
            month_net = 0.0
            n_pos = 0
            if len(test) >= MIN_XS_N and np.ptp(pred) > 0:
                w = pred - pred.mean() # demean predictions
                if long_only: # clips weights to be non-negative
                    # not drift-neutral, introduces market drift
                    w = np.clip(w, 0.0, None)
                gross = np.abs(w).sum()
                if gross > 0:
                    w = w / gross # normalizes w so total exposure = 1 each month
                    month_gross = float(np.dot(w, y_test)) # signal-weighted excess return
                    month_net = month_gross - COST_FRAC
                    n_pos = int((w != 0).sum()) # count of non-zero position weights
                    arms[name]["months_traded"] += 1
            arms[name]["monthly_net"].append(month_net)
            arms[name]["monthly_dates"].append(test_month_start)
            arms[name]["n_pos"] += n_pos

        ic_rows.append(ic_entry)

    # finalize per arm
    results = {}
    for name, feats in ARMS.items():
        a = arms[name]
        st = monthly_stats(a["monthly_net"], a["monthly_dates"], horizon)
        mean_coef = (np.mean(np.vstack(a["coefs"]), axis=0) if a["coefs"]
                     else np.zeros(len(feats)))
        results[name] = {
            "features": feats,
            "monthly_net": np.asarray(a["monthly_net"], dtype=float),
            "monthly_dates": list(a["monthly_dates"]),
            "stats": st,
            "mean_coef": dict(zip(feats, mean_coef)),
            "months_fit": a["months_fit"], "months_traded": a["months_traded"],
            "n_pos": a["n_pos"],
        }

    ic_df = pd.DataFrame(ic_rows)
    return {"horizon": horizon, "arms": results, "ic": ic_df,
            "skipped_months": skipped, "n_oos_months": len(ic_rows),
            "neutralize": neutralize_factors}


# --- rank-IC summaries ---

def ic_summary(ic_df: pd.DataFrame, name: str) -> dict:
    """Mean monthly rank-IC, sd, t-stat across months."""

    col = f"ic_{name}"
    if ic_df.empty or col not in ic_df.columns:
        return {"n": 0}
    vals = ic_df[col].dropna().to_numpy()
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    t = float(mean / (sd / np.sqrt(n))) if sd > 0 else 0.0
    return {"n": n, "mean": mean, "std": sd, "t": t}


def paired_ic_diff(ic_df: pd.DataFrame, a: str = "subjectivity", b: str = "finbert") -> dict | None:
    """Paired monthly rank-IC difference (subjectivity minus FinBERT) and its t-stat."""

    ca, cb = f"ic_{a}", f"ic_{b}"
    if ic_df.empty or ca not in ic_df.columns or cb not in ic_df.columns:
        return None
    d = (ic_df[ca] - ic_df[cb]).dropna().to_numpy()
    n = len(d)
    if n < 2:
        return None
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    t = float(mean / (sd / np.sqrt(n))) if sd > 0 else 0.0
    return {"n": n, "mean": mean, "t": t}


# --- verdict ---

def _bps(x):
    return x * 1e4


def _ci_text(st: dict) -> str:
    lo, hi = st.get("ci_low"), st.get("ci_high")
    if lo is None:
        return "n/a (n<2 or zero variance)"
    return f"[{_bps(lo):+.1f}, {_bps(hi):+.1f}] bps"


def _wf_call(st: dict):
    """Verdict over cohort months. 'Signal' requires the monthly mean positive
    after cost and its 95% CI to exclude zero."""

    n = st.get("n", 0)
    if n == 0:
        print("    ✗ NO OUT-OF-SAMPLE EVIDENCE. No cohort month produced a position.")
        return
    mean, ci_lo = st["mean"], st.get("ci_low")
    if n < MIN_OOS_MONTHS:
        if mean > 0:
            print(f"    ~ PROMISING BUT THIN. Monthly mean positive but only n={n} cohort "
                  f"months (< {MIN_OOS_MONTHS}) — too few to confirm.")
        else:
            print(f"    ✗ NO EDGE (THIN). Monthly mean non-positive and n={n} (< {MIN_OOS_MONTHS}).")
        return
    if mean > 0 and ci_lo is not None and ci_lo > 0:
        print("    ✓ SIGNAL. Monthly L/S mean positive after cost AND its 95% CI excludes zero.")
    elif mean > 0:
        print("    ~ INCONCLUSIVE. Monthly L/S mean positive but 95% CI spans zero.")
    else:
        print("    ✗ NO EDGE after 40bps. Monthly L/S mean flat/negative out-of-sample.")


def verdict_wf(label: str, arm_result: dict, ic_stats: dict, dump_monthly: bool = False):
    """Per-arm readout: monthly dollar-neutral L/S economics, rank-IC , and
    top standardized coefficients."""

    st = arm_result["stats"]
    print("-" * 72)
    print(f"  ARM: {label}  ({len(arm_result['features'])} features, "
          f"fit in {arm_result['months_fit']} folds, traded {arm_result['months_traded']} "
          f"cohort months)")
    n = st.get("n", 0)
    if n == 0:
        print("  Monthly L/S: no cohort took a position.")
    else:
        nw = st.get("nw_t")
        nw_txt = f"{nw:+.2f}" if nw is not None else "n/a"
        cap = "*" if st.get("nw_capped") else "" # lag capped vs n -> NW unreliable
        print(f"  Monthly L/S net (dollar-neutral): n={n}mo  "
              f"mean={_bps(st['mean']):+.1f}bps  t={st['t_stat']:+.2f}  "
              f"NWt={nw_txt}(lag={st['nw_lag']}{cap})  hit={st['hit_rate']*100:.0f}%")
        print(f"    95% CI on monthly mean: {_ci_text(st)}   "
              f"annSharpe={st['ann_sharpe']:+.2f}  maxDD={_bps(st['max_drawdown']):.0f}bps")
        traded = arm_result.get("months_traded", 0)
        if n >= MIN_OOS_MONTHS and traded >= n - 1 and st["hit_rate"] in (0.0, 1.0):
            print(f" >> WARNING: {st['hit_rate']*100:.0f}% hit over {traded} overlapping cohorts — likely "
                  "a regime/factor tilt, not alpha. Neutralize more factors before trusting it.")
        if st.get("nw_capped"):
            print(f" >> WARNING: NW lag capped to {st['nw_lag']} (< horizon-in-months) because "
                  f"n={n} is small — treat NWt as rough.")
    if ic_stats.get("n", 0):
        print(f"  Rank-IC (paired, all test trades): mean={ic_stats['mean']:+.4f}  "
              f"t={ic_stats['t']:+.2f}  over {ic_stats['n']} months")
    # list of (name, coef) tuples in influence-descending order
    coefs = sorted(arm_result["mean_coef"].items(), key=lambda kv: -abs(kv[1]))
    # clips coefficients smaller than 1e-6 (essentially zero)
    top = ", ".join(f"{k}={v:+.3f}" for k, v in coefs[:5] if abs(v) > 1e-6)
    print(f"  Mean coefs (standardized, top |.|): {top or 'all ~0 (model selected nothing)'}")
    if dump_monthly and n:
        pairs = sorted(zip(arm_result["monthly_dates"], arm_result["monthly_net"]))
        cells = "  ".join(f"{pd.Timestamp(d).strftime('%Y-%m')}:{_bps(v):+.0f}" for d, v in pairs)
        print(f"  per-cohort net (bps): {cells}")
    _wf_call(st)


def compare_arms(wf: dict, dump_monthly: bool = False):
    horizon = wf["horizon"]
    headline = horizon == HEADLINE_HORIZON
    neut = wf.get("neutralize") or []
    neut_txt = ("+".join(neut) if neut else "none")
    print("\n" + "=" * 72)
    print(f"  HORIZON {horizon}d {'(HEADLINE)' if headline else '(robustness)'} — "
          f"{wf['n_oos_months']} OOS months, {wf['skipped_months']} skipped on train-size guard")
    print(f"  target: excess−SPY, winsorized, factor-neutralized: {neut_txt}")
    print("=" * 72)

    ic_df = wf["ic"]
    for name in ARMS:
        verdict_wf(name, wf["arms"][name], ic_summary(ic_df, name), dump_monthly=dump_monthly)

    print("-" * 72)
    diff = paired_ic_diff(ic_df, "subjectivity", "finbert")
    print("  Subjectivity vs FinBERT (paired monthly rank-IC):")
    if diff is None:
        print("    too few comparable months to test.")
    else:
        print(f"    mean delta IC = {diff['mean']:+.4f}  t={diff['t']:+.2f}  over {diff['n']} months")
        if diff["mean"] > 0 and diff["t"] >= 2:
            print("    ✓ Six dimensions beat FinBERT-alone OOS (delta IC > 0, t ≥ 2).")
        elif diff["mean"] > 0:
            print("    ~ Six dimensions edge FinBERT, but delta IC is within noise (t < 2). Not confirmed.")
        else:
            print("    ✗ Six dimensions do NOT beat FinBERT-alone OOS. ")
    print("=" * 72)


# --- figure ---

def make_figure(wf: dict, path=FIGURE_PATH):
    sns.set_theme(style="darkgrid", palette="muted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    neut = wf.get("neutralize") or []
    neut_txt = ("+".join(neut) if neut else "none")
    fig.suptitle(f"NLP earnings-sentiment — OOS arm comparison @ {wf['horizon']}d "
                 f"(dollar-neutral monthly L/S, net of {COST_BPS:.0f}bps, "
                 f"factor-neutral: {neut_txt})",
                 fontsize=13, fontweight="bold")

    arms = wf["arms"]
    ic_df = wf["ic"]
    colors = dict(zip(ARMS, sns.color_palette("muted", len(ARMS))))

    # (0,0) overlaid cohort-month equity curves (additive, unit-gross => interpretable %)
    ax = axes[0, 0]
    drew = False
    for name in ARMS:
        net = arms[name]["monthly_net"]
        dts = arms[name]["monthly_dates"]
        if len(net):
            order = np.argsort(np.asarray(dts))
            eq = np.cumsum(net[order]) * 100
            ax.plot(range(len(eq)), eq, marker="o", ms=3, label=f"{name} ({len(eq)}mo)",
                    color=colors[name])
            drew = True
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    if drew:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "no positions", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Cumulative monthly L/S return by arm")
    ax.set_xlabel("cohort month (date-ordered)"); ax.set_ylabel("cumulative net return (%)")

    # (0,1) per-arm monthly mean net ± 95% CI (the economic head-to-head)
    ax = axes[0, 1]
    names, means, lo_err, hi_err, bar_colors = [], [], [], [], []
    for name in ARMS:
        st = arms[name]["stats"]
        if st.get("n", 0) and st.get("ci_low") is not None:
            names.append(name)
            means.append(_bps(st["mean"]))
            lo_err.append(_bps(st["mean"] - st["ci_low"]))
            hi_err.append(_bps(st["ci_high"] - st["mean"]))
            bar_colors.append(colors[name])
    if names:
        ax.bar(names, means, yerr=[lo_err, hi_err], capsize=5, color=bar_colors)
    else:
        ax.text(0.5, 0.5, "no CI-eligible arms", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("Monthly L/S mean net ± 95% CI")
    ax.set_ylabel("net monthly mean (bps)")

    # (1,0) subjectivity arm: mean standardized ElasticNet coefficients
    ax = axes[1, 0]
    mc = arms["subjectivity"]["mean_coef"]
    items = sorted(mc.items(), key=lambda kv: abs(kv[1]))   # ascending so biggest is on top
    labels = [k.replace("_z_", " ").replace("_", " ") for k, _ in items]
    vals = [v for _, v in items]
    ax.barh(labels, vals, color="mediumpurple")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("Subjectivity arm: mean ElasticNet coef (standardized)")
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("mean coefficient across folds")

    # (1,1) per-arm mean rank-IC ± standard error (threshold-free signal comparison)
    ax = axes[1, 1]
    names, ics, ses, bar_colors = [], [], [], []
    for name in ARMS:
        s = ic_summary(ic_df, name)
        if s.get("n", 0):
            names.append(name)
            ics.append(s["mean"])
            ses.append(s["std"] / np.sqrt(s["n"]) if s["n"] > 1 else 0.0)
            bar_colors.append(colors[name])
    if names:
        ax.bar(names, ics, yerr=ses, capsize=5, color=bar_colors)
    else:
        ax.text(0.5, 0.5, "no IC-eligible months", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_title("Mean monthly rank-IC ± SE")
    ax.set_ylabel("Spearman IC (pred vs realized excess)")

    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    print(f"Figure → {path}")


# --- summary CSV ---

def write_summary(all_wf: list[dict], path=SUMMARY_PATH):
    rows = []
    for wf in all_wf:
        ic_df = wf["ic"]
        for name in ARMS:
            ar = wf["arms"][name]
            st = ar["stats"]
            s = ic_summary(ic_df, name)
            has_n = st.get("n", 0) > 0
            rows.append({
                "horizon_days": wf["horizon"],
                "arm": name,
                "neutralize": ("+".join(wf.get("neutralize") or []) or "none"),
                "oos_months": st.get("n", 0),
                "months_traded": ar["months_traded"],
                "mean_net_bps": _bps(st["mean"]) if has_n else None,
                "ci_low_bps": _bps(st["ci_low"]) if st.get("ci_low") is not None else None,
                "ci_high_bps": _bps(st["ci_high"]) if st.get("ci_high") is not None else None,
                "t_stat": st.get("t_stat") if has_n else None,
                "nw_t": st.get("nw_t") if has_n else None,
                "ann_sharpe": st.get("ann_sharpe") if has_n else None,
                "hit_rate": st.get("hit_rate") if has_n else None,
                "max_drawdown_bps": _bps(st["max_drawdown"]) if has_n else None,
                "mean_rank_ic": s.get("mean"),
                "ic_t": s.get("t"),
                "ic_months": s.get("n", 0),
            })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Summary → {path}")


# --- driver ---

def main():
    ap = argparse.ArgumentParser(description="Model-based OOS arm comparison for the NLP backtest.")
    ap.add_argument("trades", nargs="?", default=TRADES_PATH, help="Path to backtest_trades.csv")
    ap.add_argument("--long-only", action="store_true",
                    help="Long-only L/S (keep longs, drop shorts) — mirrors the live book. "
                         "NOTE: not drift-neutral, so it reintroduces market exposure.")
    ap.add_argument("--neutralize", default="size", choices=list(NEUTRALIZE_CHOICES),
                    help="Cross-sectionally residualize the target against factor exposures "
                         "before fitting/evaluating. 'size'=log mcap (default), "
                         "'size_price'=+log entry price, 'none'=raw excess. Run with 'none' "
                         "to see the un-neutralized baseline for comparison.")
    ap.add_argument("--dump-monthly", action="store_true",
                    help="Print each arm's per-cohort monthly net (bps) — use to eyeball a "
                         "suspicious 100% hit rate.")
    args = ap.parse_args()
    neutralize_factors = NEUTRALIZE_CHOICES[args.neutralize]

    try:
        df = pd.read_csv(args.trades, parse_dates=["entry_date"])
    except FileNotFoundError:
        print(f"No {args.trades}. Run backtest.py first."); return
    if df.empty:
        print("No candidates to analyze."); return

    missing = [c for c in FEATURE_UNION if c not in df.columns]
    if missing:
        print(f"FATAL: {args.trades} is missing expected feature columns: {missing}")
        print("Re-run backtest.py with the current schema (SUBJ_DIM_FIELDS + NUMDEN_FIELDS).")
        return

    # --- data cleaning ---
    before = len(df)
    df = df[~df["symbol"].str.contains(FOREIGN_SUFFIX_RE, regex=True, na=False)]
    print(f"Dropped {before - len(df)} foreign-suffix rows ({df['symbol'].nunique()} symbols remain)")

    before = len(df)
    df = df[df["entry_price"] >= MIN_ENTRY_PRICE]
    print(f"Dropped {before - len(df)} sub-${MIN_ENTRY_PRICE:.0f} entry rows")

    n_total = len(df)
    # restrict to rows with complete features across ALL arms => identical rows per arm,
    # which is what makes the paired rank-IC comparison legitimate
    df = df.dropna(subset=FEATURE_UNION + ["entry_date"]).copy()
    if df.empty:
        print("No rows with complete features across all arms — nothing comparable to analyze.")
        return

    n_months = df["entry_date"].dt.to_period("M").nunique()
    print(f"{n_total} candidates | {len(df)} with complete features across all arms "
          f"({n_total - len(df)} dropped)")
    print(f"entry {df['entry_date'].min().date()} → {df['entry_date'].max().date()} | "
          f"{n_months} months ({max(0, n_months - INITIAL_TRAIN_MONTHS)} OOS after "
          f"{INITIAL_TRAIN_MONTHS}mo train) | headline horizon {HEADLINE_HORIZON}d"
          f"{' | LONG-ONLY' if args.long_only else ''} | "
          f"neutralize={args.neutralize}")

    all_wf = []
    for h in HORIZON_DAYS:
        wf = walk_forward_multi(df, h, long_only=args.long_only,
                                neutralize_factors=neutralize_factors)
        all_wf.append(wf)
        compare_arms(wf, dump_monthly=args.dump_monthly)

    write_summary(all_wf)
    # --> [0] MARKER <--
    headline = next(wf for wf in all_wf if wf["horizon"] == HEADLINE_HORIZON)
    make_figure(headline)


if __name__ == "__main__":
    main()