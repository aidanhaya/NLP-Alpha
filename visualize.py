import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

PERFORMANCE_LOG_PATH = "performance_log.csv"

def load_and_enrich(path: str) -> pd.DataFrame:
    # read csv and convert dates to datetime objects
    df = pd.read_csv(path, parse_dates=["date"])
    # chronologically sorts rows
    df = df.sort_values("date").reset_index(drop=True)

    base_portfolio = df["portfolio_value"].iloc[0] # captures first (base) portfolio value
    base_benchmark = df["benchmark_price"].iloc[0] # captures first (base) SPY price
    df["portfolio_norm"] = df["portfolio_value"] / base_portfolio * 100 # normalizes so day 1 = 100
    df["benchmark_norm"] = df["benchmark_price"] / base_benchmark * 100 # normalizes so day 1 = 100

    df["daily_return"] = df["portfolio_value"].pct_change()
    df["cumulative_return_pct"] = (df["portfolio_value"] / base_portfolio - 1) * 100

    rolling_mean = df["daily_return"].rolling(20).mean()
    rolling_std = df["daily_return"].rolling(20).std()
    # calculates sharpe ratio (higher => better risk-return ratio)
    # 252 = # of trading days / year
    df["rolling_sharpe"] = (rolling_mean / rolling_std) * np.sqrt(252)

    # drawdown_pct := % difference between current portfolio and max portfolio value
    rolling_max = df["portfolio_value"].cummax()
    df["drawdown_pct"] = (df["portfolio_value"] - rolling_max) / rolling_max * 100

    return df

def print_summary(df: pd.DataFrame) -> None:
    total_return = df["cumulative_return_pct"].iloc[-1]
    spy_return = (df["benchmark_price"].iloc[-1] / df["benchmark_price"].iloc[0] - 1) * 100
    max_dd = df["drawdown_pct"].min()
    sharpe = df["rolling_sharpe"].iloc[-1]
    days = len(df)

    print(f"\n{'='*40}")
    print(f"  Days tracked:        {days}")
    print(f"  Strategy return:     {total_return:+.2f}%")
    print(f"  SPY return:          {spy_return:+.2f}%")
    print(f"  Alpha:               {total_return - spy_return:+.2f}%")
    print(f"  Max drawdown:        {max_dd:.2f}%")
    print(f"  Rolling Sharpe(20d): {sharpe:.2f}" if not np.isnan(sharpe) else "  Rolling Sharpe(20d): N/A (<20 days)")
    print(f"{'='*40}\n")

def plot(df: pd.DataFrame, output_path: str = "performance_chart.png") -> None:
    sns.set_theme(style="darkgrid", palette="muted")
    # creates 4 stacked subplots sharing the date x-axis
    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
    fig.suptitle("NLP-Markowitz Portfolio Performance", fontsize=15, fontweight="bold", y=0.98)

    # 1. Normalized value vs SPY
    axes[0].plot(df["date"], df["portfolio_norm"], label="Strategy", linewidth=1.8)
    axes[0].plot(df["date"], df["benchmark_norm"], label="SPY", linewidth=1.5, linestyle="--", alpha=0.8)
    axes[0].set_ylabel("Value (Day 1 = 100)")
    axes[0].legend(loc="upper left")

    # 2. Cumulative return
    axes[1].plot(df["date"], df["cumulative_return_pct"], color="steelblue", linewidth=1.8)
    # dashed line at 0 marks break-even point
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("Cumulative Return (%)")

    # 3. Rolling 20-day Sharpe
    axes[2].plot(df["date"], df["rolling_sharpe"], color="mediumpurple", linewidth=1.8)
    axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    # dotted green line at sharpe = 1 (acceptable risk-adjusted performance)
    axes[2].axhline(1, color="green", linewidth=0.8, linestyle=":", alpha=0.8, label="Sharpe = 1")
    axes[2].set_ylabel("Rolling Sharpe (20d)")
    axes[2].legend(loc="upper left", fontsize=9)

    # 4. Drawdown
    # fills area between drawdown curve and 0 to make losses obvious
    axes[3].fill_between(df["date"], df["drawdown_pct"], 0, alpha=0.35, color="tomato")
    axes[3].plot(df["date"], df["drawdown_pct"], color="tomato", linewidth=1.5)
    axes[3].set_ylabel("Drawdown (%)")
    axes[3].set_xlabel("Date")

    # formats dates as "Month Date" (eg "Jan 15")
    date_fmt = mdates.DateFormatter("%b %d")
    for ax in axes:
        ax.xaxis.set_major_formatter(date_fmt)
        # plots dates at each week on x-axis
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {output_path}")
    plt.show()

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else PERFORMANCE_LOG_PATH
    try:
        df = load_and_enrich(path)
    except FileNotFoundError:
        print(f"No performance log found at '{path}'. Run main.py --rebalance first.")
        return

    if len(df) < 2:
        print("Not enough data to visualize — need at least 2 days of runs.")
        return

    print_summary(df)
    plot(df)

if __name__ == "__main__":
    main()