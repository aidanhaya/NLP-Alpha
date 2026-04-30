import argparse
import numpy as np
from datetime import date

import persistence
import signal_constructor as sc
from ibkr_manager import IBKRPortfolioManager

def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Project vector v onto the probability simplex (non-negative, sums to 1).
    Uses Lagrangian multipliers to minimize the distance between the probability
    simplex and vector v."""

    n = len(v)
    u = np.sort(v)[::-1] # sort v in descending order
    # cumulative sum of sorted values - used to compute top-k elements efficiently
    cssv = np.cumsum(u)

    # rho := max index of elements that survive (rho + 1 elements survive)
    # elements that would push cumsum > 1 get clipped using a boolean mask
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    # theta is a uniform shift subtracted from each element of v
    # elements of v > theta survive and are reduced, elements < theta get clipped to 0
    # finds theta s.t. sum of surviving weights = cssv[rho] - (rho + 1) * theta = 1
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    # subtracts theta from every element of v, clips negative values to 0

    return np.maximum(v - theta, 0)

# returns := pandas df, each column is a ticker, each row is a daily return
def _min_variance_weights(returns) -> dict:
    """
    Global minimum-variance portfolio via projected gradient descent.
    Long-only, fully invested (weights >= 0, sum to 1).
    """

    tickers = list(returns.columns)
    n = len(tickers)

    # pandas method to compute covariance matrix of all columns against each other
    # n x n matrix where entry (i,j) is the covariance between asset i and asset j
    # diagonal is each asset's own variance
    cov = returns.cov().values

    # initialized uniform array of weights
    w = np.ones(n) / n
    # learning rate = 1e-3, relatively small/safe, works well over 5000 iterations
    lr = 1e-3
    for _ in range(5000):
        # need to minimize portfolio variance := w^T @ cov @ w
        # where w := weights vector, w^T := transpose(w), cov := covariance matrix
        # derivative of variance wrt weight = d/dw (w^T @ cov @ w) = 2 * cov @ w
        grad = 2.0 * cov @ w
        # move slightly in the direction opposite the gradient: w - lr * grad
        # project updated weight vector onto simplex: _project_simplex( ... )
        w = _project_simplex(w - lr * grad)

    # returns dict of {ticker: weight}
    return {t: float(w[i]) for i, t in enumerate(tickers)}

def rebalance(
    portfolio_value: float,
    today_tickers: list,
    top_pct: float = 0.2,
    holding_days: int = 63,
    stop_loss_pct: float = 0.15,
    take_profit_pct: float = 0.25,
    dry_run: bool = False,
):
    all_records = persistence.load_scores()
    if not all_records:
        print("No scored transcripts found. Run main.py first.")
        return

    signal = sc.SentimentSignal()
    for record in all_records:
        signal.add_score(record)

    positions = persistence.load_positions()
    manager = IBKRPortfolioManager()
    today = date.today()

    if portfolio_value is None:
        portfolio_value = manager.get_net_liquidation()
        print(f"Portfolio value (from IBKR): ${portfolio_value:,.2f}")

    # fetch current prices for all held tickers (needed for stop-loss / take-profit checks)
    current_prices = manager.get_prices(list(positions)) if positions else {}

    # top 20% of today's newly scored tickers
    today_investable = set(signal.get_investable_universe(today_tickers, top_pct=top_pct))

    # --- identify exits ---
    exits = set()
    for ticker, meta in positions.items():
        entry_date = date.fromisoformat(meta["entry_date"])
        # counts business days since entry
        trading_days_held = int(np.busday_count(entry_date, today))
        price_now = current_prices.get(ticker)
        entry_price = meta["entry_price"]

        if price_now and price_now < entry_price * (1 - stop_loss_pct):
            print(f"STOP-LOSS: {ticker} (entry {entry_price:.2f}, now {price_now:.2f})")
            exits.add(ticker)
            continue

        if price_now and price_now > entry_price * (1 + take_profit_pct):
            print(f"TAKE-PROFIT: {ticker} (entry {entry_price:.2f}, now {price_now:.2f})")
            exits.add(ticker)
            continue

        if trading_days_held >= holding_days:
            print(f"TIME-LIMIT: {ticker} ({trading_days_held} trading days held)")
            exits.add(ticker)
            continue

        if ticker in today_tickers and ticker not in today_investable:
            print(f"RE-EVAL EXIT: {ticker} (new earnings signal below top {int(top_pct*100)}%)")
            exits.add(ticker)

    # --- reset clock for held stocks with a new good earnings signal ---
    for ticker in list(positions):
        if ticker in today_tickers and ticker in today_investable and ticker not in exits:
            old_date = positions[ticker]["entry_date"]
            positions[ticker]["entry_date"] = str(today)
            positions[ticker]["entry_price"] = current_prices.get(
                ticker, positions[ticker]["entry_price"]
            )
            print(f"CLOCK RESET: {ticker} (re-entered top {int(top_pct*100)}%, entry date {old_date} → {today})")

    # --- identify new entries ---
    new_entries = [t for t in today_investable if t not in positions]
    if new_entries:
        print(f"NEW ENTRIES: {new_entries}")

    # --- build active portfolio ---
    active = [t for t in positions if t not in exits] + new_entries
    if not active:
        print("No active tickers for portfolio.")
        return

    print(f"\nActive portfolio ({len(active)} tickers): {active}")

    prices_hist = manager.get_historical_prices(active)
    prices_hist = prices_hist.dropna(axis=1, how="all")
    returns = prices_hist.pct_change().dropna()

    valid = [t for t in active if t in returns.columns]
    if not valid:
        print("No valid price data returned by IBKR.")
        return
    if len(valid) < len(active):
        dropped = set(active) - set(valid)
        print(f"Dropped {len(dropped)} tickers with no price data: {sorted(dropped)}")
    returns = returns[valid]

    weights = _min_variance_weights(returns)

    print("\nTarget weights:")
    for ticker, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {ticker}: {w:.4f}")

    if dry_run:
        print("\nDry run — no orders placed.")
        return

    manager.rebalance(weights, portfolio_value)
    print("Rebalance complete.")

    # --- update positions.json ---
    for ticker in exits:
        positions.pop(ticker, None)

    # fetch entry prices for new positions using current market prices
    new_entry_prices = manager.get_prices(new_entries) if new_entries else {}
    for ticker in new_entries:
        if ticker in valid:  # only record positions we actually have price data for
            positions[ticker] = {
                "entry_date": str(today),
                "entry_price": new_entry_prices.get(ticker, 0.0),
            }

    persistence.save_positions(positions)

    spy_price = manager.get_prices(["SPY"]).get("SPY", float("nan"))
    persistence.log_performance(str(today), portfolio_value, len(weights), spy_price)
    print(f"Performance logged — portfolio: ${portfolio_value:,.2f}, SPY: {spy_price:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Rebalance portfolio using "
        "NLP-Markowitz signals.")
    parser.add_argument(
        "--portfolio-value", type=float, default=None,
        help="Total portfolio value in USD (default: auto-fetched from IBKR)",
    )
    parser.add_argument(
        "--today-tickers", nargs="+", required=True,
        help="Tickers with new transcripts today (passed automatically by main.py)",
    )
    parser.add_argument(
        "--top-pct", type=float, default=0.2,
        help="Top percentile of today's tickers to enter (default: 0.2)",
    )
    parser.add_argument(
        "--holding-days", type=int, default=63,
        help="Max trading days to hold a position (default: 63)",
    )
    parser.add_argument(
        "--stop-loss-pct", type=float, default=0.15,
        help="Exit if position falls this fraction from entry price (default: 0.15)",
    )
    parser.add_argument(
        "--take-profit-pct", type=float, default=0.25,
        help="Exit if position gains this fraction from entry price (default: 0.25)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print target weights without placing any orders",
    )
    args = parser.parse_args()

    rebalance(
        portfolio_value=args.portfolio_value,
        today_tickers=args.today_tickers,
        top_pct=args.top_pct,
        holding_days=args.holding_days,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()
