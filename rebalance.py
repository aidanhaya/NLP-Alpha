import argparse
import numpy as np

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
    tickers: list = None,
    top_pct: float = 0.2,
    dry_run: bool = False,
):
    all_records = persistence.load_scores()
    if not all_records:
        print("No scored transcripts found. Run main.py first.")
        return

    signal = sc.SentimentSignal()
    for record in all_records:
        signal.add_score(record)

    if tickers is None:
        tickers = list({r["ticker"] for r in all_records})

    investable = signal.get_investable_universe(tickers, top_pct=top_pct)
    print(f"Investable universe ({len(investable)} tickers): {investable}")

    if not investable:
        print("No investable tickers found.")
        return

    manager = IBKRPortfolioManager()

    prices = manager.get_historical_prices(investable)
    prices = prices.dropna(axis=1, how="all") # drops columns where all values are NaN
    returns = prices.pct_change().dropna() # converts raw prices to daily % returns

    valid = [t for t in investable if t in returns.columns]
    if not valid:
        print("No valid price data returned by IBKR.")
        return
    if len(valid) < len(investable):
        dropped = set(investable) - set(valid)
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

def main():
    parser = argparse.ArgumentParser(description="Rebalance portfolio using "
        "NLP-Markowitz signals.")
    parser.add_argument(
        "--portfolio-value", type=float, required=True,
        help="Total portfolio value in USD",
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Explicit ticker list to consider (defaults to all scored tickers)",
    )
    parser.add_argument(
        "--top-pct", type=float, default=0.2,
        help="Top percentile of sentiment universe to invest in (default: 0.2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print target weights without placing any orders",
    )
    args = parser.parse_args()

    rebalance(
        portfolio_value=args.portfolio_value,
        tickers=args.tickers,
        top_pct=args.top_pct,
        dry_run=args.dry_run,
    )

if __name__ == "__main__":
    main()