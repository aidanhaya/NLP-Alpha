import pandas as pd
import numpy as np

class SentimentSignal:
    def __init__(self):
        # maps each ticker to a list of scored transcript records (chronological)
        self.history: dict[str, list] = {}

    def _residual_drift(self, records: list) -> float:
        scores = [r["composite"] for r in records]
        x = list(range(len(scores)))
        slope, intercept = np.polyfit(x, scores, 1)
        predicted = slope * (len(scores) - 1) + intercept
        return scores[-1] - predicted

    def _calculate_drift(self, records: list) -> float:
        n = len(records)

        scores = [r["composite"] for r in records]
        components = []

        # Simple diff - always available if n >= 2
        if n >= 2:
            raw = scores[-1] - scores[-2]
            hist = [scores[i] - scores[i - 1] for i in range(1, n - 1)]
            components.append((raw, hist, 1.0)) # tuple: (value, history, base weight)

        # Residual drift - available if n >= 8
        if n >= 8:
            raw = self._residual_drift(records) # raw residual
            # lower bound at 8 because _residual_drift needs at least 8 data points
            hist = [self._residual_drift(records[:i]) for i in range(8, n)]
            components.append((raw, hist, 3.0)) # weighted most heavily

        if not components:
            return 0.0

        # Z-score each, then blend by weight
        z_scores = []
        weights = []
        for raw, hist, weight in components:
            if len(hist) < 2:
                # z-score set to 0.0 if not enough historical data
                z_scores.append(0.0)
            else:
                mean = sum(hist) / len(hist)
                std = np.std(hist)
                z_scores.append((raw - mean) / std if std > 0 else 0.0)
            weights.append(weight)

        total_weight = sum(weights)
        # returns weighted z-score for most recent transcript
        return sum(z * w for z, w in zip(z_scores, weights)) / total_weight

    def add_score(self, scored: dict):
        # scored is a dictionary with data on a single scored transcript
        ticker = scored["ticker"]
        if ticker not in self.history:
            self.history[ticker] = []
        self.history[ticker].append(scored)
        # Keep sorted by date
        self.history[ticker].sort(key=lambda x: x["date"])

    def get_signal(self, ticker: str) -> dict:
        if ticker not in self.history or not self.history[ticker]:
            return {"signal": 0.0, "reason": "no data"}

        records = self.history[ticker]
        current = records[-1]["composite"]

        drift = self._calculate_drift(records)

        # Composite signal: level + drift (drift weighted more)
        # note that if drift = 0.0, signal will just be 0.35 * current,
        # so it will be low positive at best and most likely won't be a top choice
        signal = 0.35 * current + 0.65 * drift

        return {
            "ticker": ticker,
            "current_score": current,
            "prior_score": records[-2]["composite"] if len(records) >= 2 else None,
            "drift": drift,
            "signal": signal,
        }

    def rank_universe(self, tickers: list) -> pd.DataFrame:
        rows = [self.get_signal(t) for t in tickers]
        # sorts tickers from highest to lowest signal, best at top
        df = pd.DataFrame(rows).sort_values("signal", ascending=False)
        # ranks tickers from 1 to N
        df["rank"] = range(1, len(df) + 1)
        # assigns percentiles from near 0.0 (worst) to 1.0 (best)
        df["percentile"] = 1 - (df["rank"] - 1) / len(df)
        return df

    def get_investable_universe(self, tickers: list, top_pct: float = 0.2) -> list:
        ranked = self.rank_universe(tickers)
        # computes signal threshold value
        cutoff = ranked["signal"].quantile(1 - top_pct)
        # filters to keep only tickers that meet or exceed the cutoff
        # returns tickers as plain python list
        return ranked[ranked["signal"] >= cutoff]["ticker"].tolist()