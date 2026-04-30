# NLP-Markowitz

Earnings call transcript scraper and sentiment pipeline that constructs a portfolio using FinBERT sentiment signals and Markowitz minimum-variance optimization, with live execution via Interactive Brokers.

---

## How it works

1. **Scrape** — Playwright fetches earnings call transcripts from Motley Fool.
2. **Preprocess** — Transcripts are split into prepared remarks and Q&A sections, cleaned, and sentence-tokenized.
3. **Score** — Each sentence is scored with [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert). The Q&A section is weighted more heavily (60/40) as it is less scripted.
4. **Signal** — A drift signal is computed per ticker from the change in composite sentiment across transcripts. Tickers are ranked and a top-percentile investable universe is selected (default top 20%).
5. **Optimize** — Markowitz minimum-variance weights are computed via projected gradient descent on historical return covariance.
6. **Execute** — Market orders are placed through the IBKR paper trading API. Positions are exited on stop-loss, take-profit, time limit, or signal dropout.

Scores are persisted to `transcript_scores.json` and position metadata to `positions.json` (both gitignored) so re-runs skip already-processed work.

---

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

Rebalancing requires an IBKR paper trading gateway running locally on `127.0.0.1:7497`.

---

## Usage

### Stage 1 — Bootstrap (first run only)

Scrapes and scores a large batch of historical transcripts to build up enough history for signal construction. Each listing page covers ~20 transcripts, so `--pages 100` yields ~2,000 transcripts.

```bash
python main.py --bootstrap --pages 100
```

You can extend the history later — already-scored transcripts are skipped automatically:

```bash
python main.py --bootstrap --pages 200
```

---

### Stage 2 — Daily run (scrape + score only)

Scrapes today's transcripts, backfills history for any tickers below 8 records, then regenerates signal rankings. No orders are placed.

```bash
python main.py
```

---

### Stage 3 — Daily run with rebalancing

Runs the full pipeline and places orders through IBKR. `--portfolio-value` is the total USD value to deploy.

```bash
python main.py --rebalance --portfolio-value 100000
```

**Dry run** — prints target weights without placing any orders:

```bash
python main.py --rebalance --portfolio-value 100000 --dry-run
```

**Risk / holding parameters** (all optional, shown with defaults):

```bash
python main.py --rebalance --portfolio-value 100000 \
  --holding-days 63 \       # exit after this many trading days
  --stop-loss-pct 0.15 \    # exit if position drops 15% from entry
  --take-profit-pct 0.25    # exit if position gains 25% from entry
```

---

### Stage 4 — Standalone rebalance

Runs rebalance logic directly against already-persisted scores, without re-scraping or re-scoring. Useful for testing weight changes or re-running after a connectivity issue. `--today-tickers` is the list of tickers that had new transcripts today (used to determine signal dropout exits).

```bash
python rebalance.py \
  --portfolio-value 100000 \
  --today-tickers AAPL MSFT NVDA
```

With full options:

```bash
python rebalance.py \
  --portfolio-value 100000 \
  --today-tickers AAPL MSFT NVDA \
  --top-pct 0.2 \
  --holding-days 63 \
  --stop-loss-pct 0.15 \
  --take-profit-pct 0.25 \
  --dry-run
```

---

## Exit rules

A position is closed when any of the following conditions are met:

| Condition | Trigger |
|-----------|---------|
| Stop-loss | Price falls more than `stop-loss-pct` below entry price |
| Take-profit | Price rises more than `take-profit-pct` above entry price |
| Time limit | Position has been held for `holding-days` trading days |
| Signal dropout | Ticker no longer in the top-percentile investable universe today |

If a held ticker re-enters the investable universe on the same day it would have been exited, its entry date and price are reset (clock restart).

---

## Persisted files

| File | Contents |
|------|----------|
| `transcript_scores.json` | All scored transcripts; re-runs skip keys already present |
| `positions.json` | Current holdings with entry date and entry price |
| `signals_output.csv` | Latest signal rankings; regenerated each run |