# NLP-Alpha

**Does earnings-call sentiment actually predict returns — and can you trade it after costs?**

NLP-Alpha scores the tone of earnings-call transcripts with FinBERT, turns that into a per-ticker trading signal, and then puts the signal through two very different tests: a **live pipeline** that builds and executes a portfolio through Interactive Brokers, and a **point-in-time backtest** that asks, honestly and out-of-sample, whether the signal has any edge at all.

This repository is the consolidated home of a project that has gone through three iterations. Earlier versions lived in separate repos (`NLP-Markowitz`, `NLP-Modified`); they are now preserved here as git tags so the whole arc lives in one place.

---

## Project history

| Version | Tag | Focus |
| ------- | --- | ----- |
| **v1** | [`v1.0.0`](../../releases/tag/v1.0.0) | Live FinBERT → Markowitz → IBKR paper-trading pipeline |
| **v2** | [`v2.0.0`](../../releases/tag/v2.0.0) | Point-in-time backtest harness; cost-aware, out-of-sample edge validation |
| **v3** | `master` *(in progress)* | Fixing the backtest's execution model and re-validating before trusting any verdict |

### v1 — The portfolio (`v1.0.0`)

The first version was an end-to-end trading system, optimistic and complete. It scrapes earnings-call transcripts, scores them with FinBERT, ranks tickers by a sentiment **drift** signal, selects a top-percentile investable universe, sizes positions with Markowitz minimum-variance optimization, and executes through the IBKR paper-trading API — with stop-loss, take-profit, time-limit, and signal-dropout exits. It is a working pipeline, and most of it still lives on `master` today.

What v1 *didn't* do was establish that the underlying signal was worth trading. It assumed the edge and built the machinery around it.

### v2 — The reckoning (`v2.0.0`)

v2 stepped back from building and asked the question v1 had skipped: *does this signal have edge, after costs, out-of-sample?* That meant abandoning portfolio construction for proper hypothesis testing.

The backtest harness (`backtest.py` + `backtest_analyze.py`, on Financial Modeling Prep data) is built around a clean separation: an expensive run-once step scores every in-window transcript and prices forward returns at several intraday horizons; a cheap, re-runnable step then sweeps thresholds, signal definitions, directions (fade vs. momentum), and report timing (AMC vs. BMO), applies a flat 40 bps round-trip cost, splits train/test, and prints a blunt **GO / MARGINAL / KILL** verdict. The point was never to confirm the strategy — it was to find out, cheaply, whether there was anything there before committing to a real-time rewrite.

### v3 — Trust the measurement first (`master`, in progress)

v3 began with an uncomfortable discovery: **the v2 backtest was measuring the wrong trade.** FMP caps rows per request and returns the most recent bars first, so the single 9-day price window per candidate silently dropped its early sessions — entries landed roughly five sessions late instead of at the intended next-session open. Every return column was contaminated.

Rather than re-run the expensive scoring, `reprice_trades.py` corrects only the pricing: the signal columns (composite, z-scores, priors) are price-independent and carry over untouched, while entries and forward returns are re-fetched one session at a time, safely under FMP's caps. The output (`backtest_trades_repriced.csv`) feeds straight back into the analyzer.

**Current plans, in order:**

1. **Reprice and re-validate.** Run the corrected trades through `backtest_analyze.py` and treat *that* verdict — not v2's — as the real one.
2. **Fix `report_timing`.** It is currently degenerate (everything labels as `BMO`), so genuine before-open reports are entered a session late and the AMC-fade hypothesis can't be tested yet. A timing relabel is the next prerequisite.
3. **Clean the universe.** Foreign-suffixed listings carry local-currency caps and non-ET sessions; `--us-only` drops them for now.
4. **Then, and only if an edge survives:** the streaming/intraday execution rewrite — moving away from the daily Markowitz rebalance toward something that can actually act on a short-horizon signal.

The honest possibility remains that the corrected backtest kills the signal. That would be the most valuable result the project could produce, and it would cost nothing further to learn.

---

## How the signal works

Shared across every version:

1. **Scrape** — transcripts are fetched (Playwright from Motley Fool in the live pipeline; the FMP API in the backtest).
2. **Preprocess** — each transcript is split into prepared remarks and Q&A, cleaned, and sentence-tokenized.
3. **Score** — every sentence is scored with [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert). Q&A is weighted more heavily (60/40) because it is less scripted. The per-transcript **composite** is mean positive minus mean negative probability.
4. **Signal** — a per-ticker drift signal is computed from how composite sentiment changes across a company's transcripts (a simple diff plus, once enough history exists, a residual-drift term), z-scored against the ticker's own history and blended with the current level.
5. **Rank** — tickers are ranked and the top percentile (default 20%) becomes the investable universe.

---

## The live pipeline (v1 lineage)

### Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

Rebalancing requires an IBKR paper-trading gateway running locally on `127.0.0.1:7497`.

### Usage

**Bootstrap (first run only)** — scrape and score a large batch of historical transcripts to build up signal history. Already-scored transcripts are skipped, so you can extend the history later.

```bash
python main.py --bootstrap --pages 100
```

**Daily run (scrape + score only)** — scrapes today's transcripts, backfills thin tickers, regenerates rankings. No orders placed.

```bash
python main.py
```

**Daily run with rebalancing** — full pipeline plus IBKR execution. Portfolio value is auto-fetched from IBKR.

```bash
python main.py --rebalance                       # live
python main.py --rebalance --dry-run             # print target weights, place nothing
python main.py --rebalance --portfolio-value 100000   # override auto-fetched value
```

Risk / holding parameters (optional, shown with defaults):

```bash
python main.py --rebalance \
  --holding-days 63 \      # exit after this many trading days
  --stop-loss-pct 0.15 \   # exit if a position drops 15% from entry
  --take-profit-pct 0.25   # exit if a position gains 25% from entry
```

**Standalone rebalance** — runs against already-persisted scores without re-scraping.

```bash
python rebalance.py --today-tickers AAPL MSFT NVDA
```

**Visualize performance** — 4-panel chart plus a terminal summary (needs ≥2 days of runs).

```bash
python visualize.py
```

#### Exit rules

| Condition | Trigger |
| --------- | ------- |
| Stop-loss | Price falls more than `stop-loss-pct` below entry |
| Take-profit | Price rises more than `take-profit-pct` above entry |
| Time limit | Position held for `holding-days` trading days |
| Signal dropout | Ticker no longer in today's top-percentile universe |

If a held ticker re-enters the investable universe on the same day it would have exited, its entry date and price reset (clock restart).

---

## The backtest (v2 → v3 lineage)

The backtest runs on Financial Modeling Prep data and is independent of IBKR.

```bash
export FMP_API_KEY=...

python backtest.py --months 12            # expensive: score + price all candidates
python reprice_trades.py --us-only        # v3 fix: correct the entry-lag pricing bug
python backtest_analyze.py backtest_trades_repriced.csv   # cheap: sweep + verdict
```

- `backtest.py` writes `backtest_trades.csv` — one row per priced candidate, with signal columns and forward gross returns. It does **not** apply thresholds, direction, or cost.
- `reprice_trades.py` corrects only the entry/return columns (see v3 above) and writes `backtest_trades_repriced.csv`. It checkpoints every 200 rows and resumes if interrupted.
- `backtest_analyze.py` sweeps configurations, applies the 40 bps round-trip cost, validates train/test, writes `backtest_summary.csv` + `backtest_figure.png`, and prints the verdict.

---

## Working with versions

Each prior version is a tag you can check out or download as a release:

```bash
git checkout v1.0.0    # the original Markowitz/IBKR pipeline
git checkout v2.0.0    # the first backtest harness
git checkout master    # current work
```

`master` is always the current line of development; the tags are immutable snapshots.

---

## Repository layout

| File | Role |
| ---- | ---- |
| `webscraper.py` | Playwright scraper for Motley Fool transcripts (live pipeline) |
| `preprocessing.py` | Split / clean / sentence-tokenize transcripts |
| `sentiment_scoring.py` | FinBERT scoring and per-transcript composite |
| `signal_constructor.py` | Drift signal, ranking, investable universe |
| `rebalance.py` | Exit logic + Markowitz min-variance weights |
| `ibkr_manager.py` | IBKR connection, pricing, order placement |
| `main.py` | Orchestrates the live pipeline (bootstrap / daily / rebalance) |
| `visualize.py` | Performance chart + summary |
| `persistence.py` | JSON/CSV state for scores, positions, performance |
| `fmp_client.py` | Financial Modeling Prep API wrapper (backtest) |
| `backtest.py` | Candidate generation + forward pricing |
| `reprice_trades.py` | v3 entry-lag fix; re-prices trades without re-scoring |
| `backtest_analyze.py` | Threshold/direction sweep, cost, OOS verdict |

## Generated files (gitignored)

These are produced by runs and are not tracked: `transcript_scores.json`, `positions.json`, `signals_output.csv`, `performance_log.csv`, `performance_chart.png`, and the backtest artifacts (`backtest_trades*.csv`, `backtest_summary.csv`, `backtest_scores.json`, `backtest_prices.json`, `backtest_figure.png`).
