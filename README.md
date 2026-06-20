# NLP-Alpha

**Does earnings-call sentiment actually predict returns — and can you trade it after costs?**

NLP-Alpha scores the tone of earnings-call transcripts with FinBERT and a fine-tuned subjectivity model, turns those scores into per-ticker signals, and then puts the signals through two very different tests: a **live pipeline** that builds and executes a portfolio through Interactive Brokers, and a **point-in-time backtest** that asks, honestly and out-of-sample, whether the signal has any edge at all.

This repository is the consolidated home of a project that has gone through three iterations. Earlier versions lived in separate repos (`NLP-Markowitz`, `NLP-Modified`); they are now preserved here as git tags so the whole arc lives in one place.

---

## Project history

| Version | Tag | Focus |
| ------- | --- | ----- |
| **v1** | [`v1.0.0`](../../releases/tag/v1.0.0) | Live FinBERT → Markowitz → IBKR paper-trading pipeline |
| **v2** | [`v2.0.0`](../../releases/tag/v2.0.0) | Point-in-time backtest harness; cost-aware, out-of-sample edge validation |
| **v3** | `master` *(current)* | Subjectivity features, model-based walk-forward, honest three-arm OOS comparison |

### v1 — The portfolio (`v1.0.0`)

The first version was an end-to-end trading system, optimistic and complete. It scrapes earnings-call transcripts, scores them with FinBERT, ranks tickers by a sentiment **drift** signal, selects a top-percentile investable universe, sizes positions with Markowitz minimum-variance optimization, and executes through the IBKR paper-trading API — with stop-loss, take-profit, time-limit, and signal-dropout exits. It is a working pipeline, and most of it still lives on `master` today.

What v1 *didn't* do was establish that the underlying signal was worth trading. It assumed the edge and built the machinery around it.

### v2 — The reckoning (`v2.0.0`)

v2 stepped back from building and asked the question v1 had skipped: *does this signal have edge, after costs, out-of-sample?* That meant abandoning portfolio construction for proper hypothesis testing.

The backtest harness (`backtest.py` + `backtest_analyze.py`, on Financial Modeling Prep data) is built around a clean separation: an expensive run-once step scores every in-window transcript and prices forward returns at several horizons; a cheap, re-runnable step then sweeps thresholds, signal definitions, and directions (fade vs. momentum), applies a flat 40 bps round-trip cost, splits train/test, and reports the verdict. The point was never to confirm the strategy — it was to find out, cheaply, whether there was anything there before committing to a real-time rewrite.

### v3 — Measurement before momentum (`master`)

v3 began with an uncomfortable discovery: **the v2 backtest was measuring the wrong trade.** FMP caps rows per request and returns the most recent bars first, so the single 9-day price window per candidate silently dropped its early sessions — entries landed roughly five sessions late instead of at the intended next-session open. Every return column was contaminated. That pricing bug has since been fixed directly in `backtest.py`.

v3 then expanded the signal itself, adding a second scorer alongside FinBERT: a RoBERTa encoder fine-tuned on the SubjECTive-QA dataset with six independent classification heads (Assertive, Cautious, Optimistic, Specific, Clear, Relevant). Each head measures a different dimension of *how* management communicates, orthogonal to FinBERT's polarity. Both scorers run at inference time during `backtest.py` and are cached to disk.

The analyzer was then rewritten to match: the v2 threshold/direction grid sweep was replaced by a regularized linear model (ElasticNetCV) that evaluates a feature *set* — the six subjectivity dimensions — as a unit inside the same anchored walk-forward. The key question it now answers is: **do the six subjectivity dimensions beat FinBERT-alone out-of-sample?** If not, that is a cheap, important negative result before any further investment.

v3 also dropped two pieces of scope that turned out not to matter for the question being asked: the universe is no longer filtered down to a retail-liquidity subset, and entries no longer condition on whether a report landed before or after market close — only the calendar day matters. Pricing moved from intraday bars to daily bars, with forward returns measured at 30/90/180 trading days.

The honest possibility remains that the backtest kills the signal. That would be the most valuable result the project could produce, and it would cost nothing further to learn.

---

## How the signal works

### FinBERT (sentiment)

1. **Scrape** — transcripts fetched from Motley Fool (live pipeline, Playwright) or FMP API (backtest).
2. **Preprocess** — split prepared remarks from Q&A, clean, sentence-tokenize.
3. **Score** — every sentence scored with [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert); Q&A weighted 60/40 over prepared remarks. Per-transcript **composite** = mean positive minus mean negative probability.
4. **Signal** — a per-ticker drift signal: how composite sentiment changes across a company's own transcript history (simple diff blended with an EWMA-baseline deviation once enough history exists), z-scored against the ticker's own prior distribution.

### SubjECTive-QA (subjectivity)

1. **Segment** — Q&A split into (question, answer) pairs via a speaker-turn state machine.
2. **Score** — each pair scored by a fine-tuned RoBERTa model on six dimensions as a continuous 0–2 expectation (probability-weighted, no argmax).
3. **Aggregate** — per call: `{dim}_mean` (the level) and `frac_low_{dim}` (fraction of answers in the evasive/unclear tail below 0.8), plus a text-derived `numerical_density` (quantitative tokens per 100 words, no model).
4. **Signal** — for each dimension, the same `level_z` / `drift_z` treatment as FinBERT: z-scored against the firm's own prior baseline, firm-relative not universe-relative.

---

## The backtest

The backtest runs on Financial Modeling Prep data and is independent of IBKR.

### Step 1 — generate candidates (expensive, run once)

```bash
export FMP_API_KEY=...

python backtest.py --months 36          # full run (GPU recommended; ~10h cold)
python backtest.py --months 36 --max-tickers 50   # pilot
```

Writes `backtest_trades.csv` — one row per priced candidate — with:
- FinBERT `composite`, `level_z`, `drift_z`, `signal_blend`
- 18 subjectivity dimension fields (`level_z_{dim}`, `drift_z_{dim}`, `frac_low_z_{dim}` for each of the six dimensions)
- 2 numerical-density fields (`level_z_numerical_density`, `drift_z_numerical_density`)
- Gross forward returns at 30/90/180 trading days (daily bars, entry at next trading day's open)
- Matched SPY returns over the identical entry day and horizon

Does **not** apply thresholds, direction, or cost — those live in the analyzer.

Both scorers cache to disk (`backtest_scores.json`, `subjectivity_scores.json`). Interrupted runs resume from cache. See `runpod_setup_backtest.sh` to run on a GPU pod.

### Step 2 — analyze (cheap, re-runnable)

```bash
pip install -r requirements.txt   # adds scikit-learn vs v2
python backtest_analyze.py
python backtest_analyze.py --long-only   # trigger only long-side predictions
```

Reads `backtest_trades.csv`, writes `backtest_summary.csv` + `backtest_figure.png`.

The analyzer runs an **anchored expanding walk-forward** (12-month initial train, monthly test slices, 2-day embargo, label-window purge) across **three feature arms** on the same splits:

| Arm | Features | Purpose |
| --- | -------- | ------- |
| `finbert` | composite, level_z, drift_z | FinBERT-only baseline (3 features) |
| `lexical` | numerical-density level_z/drift_z | Model-free, CPU-only baseline (2 features) |
| `subjectivity` | 18 SubjECTive-QA dimension fields | The new signal under test |
| `combined` | all of the above | Complementarity side-check (not the headline) |

Per fold: `StandardScaler` fit on train only → `ElasticNetCV` with `TimeSeriesSplit` inner CV, predicting the **matched excess return** (`ret_Nd − spy_ret_Nd`). Trade is triggered when `|predicted excess| > 40bps`; `net = sign(pred) · realized_excess − 40bps`.

Two readouts:

1. **Rank-IC** (Spearman of predicted vs realized excess over *all* test trades each month, averaged across folds) — the paired, threshold-free, intercept-invariant signal-quality comparison. This is the honest head-to-head: same test trades for every arm, no trigger threshold involved.
2. **Pooled OOS mean net-of-cost ± 95% CI** — the economic readout. Each arm triggers a different subset, so these are not paired; read alongside the rank-IC, not instead of it.

The headline is the **subjectivity vs finbert** paired monthly ΔIC at the 90d horizon. 30d and 180d run as robustness. Mean ElasticNet coefficients across folds identify which dimensions carry the edge.

### Running on RunPod

```bash
# On the pod
export FMP_API_KEY=your_key_here
export RUNPOD_API_KEY=your_runpod_key   # enables auto-stop on completion
bash runpod_setup_backtest.sh
```

See `runpod_setup_backtest.sh` for cache-seeding instructions (both score caches + subjectivity checkpoint). The subjectivity checkpoint (`subjectivity_model/`) must be present; see below for training it.

### Training the subjectivity model (one-time)

```bash
export HF_TOKEN=hf_...    # account must have accepted SubjECTive-QA terms
bash runpod_setup_subjectivity.sh
```

Or directly:

```bash
python train_subjectivity.py --dataset-config 5768
```

Trains a RoBERTa encoder with six 3-class heads on [gtfintechlab/SubjECTive-QA](https://huggingface.co/datasets/gtfintechlab/SubjECTive-QA). Saves checkpoint to `subjectivity_model/`. Pass `--smoke-test` for a 2-minute end-to-end check before committing to the full run.

**Scope note:** trained on 120 large-cap NYSE companies, 2007–2021, Q&A pairs only. Large-cap → broader-universe transfer is unvalidated. If a dimension's feature looks dead in the walk-forward, suspect transfer before concluding the dimension is uninformative.

---

## The live pipeline (v1 lineage)

### Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

Rebalancing requires an IBKR paper-trading gateway running locally on `127.0.0.1:7497`.

### Usage

**Bootstrap (first run only)** — scrape and score a large batch of historical transcripts to build up signal history.

```bash
python main.py --bootstrap --pages 100
```

**Daily run (scrape + score only)**

```bash
python main.py
```

**Daily run with rebalancing**

```bash
python main.py --rebalance                            # live
python main.py --rebalance --dry-run                  # print target weights, place nothing
python main.py --rebalance --portfolio-value 100000   # override auto-fetched value
```

Risk / holding parameters (optional, shown with defaults):

```bash
python main.py --rebalance \
  --holding-days 63 \      # exit after this many trading days
  --stop-loss-pct 0.15 \   # exit if a position drops 15% from entry
  --take-profit-pct 0.25   # exit if a position gains 25% from entry
```

**Standalone rebalance**

```bash
python rebalance.py --today-tickers AAPL MSFT NVDA
```

**Visualize performance** (needs ≥2 days of runs)

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

## Working with versions

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
| `preprocessing.py` | Split / clean / sentence-tokenize / segment Q&A pairs |
| `sentiment_scoring.py` | FinBERT scoring and per-transcript composite |
| `subjectivity_scoring.py` | SubjECTive-QA inference, aggregation, and cache |
| `train_subjectivity.py` | Fine-tune the multi-task subjectivity model (one-time) |
| `signal_constructor.py` | Field-agnostic drift signal, ranking, investable universe |
| `rebalance.py` | Exit logic + Markowitz min-variance weights |
| `ibkr_manager.py` | IBKR connection, pricing, order placement |
| `main.py` | Orchestrates the live pipeline (bootstrap / daily / rebalance) |
| `visualize.py` | Performance chart + summary |
| `persistence.py` | JSON/CSV state for scores, positions, performance |
| `fmp_client.py` | Financial Modeling Prep API wrapper (backtest) |
| `backtest.py` | Candidate generation, subjectivity scoring, forward pricing |
| `backtest_analyze.py` | ElasticNet walk-forward, three-arm OOS comparison, verdict |
| `runpod_setup.sh` | RunPod provisioning (FinBERT-only backtest, legacy) |
| `runpod_setup_backtest.sh` | RunPod provisioning (both scorers, current) |
| `runpod_setup_subjectivity.sh` | RunPod provisioning (subjectivity model training) |

## Generated files (gitignored)

`transcript_scores.json`, `positions.json`, `signals_output.csv`, `performance_log.csv`, `performance_chart.png`, and the backtest artifacts: `backtest_trades.csv`, `backtest_summary.csv`, `backtest_scores.json`, `backtest_figure.png`, `subjectivity_scores.json`, `subjectivity_model/`.