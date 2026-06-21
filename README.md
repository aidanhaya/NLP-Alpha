# NLP-Alpha

**Natural language processing of earnings call transcripts to predict returns.**

NLP-Alpha scores the tone of earnings-call transcripts with FinBERT and a fine-tuned subjectivity model, turns those scores into per-ticker signals, and runs them through a **point-in-time backtest** that determines whether the signal has any edge. An earlier version of the project traded the signal live through Interactive Brokers; that pipeline is preserved at the `v1.0.0` tag (see [Project history](#project-history)).

This repository is the consolidated home of a project that has gone through four tagged milestones. The earliest two lived in separate repos (`NLP-Markowitz`, `NLP-Modified`); all of it now lives here as git tags so the whole arc is in one place.

---

## Project history

| Version | Tag | Focus |
| ------- | --- | ----- |
| **v1** | [`v1.0.0`](../../releases/tag/v1.0.0) | Motley Fool transcripts → FinBERT → Markowitz → IBKR paper-trading pipeline |
| **v2** | [`v2.0.0`](../../releases/tag/v2.0.0) | FMP data + FMP transcripts; point-in-time backtest harness |
| **v3.1** | [`v3.1.0`](../../releases/tag/v3.1.0) | Retail-universe testing; walk-forward CIs, CAPM, SPY-matched returns |
| **v3.2** | [`v3.2.0`](../../releases/tag/v3.2.0) `master` *(current)* | Multi-faceted sentiment (RoBERTa/SubjECTive-QA), ElasticNet optimization, Newey-West, full 3-year backtest |

### v1 — The portfolio (`v1.0.0`)

The first version was an end-to-end trading system. It scrapes earnings-call transcripts from Motley Fool, scores them with FinBERT, ranks tickers by a sentiment **drift** signal, selects a top-percentile investable universe, sizes positions with Markowitz minimum-variance optimization, and executes through the IBKR paper-trading API — with stop-loss, take-profit, time-limit, and signal-dropout exits. The live pipeline (`main.py`, `webscraper.py`, `rebalance.py`, `ibkr_manager.py`, `visualize.py`) lives only at this tag; see [Working with versions](#working-with-versions) to check it out.

### v2 — The backtest (`v2.0.0`)

v2 moved off live execution and onto Financial Modeling Prep data to test the signal properly, out-of-sample. The backtest harness (`backtest.py` + `backtest_analyze.py`) is built around a clean separation: an expensive run-once step scores every in-window transcript and prices forward returns at several horizons; a cheap, re-runnable step sweeps thresholds, signal definitions, and directions (fade vs. momentum), applies a flat 40 bps round-trip cost, splits train/test, and reports a verdict.

### v3.1 — Sharpening the verdict (`v3.1.0`)

v3.1 tightened the backtest's claim to credibility: candidates gained a retail-accessibility label (market cap floor, US-listing filter) so the universe matched what a retail account could actually trade, the walk-forward gained confidence intervals and a CAPM-adjusted readout alongside the raw spread, and forward returns were matched against SPY over the identical entry window rather than judged in isolation.

### v3.2 — Multi-faceted sentiment (`v3.2.0`, current)

v3.2 is the largest expansion since v1. It adds a second scorer alongside FinBERT: a RoBERTa encoder fine-tuned on the SubjECTive-QA dataset with six independent classification heads (Assertive, Cautious, Optimistic, Specific, Clear, Relevant) that measure *how* management communicates, orthogonal to FinBERT's polarity. The retail-universe filter from v3.1 was dropped — broadening back out didn't change the verdict — and the analyzer was rewritten around a regularized linear model (`ElasticNetCV`) that evaluates the six subjectivity dimensions as a feature set inside an anchored walk-forward, with Newey-West standard errors to account for the serial correlation between overlapping 30/90/180-day holding periods. The backtest window itself was also extended to a genuine three years: `backtest.py` now derives each symbol's trade window from its own transcript history rather than FMP's discovery feed, which silently capped multi-year runs to about six months of trades.

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

The backtest runs on Financial Modeling Prep data and is fully self-contained — no IBKR dependency.

### Step 1 — generate candidates (expensive, run once)

```bash
export FMP_API_KEY=...

python backtest.py --months 36 # full run (GPU recommended; ~10h cold)
python backtest.py --months 36 --max-tickers 50 # pilot
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
pip install -r requirements.txt
python backtest_analyze.py
python backtest_analyze.py --long-only # trigger only long-side predictions
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
2. **Pooled OOS mean net-of-cost ± 95% CI** — the economic readout, with a Newey-West (Bartlett-kernel) standard error to account for the autocorrelation between overlapping holding periods. Each arm triggers a different subset, so these are not paired; read alongside the rank-IC, not instead of it.

The headline is the **subjectivity vs finbert** paired monthly ΔIC at the 90d horizon. 30d and 180d run as robustness. Mean ElasticNet coefficients across folds identify which dimensions carry the edge.

### Running on RunPod

```bash
# On the pod
export FMP_API_KEY=your_key_here
export RUNPOD_API_KEY=your_runpod_key # enables auto-stop on completion
bash runpod_setup_backtest.sh
```

See `runpod_setup_backtest.sh` for cache-seeding instructions (both score caches + subjectivity checkpoint). The subjectivity checkpoint (`subjectivity_model/`) must be present; see below for training it.

### Training the subjectivity model (one-time)

```bash
export HF_TOKEN=hf_... # account must have accepted SubjECTive-QA terms
bash runpod_setup_subjectivity.sh
```

Or directly:

```bash
python train_subjectivity.py --dataset-config 5768
```

Trains a RoBERTa encoder with six 3-class heads on [gtfintechlab/SubjECTive-QA](https://huggingface.co/datasets/gtfintechlab/SubjECTive-QA). Saves checkpoint to `subjectivity_model/`. Pass `--smoke-test` for a 2-minute end-to-end check before committing to the full run.

**Scope note:** trained on 120 large-cap NYSE companies, 2007–2021, Q&A pairs only. Large-cap → broader-universe transfer is unvalidated. If a dimension's feature looks dead in the walk-forward, suspect transfer before concluding the dimension is uninformative.

---

## The live pipeline (archived at `v1.0.0`)

The original IBKR/Markowitz pipeline — scrape, score, rank, rebalance, execute — is no longer part of `master`; it was retired once the project's focus moved fully to the backtest. The code (`main.py`, `webscraper.py`, `rebalance.py`, `ibkr_manager.py`, `visualize.py`) and its usage instructions still live at the `v1.0.0` tag:

```bash
git checkout v1.0.0
cat README.md # full live-pipeline setup and usage instructions, as they were at v1
```

---

## Working with versions

```bash
git checkout v1.0.0    # the original Markowitz/IBKR pipeline
git checkout v2.0.0    # the first backtest harness
git checkout v3.1.0    # retail-universe testing, CAPM/CI refinements
git checkout v3.2.0    # multi-faceted sentiment, ElasticNet, Newey-West
git checkout master    # current work (== v3.2.0)
```

`master` is always the current line of development; the tags are immutable snapshots.

---

## Repository layout

| File | Role |
| ---- | ---- |
| `preprocessing.py` | Split / clean / sentence-tokenize / segment Q&A pairs |
| `sentiment_scoring.py` | FinBERT scoring and per-transcript composite |
| `subjectivity_scoring.py` | SubjECTive-QA inference, aggregation, and cache |
| `train_subjectivity.py` | Fine-tune the multi-task subjectivity model (one-time) |
| `signal_constructor.py` | Field-agnostic drift signal, ranking, investable universe |
| `fmp_client.py` | Financial Modeling Prep API wrapper (backtest) |
| `backtest.py` | Candidate generation, subjectivity scoring, forward pricing |
| `backtest_analyze.py` | ElasticNet walk-forward, three-arm OOS comparison, verdict |
| `runpod_setup.sh` | RunPod provisioning (FinBERT-only backtest, legacy) |
| `runpod_setup_backtest.sh` | RunPod provisioning (both scorers, current) |
| `runpod_setup_subjectivity.sh` | RunPod provisioning (subjectivity model training) |

The live pipeline's files (`main.py`, `webscraper.py`, `rebalance.py`, `ibkr_manager.py`, `visualize.py`) are not present on `master` — see [the live pipeline](#the-live-pipeline-archived-at-v100) above.

## Generated files (gitignored)

Backtest artifacts: `backtest_trades.csv`, `backtest_summary.csv`, `backtest_scores.json`, `backtest_figure.png`, `subjectivity_scores.json`, `subjectivity_model/`.