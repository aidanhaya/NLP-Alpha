# NLP-Markowitz

> **Work in progress.** The earnings call webscraper and FinBERT sentiment scoring pipeline are largely complete. The next phase is integrating the signal output with an IBKR paper trading account and applying Markowitz Mean-Variance Optimization, the Efficient Frontier, the Sharpe Ratio, and PCA to construct an optimal portfolio from the tickers that pass a configurable signal threshold.

---

## How it works

1. **Scrape** — Playwright fetches earnings call transcripts from Motley Fool.
2. **Preprocess** — Transcripts are split into prepared remarks and Q&A sections, cleaned, and sentence-tokenized.
3. **Score** — Each sentence is scored with [ProsusAI/FinBERT](https://huggingface.co/ProsusAI/finbert). The Q&A section is weighted more heavily (60/40) as it is less scripted.
4. **Signal** — A drift signal is computed per ticker from the change in composite sentiment across transcripts. Tickers are ranked and a top-percentile investable universe is selected.

Scores are persisted to `transcript_scores.json` (gitignored) so re-runs skip already-scored transcripts. Signal rankings are written to `signals_output.csv` (gitignored) after each run.

---

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

---

## Usage

### Bootstrap (first run)

Scrapes and scores a large batch of historical transcripts. Each listing page covers roughly 20 transcripts, so `--pages 100` yields ~2,000 transcripts.

```bash
python main.py --bootstrap --pages 100
```

You can extend the history later — already-scored transcripts are skipped automatically:

```bash
python main.py --bootstrap --pages 200
```

### Daily run

Scrapes today's transcripts, backfills history for any tickers below 8 records, then regenerates signal rankings.

```bash
python main.py
```