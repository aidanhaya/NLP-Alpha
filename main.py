import argparse
from collections import Counter
from datetime import datetime

import webscraper as ws
import preprocessing as pp
import sentiment_scoring as scr
import signal_constructor as sc
import persistence

# helper to run a single transcript through the processing pipeline
def _run_pipeline(data: dict, ticker: str, date: str, scorer: scr.FinBERTScorer) -> dict:
    # splits raw text into prepared and qa sections
    split_text = pp.split_transcript(data["text"])
    # cleans each section
    cleaned = {
        "prepared": pp.clean_text(split_text["prepared"]),
        "qa": pp.clean_text(split_text["qa"]),
    }
    # breaks each section into individual sentences
    tokenized = {
        "prepared": pp.sentence_tokenize(cleaned["prepared"]),
        "qa": pp.sentence_tokenize(cleaned["qa"]),
    }
    # runs sentiment analysis on the tokenized sections
    scored = scr.score_transcript(tokenized, ticker, date, scorer)
    scored["date"] = scored["date"][:10]
    return scored

def main():
    parser = argparse.ArgumentParser()
    # lets us control whether we are running bootstrap or daily version from CLI
    parser.add_argument("--bootstrap", action="store_true",
                        help="Scrape historical transcripts across many pages")
    parser.add_argument("--pages", type=int, default=40,
                        help="Number of listing pages to scrape in bootstrap mode (default: 40)")
    args = parser.parse_args()

    # loads previously saved scores from disk
    all_records = persistence.load_scores()
    # builds set of already scored (ticker, date) pairs
    already_scored = persistence.get_scored_keys(all_records)
    # initializes FinBERT model
    scorer = scr.FinBERTScorer()

    # bootstrap branch
    if args.bootstrap:
        raw = ws.scrape_historical_transcripts(
            n_pages=args.pages,
            already_scored=already_scored,
        )
        total = len(raw)
        for i, ((ticker, date_str), data) in enumerate(raw.items()):
            print(f"[{i+1}/{total}] Scoring {ticker} ({date_str})...")
            try:
                scored = _run_pipeline(data, ticker, date_str, scorer)
                all_records.append(scored)
            except Exception as e:
                print(f" > Error scoring {ticker} ({date_str}): {e}")
            # saves scores every 10 transcripts as a checkpoint
            if (i + 1) % 10 == 0:
                persistence.save_scores(all_records)
        # final save at the end
        persistence.save_scores(all_records)
    # daily branch
    else:
        today = str(datetime.today().date())
        raw = ws.scrape_all_transcripts()
        for ticker, data in raw.items():
            if (ticker, today) in already_scored:
                print(f"Already scored {ticker} today, skipping.")
                continue
            print(f"Scoring {ticker}...")
            try:
                scored = _run_pipeline(data, ticker, today, scorer)
                all_records.append(scored)
            except Exception as e:
                print(f" > Error scoring {ticker}: {e}")
        persistence.save_scores(all_records)

        # For any ticker from today's scrape that is below the threshold,
        # immediately backfill its history via the quote page scraper.
        # counts is a frequency for how many transcripts are recorded per ticker
        counts = Counter(r["ticker"] for r in all_records)
        # filters daily tickers to find those with fewer than 8 total records
        needs_backfill = [t for t in raw if counts.get(t, 0) < 8]
        if needs_backfill:
            print(f"Backfilling history for {len(needs_backfill)} tickers below {8} records: {needs_backfill}")
            # fetches set of (ticker, date) pairs that have already been scored
            already_scored = persistence.get_scored_keys(all_records)
            backfill_raw = ws.scrape_ticker_history(
                tickers=needs_backfill,
                already_scored=already_scored,
            )
            # tuple key = (ticker, date_str)
            # value = data
            for (ticker, date_str), data in backfill_raw.items():
                try:
                    scored = _run_pipeline(data, ticker, date_str, scorer)
                    all_records.append(scored)
                except Exception as e:
                    print(f" > Error scoring {ticker} ({date_str}): {e}")
            persistence.save_scores(all_records)

    # Signal generation
    signal = sc.SentimentSignal()
    # provides history in SentimentSignal class with historical results
    for record in all_records:
        signal.add_score(record)

    # in daily mode, rank only tickers with transcripts today; in bootstrap, rank everything
    if args.bootstrap:
        tickers = list({r["ticker"] for r in all_records})
    else:
        tickers = list(raw.keys())
    if not tickers:
        print("No scored transcripts found. Run with --bootstrap first.")
        return

    df = signal.rank_universe(tickers) # ranked df of candidate tickers
    print("\n=== Signal Rankings ===")
    print(df.round(4).to_string(index=False))
    df.to_csv("signals_output.csv", index=False)

    investable = signal.get_investable_universe(tickers)
    print(f"\nTop 20% investable universe ({len(investable)} tickers): {investable}")

if __name__ == "__main__":
    main()