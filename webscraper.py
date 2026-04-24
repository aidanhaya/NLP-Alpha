from bs4 import BeautifulSoup
import re
import time
from datetime import datetime
from playwright.sync_api import sync_playwright

LISTING_PAGES = [
    "https://www.fool.com/earnings-call-transcripts/",
    "https://www.fool.com/earnings-call-transcripts/page/2/",
]

def scrape_all_transcripts(delay: float = 2.0) -> dict:
    results = {}
    today = str(datetime.today().date())  # e.g. "2026-04-24"

    with sync_playwright() as p: # opens a playwright session
        # launches real Chromium browser
        browser = p.chromium.launch(headless=True) # set to false to show visible window
        # opens Chromium browser tab
        page = browser.new_page()

        # Step 1: Collect transcript links from both listing pages
        pattern = re.compile(
            r"/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/"  # date path
            r"(?:[a-z0-9]+-)*"  # company name slug (non-capturing)
            r"([A-Z]{1,5}|[a-z]{1,5})"  # ticker capture group
            r"-q[1-4]-\d{4}"  # quarter + year
            r"-earnings(?:-call)?-transcript/"  # either format
        )
        date_pattern = re.compile(r"/earnings/call-transcripts/(\d{4})/(\d{2})/(\d{2})/")
        seen_tickers = set()
        ticker_href_pairs = []

        for listing_url in LISTING_PAGES:
            page.goto(listing_url, timeout=60000)
            page.wait_for_selector("a[href*='/earnings/call-transcripts/']", timeout=30000)
            soup = BeautifulSoup(page.content(), "html.parser")

            for l in soup.find_all("a", href=True):
                href = l["href"]
                if "/earnings/call-transcripts/" not in href:
                    continue
                match = pattern.search(href)
                if not match:
                    continue
                ticker = match.group(1).lower()
                if ticker in seen_tickers:
                    continue
                # filter to today's transcripts only using the date in the URL
                date_match = date_pattern.search(href)
                if date_match:
                    link_date = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                    if link_date != today:
                        continue
                seen_tickers.add(ticker)
                ticker_href_pairs.append((ticker, href))

        print(f"Found {len(ticker_href_pairs)} tickers from today ({today}): {[t for t, _ in ticker_href_pairs]}")

        # Step 2: Visit each transcript page in the same browser session
        for ticker, href in ticker_href_pairs:
            print(f"Scraping {ticker}...")
            try:
                transcript_url = href if href.startswith("http") else "https://www.fool.com" + href
                page.goto(transcript_url, timeout=60000)
                page.wait_for_selector("div.article-body", timeout=30000)
                soup = BeautifulSoup(page.content(), "html.parser")

                # extracts text under "article-body" div
                article = soup.find("div", class_="article-body")
                if not article:
                    print(f" > No article body found for {ticker}, skipping.")
                    continue

                paragraphs = [p.get_text(strip=True) for p in article.find_all("p")]
                full_text = "\n".join(paragraphs)

                date_tag = soup.find("time")
                if date_tag:
                    date = date_tag.get("datetime") or date_tag.get_text(strip=True)
                else:
                    date = str(datetime.today().date())

                results[ticker] = {
                    "ticker": ticker,
                    "url": transcript_url,
                    "date": date,
                    "text": full_text,
                }

            # handles 404 errors, browser crashes, no "article-body" div, etc
            except Exception as e:
                print(f" > Error scraping {ticker}: {e}")

            time.sleep(delay)

        browser.close()

    return results

def scrape_historical_transcripts(
    n_pages: int = 40,
    delay: float = 2.0,
    already_scored: set = None, # set of (ticker, date) tuples to skip
) -> dict:
    if already_scored is None:
        already_scored = set()

    # list of all page URLs - page 1 has different format, so hardcoded
    page_urls = ["https://www.fool.com/earnings-call-transcripts/"] + [
        f"https://www.fool.com/earnings-call-transcripts/page/{n}/"
        for n in range(2, n_pages + 1)
    ]

    # regex object to match valid transcript URLs and capture ticker symbols
    pattern = re.compile(
        r"/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/"
        r"(?:[a-z0-9]+-)*"
        r"([A-Z]{1,5}|[a-z]{1,5})"
        r"-q[1-4]-\d{4}"
        r"-earnings(?:-call)?-transcript/"
    )
    # regex object to capture year, month, date from URL path
    date_pattern = re.compile(r"/earnings/call-transcripts/(\d{4})/(\d{2})/(\d{2})/")

    results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Step 1: Collect (ticker, date_str, href) across all listing pages
        seen_pairs = set()
        candidates = []

        # loops through page_urls to scrape transcript links from each page
        for listing_url in page_urls:
            print(f"Collecting links from {listing_url}...")
            try:
                page.goto(listing_url, timeout=60000)
                page.wait_for_selector("a[href*='/earnings/call-transcripts/']", timeout=30000)
            except Exception as e:
                print(f" > Could not load {listing_url}: {e}")
                # exception => gone past last page, so breaks after one exception
                break
            soup = BeautifulSoup(page.content(), "html.parser")

            # parses page HTML and loops over all links
            for l in soup.find_all("a", href=True):
                href = l["href"]
                # skips non-transcript links
                if "/earnings/call-transcripts/" not in href:
                    continue
                # tries to extract ticker from URL
                match = pattern.search(href)
                if not match:
                    continue
                ticker = match.group(1).lower()
                # tries to extract date from URL
                date_match = date_pattern.search(href)
                if not date_match:
                    continue
                date_str = f"{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}"
                pair = (ticker, date_str)
                # skips duplicates
                if pair in seen_pairs:
                    continue
                # adds new candidate
                seen_pairs.add(pair)
                candidates.append((ticker, date_str, href))

        # Step 2: Filter out already-scored pairs
        # cross-reference against set in JSON store (O(1))
        to_scrape = [(t, d, h) for t, d, h in candidates if (t, d) not in already_scored]
        print(f"Found {len(candidates)} unique (ticker, date) pairs; "
              f"{len(candidates) - len(to_scrape)} already scored, "
              f"{len(to_scrape)} to scrape.")

        # Step 3: Visit each transcript page
        for ticker, date_str, href in to_scrape:
            print(f"Fetching {ticker} ({date_str})...")
            try:
                transcript_url = href if href.startswith("http") else "https://www.fool.com" + href
                page.goto(transcript_url, timeout=60000)
                page.wait_for_selector("div.article-body", timeout=30000)
                soup = BeautifulSoup(page.content(), "html.parser")

                article = soup.find("div", class_="article-body")
                if not article:
                    print(f" > No article body for {ticker} ({date_str}), skipping.")
                    continue

                paragraphs = [p.get_text(strip=True) for p in article.find_all("p")]
                full_text = "\n".join(paragraphs)

                date_tag = soup.find("time")
                # searches for time div, uses date_str in tuple if not found
                if date_tag:
                    date = date_tag.get("datetime") or date_tag.get_text(strip=True)
                else:
                    date = date_str

                results[(ticker, date_str)] = {
                    "ticker": ticker,
                    "url": transcript_url,
                    "date": date,
                    "text": full_text,
                }

            except Exception as e:
                print(f" > Error fetching {ticker} ({date_str}): {e}")

            time.sleep(delay)

        browser.close()

    return results