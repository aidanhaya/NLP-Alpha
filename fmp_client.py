"""
fmp_client.py — thin, defensive wrapper over the Financial Modeling Prep (FMP) API
for the historical backtest (backtest.py).

Endpoints used (all /stable/, Ultimate plan). VERIFY the exact JSON shapes against
your own account once — FMP occasionally renames fields between tiers/versions, so
every extractor below tries a few key spellings and fails soft. Run this module
directly (`python fmp_client.py AAPL`) for a quick smoke test of each endpoint.

    transcripts (window) : /stable/earning-call-transcript-latest?page=&limit=
    transcript (one)     : /stable/earning-call-transcript?symbol=&year=&quarter=
    transcript dates     : /stable/earning-call-transcript-dates?symbol=
    historical mkt cap   : /stable/historical-market-capitalization?symbol=&from=&to=
    intraday bars        : /stable/historical-chart/{interval}?symbol=&from=&to=

API key is read from the FMP_API_KEY environment variable.
All datetimes are treated as US/Eastern (FMP's convention for US equities).
"""

import os
import time
import sys
from datetime import timedelta
from datetime import date, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://financialmodelingprep.com/stable"


# creates a requests.Session to reuse an underlying TCP connection across all calls
def _session() -> requests.Session:
    s = requests.Session()
    # upon failure, retry 4 times with increasing wait time (backoff factor)
    retry = Retry(
        total=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16))
    return s

# returns the first JSON field present (since sometimes this changes with FMP)
def _first(d: dict, *keys, default=None):
    """Return the first present, non-None key from d (tolerates FMP renames)."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


class FMPClient:
    def __init__(self, api_key: str = None, polite_delay: float = 0.0):
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set FMP_API_KEY (export FMP_API_KEY=...).")
        self.s = _session()
        # Ultimate is 3000 req/min; 0 delay is fine, but expose a knob.
        self.polite_delay = polite_delay

    def _get(self, path: str, **params) -> list | dict:
        params["apikey"] = self.api_key
        url = f"{BASE}/{path}"
        r = self.s.get(url, params=params, timeout=30)
        r.raise_for_status() # turns 4xx/5xx responses into exceptions
        if self.polite_delay:
            time.sleep(self.polite_delay)
        data = r.json()
        return data

    # --- transcript enumeration ---

    def list_transcripts_in_window(self, start: date, end: date,
            page_size: int = 100, max_pages: int = 400) -> list[dict]:
        """
        Walk the latest-transcripts feed backward until we pass `start`.
        Returns dicts: {symbol, year, quarter, dt (datetime, ET), date (date)}.
        """
        out, seen = [], set()
        for page in range(max_pages):
            try:
                rows = self._get("earning-call-transcript-latest", page=page, limit=page_size)
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 400:
                    break  # hit FMP's page cap — treat as end of results
                raise  # re-raise anything else (401, 500, etc.)
            if not rows:
                break
            page_min_date = None
            for row in rows:
                sym = _first(row, "symbol", "ticker")
                yr = _first(row, "year", "fiscalYear")
                q = _first(row, "quarter", "period")
                dt = parse_fmp_datetime(_first(row, "date", "datetime", "publishedDate"))
                if not (sym and yr and q is not None and dt):
                    continue
                yr, q = _coerce_year_quarter(yr, q)
                if yr is None:
                    continue
                d = dt.date()
                if page_min_date is None or d < page_min_date:
                    page_min_date = d
                if d < start or d > end:
                    continue
                key = (sym.upper(), yr, q)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"symbol": sym.upper(), "year": yr, "quarter": q,
                            "dt": dt, "date": d})
            # Stop once the whole page predates the window.
            if page_min_date is not None and page_min_date < start:
                break
        return out

    def transcript_dates(self, symbol: str) -> list[dict]:
        """All available transcripts for a symbol:
        [{year, quarter, dt}, ...] (oldest→newest)."""
        rows = self._get("earning-call-transcript-dates", symbol=symbol.upper())
        out = []
        for row in rows or []:
            yr = _first(row, "year", "fiscalYear")
            q = _first(row, "quarter", "period")
            dt = parse_fmp_datetime(_first(row, "date", "datetime"))
            yr, q = _coerce_year_quarter(yr, q)
            if yr is None or q is None:
                continue
            out.append({"year": yr, "quarter": q, "dt": dt})
        out.sort(key=lambda r: (r["year"], r["quarter"]))
        return out

    def get_transcript(self, symbol: str, year: int, quarter: int) -> dict | None:
        """Full transcript: {symbol, year, quarter, dt, content (str)} or None."""
        rows = self._get("earning-call-transcript", symbol=symbol.upper(),
            year=year, quarter=quarter)
        if not rows:
            return None
        row = rows[0] if isinstance(rows, list) else rows
        content = _first(row, "content", "transcript", "text", default="")
        return {
            "symbol": symbol.upper(),
            "year": year,
            "quarter": quarter,
            "dt": parse_fmp_datetime(_first(row, "date", "datetime")),
            "content": content or "",
        }


    # --- point-in-time market cap ---

    def historical_market_cap(self, symbol: str,
                              start: date = None, end: date = None) -> list[tuple[date, float]]:
        """Sorted [(date, marketCap), ...] ascending. One call covers the lookup series."""
        params = {"symbol": symbol.upper(), "limit": 5000}
        if start:
            params["from"] = start.isoformat()
        if end:
            params["to"] = end.isoformat()
        rows = self._get("historical-market-capitalization", **params)
        series = []
        for row in rows or []:
            d = parse_fmp_datetime(_first(row, "date", "datetime"))
            mc = _first(row, "marketCap", "marketCapitalization")
            if d and mc is not None:
                series.append((d.date(), float(mc)))
        series.sort(key=lambda x: x[0])
        return series

    # --- intraday bars ---

    def intraday_bars(self, symbol: str, interval: str,
                      start: date, end: date) -> list[dict]:
        """
        OHLCV bars for [start, end]. interval in {'1min','5min','15min','1hour',...}.
        Returns [{dt (datetime ET), open, high, low, close, volume}, ...] ascending.
        """
        rows = self._get(f"historical-chart/{interval}", symbol=symbol.upper(),
                         **{"from": start.isoformat(), "to": end.isoformat()})
        bars = []
        for row in rows or []:
            dt = parse_fmp_datetime(_first(row, "date", "datetime"))
            if not dt:
                continue
            bars.append({
                "dt": dt,
                "open": _to_float(_first(row, "open")),
                "high": _to_float(_first(row, "high")),
                "low": _to_float(_first(row, "low")),
                "close": _to_float(_first(row, "close")),
                "volume": _to_float(_first(row, "volume"), default=0.0),
            })
        bars.sort(key=lambda b: b["dt"])
        return bars


# --- normalization helpers ---

def _to_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_year_quarter(yr, q):
    """Normalize year→int and quarter ('Q3', 3, '3')→int in {1..4}."""
    try:
        yr = int(yr)
    except (TypeError, ValueError):
        return None, None
    if isinstance(q, str):
        q = q.upper().replace("Q", "").strip()
    try:
        q = int(q)
    except (TypeError, ValueError):
        return yr, None
    if q not in (1, 2, 3, 4):
        return yr, None
    return yr, q


def parse_fmp_datetime(v) -> datetime | None:
    """Parse FMP date strings. Returns naive datetime (interpreted as US/Eastern)."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    s = str(v).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s[:len(fmt) + 6], fmt) if "%f" in fmt \
                else datetime.strptime(s[:19] if len(s) >= 19 and ("T" in s or " " in s) else s[:10], fmt)
        except ValueError:
            continue
    # last resort: ISO
    try:
        return datetime.fromisoformat(s.replace("Z", ""))
    except ValueError:
        return None


def main():
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    c = FMPClient()
    print(f"== smoke test for {sym} ==")
    today = date.today()

    dates = c.transcript_dates(sym)
    print(f"transcript_dates: {len(dates)} found; latest: {dates[-3:]}")

    if dates:
        last = dates[-1]
        t = c.get_transcript(sym, last["year"], last["quarter"])
        print(f"get_transcript: dt={t['dt']} content_chars={len(t['content'])}")

    mc = c.historical_market_cap(sym, today - timedelta(days=400), today)
    print(f"historical_market_cap: {len(mc)} rows; latest: {mc[-1] if mc else None}")

    bars = c.intraday_bars(sym, "1min", today - timedelta(days=370),
                           today - timedelta(days=368))
    print(f"intraday_bars(1min, ~1yr ago): {len(bars)} bars; "
          f"first: {bars[0]['dt'] if bars else None}")


if __name__ == "__main__":
    main()