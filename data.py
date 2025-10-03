"""
Market Data Module
==================
Fetches price data from Polygon.io (free tier) or Yahoo Finance as fallback.
Falls back to synthetic GBM data if no API key is set.
"""

import os
import json
import math
import random
import urllib.request
import urllib.parse
from datetime import datetime, timedelta


POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")


def fetch_prices(ticker: str, start: str, end: str) -> dict:
    """
    Returns { 'dates': [...], 'prices': [...] }
    Tries Polygon → Yahoo Finance → Synthetic fallback.
    """
    if POLYGON_API_KEY:
        result = _fetch_polygon(ticker, start, end)
        if result:
            return result

    result = _fetch_yahoo(ticker, start, end)
    if result:
        return result

    print(f"[data] Using synthetic GBM data for {ticker}")
    return _synthetic_gbm(ticker, start, end)


# ── Polygon.io ──────────────────────────────────────────────────────────────

def _fetch_polygon(ticker: str, start: str, end: str) -> dict | None:
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start}/{end}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        if data.get("resultsCount", 0) == 0:
            return None
        results = data["results"]
        dates  = [datetime.utcfromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d") for r in results]
        prices = [r["c"] for r in results]
        return {"dates": dates, "prices": prices, "source": "Polygon.io"}
    except Exception as e:
        print(f"[data] Polygon error: {e}")
        return None


# ── Yahoo Finance (unofficial) ──────────────────────────────────────────────

def _fetch_yahoo(ticker: str, start: str, end: str) -> dict | None:
    try:
        t1 = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
        t2 = int(datetime.strptime(end,   "%Y-%m-%d").timestamp())
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            f"?interval=1d&period1={t1}&period2={t2}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        chart = data["chart"]["result"][0]
        timestamps = chart["timestamp"]
        closes = chart["indicators"]["quote"][0]["close"]

        pairs = [(datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"), c)
                 for t, c in zip(timestamps, closes) if c is not None]
        dates, prices = zip(*pairs)
        return {"dates": list(dates), "prices": list(prices), "source": "Yahoo Finance"}
    except Exception as e:
        print(f"[data] Yahoo error: {e}")
        return None


# ── Synthetic GBM fallback ──────────────────────────────────────────────────

