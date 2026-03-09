"""
Scan engine: orchestrates parallel data fetching and per-analyst scoring
across a watchlist of tickers.

All scoring is LLM-free (rule-based).  Results are ranked by composite score.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.scanner.data import TickerData, fetch_ticker_data
from src.scanner.scorers import (
    score_ben_graham,
    score_bill_ackman,
    score_cathie_wood,
    score_fundamentals,
    score_growth,
    score_michael_burry,
    score_sentiment,
    score_technical,
    score_valuation,
    score_warren_buffett,
)

# ────────────────────────────────────────────────────────────────────────────
# Scorer registry
# Each entry defines the scoring callable and what data it needs pre-fetched.
# ────────────────────────────────────────────────────────────────────────────

ScorerFn = Callable[[TickerData], dict]

SCORER_REGISTRY: dict[str, dict] = {
    "fundamentals": {
        "fn": score_fundamentals,
        "label": "Fundmntls",
        "description": "Profitability, growth, financial health, and valuation ratios",
        "needs_prices": False,
        "needs_news": False,
        "needs_insider": False,
        "style": "quality",
    },
    "growth": {
        "fn": score_growth,
        "label": "Growth",
        "description": "Revenue/EPS growth trends, margin expansion, insider conviction",
        "needs_prices": False,
        "needs_news": False,
        "needs_insider": True,
        "style": "growth",
    },
    "valuation": {
        "fn": score_valuation,
        "label": "Valuation",
        "description": "FCF yield, P/E, EV/EBITDA, P/B, and PEG ratio scoring",
        "needs_prices": False,
        "needs_news": False,
        "needs_insider": False,
        "style": "value",
    },
    "technical": {
        "fn": score_technical,
        "label": "Technical",
        "description": "Trend, momentum, mean-reversion, and volatility signals",
        "needs_prices": True,
        "needs_news": False,
        "needs_insider": False,
        "style": "technical",
    },
    "sentiment": {
        "fn": score_sentiment,
        "label": "Sentiment",
        "description": "Pre-tagged news sentiment + insider trade direction (no LLM)",
        "needs_prices": False,
        "needs_news": True,
        "needs_insider": True,
        "style": "sentiment",
    },
    "warren_buffett": {
        "fn": score_warren_buffett,
        "label": "Buffett",
        "description": "Buffett: fundamentals, moat, management quality, intrinsic value",
        "needs_prices": False,
        "needs_news": False,
        "needs_insider": False,
        "style": "value",
    },
    "ben_graham": {
        "fn": score_ben_graham,
        "label": "Graham",
        "description": "Graham: earnings stability, balance-sheet strength, Graham Number",
        "needs_prices": False,
        "needs_news": False,
        "needs_insider": False,
        "style": "deep_value",
    },
    "cathie_wood": {
        "fn": score_cathie_wood,
        "label": "CathieWood",
        "description": "Cathie Wood: disruptive potential, innovation, high-growth valuation",
        "needs_prices": False,
        "needs_news": False,
        "needs_insider": False,
        "style": "growth",
    },
    "bill_ackman": {
        "fn": score_bill_ackman,
        "label": "Ackman",
        "description": "Ackman: business quality, financial discipline, activism potential",
        "needs_prices": False,
        "needs_news": False,
        "needs_insider": False,
        "style": "activist",
    },
    "michael_burry": {
        "fn": score_michael_burry,
        "label": "MBurry",
        "description": "Burry: deep value, balance sheet stress, contrarian sentiment",
        "needs_prices": False,
        "needs_news": True,
        "needs_insider": True,
        "style": "contrarian",
    },
}

# Default analyst selection used when none is specified
DEFAULT_ANALYSTS: list[str] = [
    "fundamentals",
    "growth",
    "valuation",
    "sentiment",
    "warren_buffett",
]


# ────────────────────────────────────────────────────────────────────────────
# Result types
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    ticker: str
    composite_score: float              # 0 – 100
    signal: str                         # "bullish" | "neutral" | "bearish"
    analyst_scores: dict[str, dict]     # key → scorer output dict
    error: Optional[str] = None

    @property
    def bullish_count(self) -> int:
        return sum(
            1 for v in self.analyst_scores.values() if v.get("signal") == "bullish"
        )

    @property
    def bearish_count(self) -> int:
        return sum(
            1 for v in self.analyst_scores.values() if v.get("signal") == "bearish"
        )


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _score_one_ticker(
    td: TickerData,
    scorer_keys: list[str],
    weights: Optional[dict[str, float]],
) -> ScanResult:
    """Score a single TickerData object with all requested scorers."""
    if td.error:
        return ScanResult(
            ticker=td.ticker,
            composite_score=0.0,
            signal="neutral",
            analyst_scores={},
            error=td.error,
        )

    analyst_scores: dict[str, dict] = {}
    for key in scorer_keys:
        cfg = SCORER_REGISTRY.get(key)
        if cfg is None:
            continue
        fn: ScorerFn = cfg["fn"]
        try:
            analyst_scores[key] = fn(td)
        except Exception as exc:  # pragma: no cover
            analyst_scores[key] = {
                "score": 0.5,
                "signal": "neutral",
                "confidence": 0,
                "details": {"error": str(exc)},
            }

    if not analyst_scores:
        return ScanResult(
            ticker=td.ticker,
            composite_score=50.0,
            signal="neutral",
            analyst_scores={},
        )

    # Weighted composite
    if weights:
        total_weight = sum(weights.get(k, 1.0) for k in analyst_scores)
        if total_weight > 0:
            composite = (
                sum(analyst_scores[k]["score"] * weights.get(k, 1.0) for k in analyst_scores)
                / total_weight
            )
        else:
            composite = 0.5
    else:
        scores = [v["score"] for v in analyst_scores.values()]
        composite = sum(scores) / len(scores)

    composite_pct = round(composite * 100, 1)
    signal = (
        "bullish" if composite >= 0.60
        else "bearish" if composite <= 0.40
        else "neutral"
    )

    return ScanResult(
        ticker=td.ticker,
        composite_score=composite_pct,
        signal=signal,
        analyst_scores=analyst_scores,
    )


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────

def run_scan(
    tickers: list[str],
    end_date: str,
    start_date: str,
    analyst_keys: Optional[list[str]] = None,
    api_key: Optional[str] = None,
    weights: Optional[dict[str, float]] = None,
    top_n: Optional[int] = None,
    min_score: float = 0.0,
    workers: int = 4,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> list[ScanResult]:
    """
    Run a cost-efficient, LLM-free scan over *tickers*.

    Parameters
    ----------
    tickers
        Ticker symbols to scan.  Duplicates are deduplicated.
    end_date
        Analysis end date (YYYY-MM-DD).
    start_date
        Look-back start date used for price / technical analysis.
    analyst_keys
        Subset of SCORER_REGISTRY keys.  Defaults to DEFAULT_ANALYSTS.
    api_key
        Financial data API key (falls back to env var).
    weights
        Per-analyst weight overrides for composite score.
    top_n
        Retain only the top-N ranked results.
    min_score
        Exclude results below this composite score (0–100).
    workers
        Thread-pool size for parallel data fetching.
    progress_callback
        Optional callable(ticker, status) for progress updates.

    Returns
    -------
    list[ScanResult]
        Sorted descending by composite_score.
    """
    keys = analyst_keys or DEFAULT_ANALYSTS
    unique_tickers = list(dict.fromkeys(t.upper() for t in tickers))

    # Determine which data categories to fetch
    fetch_prices = any(SCORER_REGISTRY.get(k, {}).get("needs_prices") for k in keys)
    fetch_news = any(SCORER_REGISTRY.get(k, {}).get("needs_news") for k in keys)
    fetch_insider = any(SCORER_REGISTRY.get(k, {}).get("needs_insider") for k in keys)

    # ── Parallel data fetch ───────────────────────────────────────────────
    def _fetch(ticker: str) -> TickerData:
        if progress_callback:
            progress_callback(ticker, "fetching")
        td = fetch_ticker_data(
            ticker=ticker,
            end_date=end_date,
            start_date=start_date,
            api_key=api_key,
            fetch_prices=fetch_prices,
            fetch_news=fetch_news,
            fetch_insider=fetch_insider,
        )
        if progress_callback:
            progress_callback(ticker, "scoring")
        return td

    ticker_data_list: list[TickerData] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, t): t for t in unique_tickers}
        for future in concurrent.futures.as_completed(futures):
            try:
                ticker_data_list.append(future.result())
            except Exception as exc:  # pragma: no cover
                t = futures[future]
                ticker_data_list.append(
                    TickerData(ticker=t, end_date=end_date, start_date=start_date, error=str(exc))
                )

    # ── Score each ticker ─────────────────────────────────────────────────
    results = [_score_one_ticker(td, keys, weights) for td in ticker_data_list]

    # ── Filter & rank ─────────────────────────────────────────────────────
    results = [r for r in results if r.composite_score >= min_score]
    results.sort(key=lambda r: r.composite_score, reverse=True)

    if top_n is not None:
        results = results[:top_n]

    if progress_callback:
        for r in results:
            progress_callback(r.ticker, "done")

    return results
