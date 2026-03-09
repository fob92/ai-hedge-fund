"""
Shared data structures and single-pass data fetching for the scanner.

Each ticker's raw data is fetched once and stored in a TickerData object,
which is then passed to all scorers.  Since api.py uses an in-process cache
(keyed on ticker + period + date + limit) every subsequent call for the same
data is free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)
from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    get_prices,
    prices_to_df,
    search_line_items,
)

# ---------------------------------------------------------------------------
# Comprehensive line-item lists – union of all analyst needs
# TTM period (for Buffett, Burry, Valuation, Growth …)
# ---------------------------------------------------------------------------
TTM_LINE_ITEMS: list[str] = [
    "capital_expenditure",
    "depreciation_and_amortization",
    "net_income",
    "outstanding_shares",
    "total_assets",
    "total_liabilities",
    "shareholders_equity",
    "dividends_and_other_cash_distributions",
    "issuance_or_purchase_of_equity_shares",
    "gross_profit",
    "revenue",
    "free_cash_flow",
    "total_debt",
    "cash_and_equivalents",
    "working_capital",
    "interest_expense",
    "operating_income",
    "ebit",
    "ebitda",
]

# Annual period (for Graham, Cathie Wood, Ackman …)
ANNUAL_LINE_ITEMS: list[str] = [
    "earnings_per_share",
    "revenue",
    "net_income",
    "book_value_per_share",
    "total_assets",
    "total_liabilities",
    "current_assets",
    "current_liabilities",
    "dividends_and_other_cash_distributions",
    "outstanding_shares",
    "gross_profit",
    "free_cash_flow",
    "capital_expenditure",
    "research_and_development",
    "operating_expense",
    "issuance_or_purchase_of_equity_shares",
    "shareholders_equity",
    "depreciation_and_amortization",
]


@dataclass
class TickerData:
    """All raw data needed for a single ticker's scan."""

    ticker: str
    end_date: str
    start_date: str

    # Financial metrics
    ttm_metrics: list[FinancialMetrics] = field(default_factory=list)
    annual_metrics: list[FinancialMetrics] = field(default_factory=list)

    # Line items
    ttm_line_items: list[LineItem] = field(default_factory=list)
    annual_line_items: list[LineItem] = field(default_factory=list)

    # Supplementary data
    market_cap: Optional[float] = None
    insider_trades: list[InsiderTrade] = field(default_factory=list)
    news: list[CompanyNews] = field(default_factory=list)
    prices_df: Optional[pd.DataFrame] = None

    # Error state
    error: Optional[str] = None


def fetch_ticker_data(
    ticker: str,
    end_date: str,
    start_date: str,
    api_key: Optional[str] = None,
    fetch_prices: bool = True,
    fetch_news: bool = True,
    fetch_insider: bool = True,
) -> TickerData:
    """
    Fetch all data needed for scanning a single ticker.

    This is a **blocking** call intended to be run in a thread pool so that
    multiple tickers are fetched concurrently.
    """
    td = TickerData(ticker=ticker, end_date=end_date, start_date=start_date)

    try:
        # ── Financial metrics ──────────────────────────────────────────────
        td.ttm_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=12,
            api_key=api_key,
        )
        td.annual_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="annual",
            limit=10,
            api_key=api_key,
        )

        # ── Line items ─────────────────────────────────────────────────────
        td.ttm_line_items = search_line_items(
            ticker=ticker,
            line_items=TTM_LINE_ITEMS,
            end_date=end_date,
            period="ttm",
            limit=10,
            api_key=api_key,
        )
        td.annual_line_items = search_line_items(
            ticker=ticker,
            line_items=ANNUAL_LINE_ITEMS,
            end_date=end_date,
            period="annual",
            limit=10,
            api_key=api_key,
        )

        # ── Market cap ────────────────────────────────────────────────────
        td.market_cap = get_market_cap(
            ticker=ticker, end_date=end_date, api_key=api_key
        )

        # ── Insider trades ────────────────────────────────────────────────
        if fetch_insider:
            td.insider_trades = get_insider_trades(
                ticker=ticker,
                end_date=end_date,
                limit=500,
                api_key=api_key,
            )

        # ── Company news (last 90 days) ────────────────────────────────────
        if fetch_news:
            news_start = (
                datetime.fromisoformat(end_date) - timedelta(days=90)
            ).date().isoformat()
            td.news = get_company_news(
                ticker=ticker,
                end_date=end_date,
                start_date=news_start,
                limit=50,
                api_key=api_key,
            )

        # ── Price data ────────────────────────────────────────────────────
        if fetch_prices:
            prices = get_prices(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                api_key=api_key,
            )
            if prices:
                td.prices_df = prices_to_df(prices)

    except Exception as exc:  # pragma: no cover
        td.error = str(exc)

    return td
