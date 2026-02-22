"""Unit tests for the YFinanceProvider.

All yfinance network calls are mocked so the tests run offline.
"""

import math
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from src.tools.providers.yfinance_provider import YFinanceProvider, _safe_float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_df(dates, opens, closes, highs, lows, volumes):
    index = pd.DatetimeIndex(dates)
    return pd.DataFrame({"Open": opens, "Close": closes, "High": highs, "Low": lows, "Volume": volumes}, index=index)


def _make_stmt(rows: dict[str, list], columns: list[str]) -> pd.DataFrame:
    """Build a financial-statement DataFrame (rows = metrics, columns = dates)."""
    cols = pd.DatetimeIndex(columns)
    return pd.DataFrame(rows, index=list(rows.keys())).T.reindex(list(rows.keys())).T.set_axis(cols, axis=1)


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_inf_returns_none(self):
        assert _safe_float(float("inf")) is None

    def test_valid_int(self):
        assert _safe_float(42) == 42.0

    def test_valid_string(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_invalid_string(self):
        assert _safe_float("abc") is None


# ---------------------------------------------------------------------------
# get_prices
# ---------------------------------------------------------------------------


class TestGetPrices:
    def test_returns_price_objects(self):
        provider = YFinanceProvider()
        df = _make_price_df(
            ["2024-01-02", "2024-01-03"],
            [185.0, 186.0],
            [186.0, 187.0],
            [187.0, 188.0],
            [184.0, 185.0],
            [50_000_000, 55_000_000],
        )
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = df

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            prices = provider.get_prices("AAPL", "2024-01-01", "2024-01-05")

        assert len(prices) == 2
        assert all(isinstance(p, Price) for p in prices)
        assert prices[0].open == pytest.approx(185.0)
        assert prices[0].close == pytest.approx(186.0)
        assert prices[0].volume == 50_000_000
        assert prices[0].time == "2024-01-02T00:00:00"

    def test_empty_df_returns_empty_list(self):
        provider = YFinanceProvider()
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            assert provider.get_prices("AAPL", "2024-01-01", "2024-01-05") == []


# ---------------------------------------------------------------------------
# get_financial_metrics
# ---------------------------------------------------------------------------


class TestGetFinancialMetrics:
    def _build_mock_ticker(self, info=None, income_stmt=None, balance_sheet=None, cashflow=None, quarterly_income_stmt=None, quarterly_balance_sheet=None, quarterly_cashflow=None):
        mock_ticker = MagicMock()
        mock_ticker.info = info or {}
        mock_ticker.income_stmt = income_stmt if income_stmt is not None else pd.DataFrame()
        mock_ticker.balance_sheet = balance_sheet if balance_sheet is not None else pd.DataFrame()
        mock_ticker.cashflow = cashflow if cashflow is not None else pd.DataFrame()
        mock_ticker.quarterly_income_stmt = quarterly_income_stmt if quarterly_income_stmt is not None else pd.DataFrame()
        mock_ticker.quarterly_balance_sheet = quarterly_balance_sheet if quarterly_balance_sheet is not None else pd.DataFrame()
        mock_ticker.quarterly_cashflow = quarterly_cashflow if quarterly_cashflow is not None else pd.DataFrame()
        return mock_ticker

    def test_annual_returns_financial_metrics_list(self):
        provider = YFinanceProvider()
        cols = pd.DatetimeIndex(["2023-12-31", "2022-12-31"])
        inc = pd.DataFrame(
            {
                "Total Revenue": [100_000, 90_000],
                "Gross Profit": [40_000, 36_000],
                "Operating Income": [20_000, 18_000],
                "Net Income": [15_000, 13_000],
                "Interest Expense": [-1_000, -900],
            },
            index=cols,
        ).T
        bs = pd.DataFrame(
            {
                "Total Assets": [200_000, 180_000],
                "Stockholders Equity": [100_000, 90_000],
                "Total Debt": [50_000, 45_000],
                "Current Assets": [60_000, 55_000],
                "Current Liabilities": [30_000, 27_000],
                "Cash And Cash Equivalents": [20_000, 18_000],
            },
            index=cols,
        ).T
        cf = pd.DataFrame(
            {
                "Free Cash Flow": [18_000, 16_000],
                "Operating Cash Flow": [22_000, 20_000],
            },
            index=cols,
        ).T
        info = {"marketCap": 500_000, "currency": "USD"}
        mock_ticker = self._build_mock_ticker(info=info, income_stmt=inc, balance_sheet=bs, cashflow=cf)

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            metrics = provider.get_financial_metrics("AAPL", "2024-01-01", period="annual", limit=2)

        assert len(metrics) == 2
        m0 = metrics[0]
        assert isinstance(m0, FinancialMetrics)
        assert m0.ticker == "AAPL"
        assert m0.report_period == "2023-12-31"
        assert m0.period == "annual"
        assert m0.gross_margin == pytest.approx(0.4)
        assert m0.net_margin == pytest.approx(0.15)
        # Revenue growth: (100k - 90k) / 90k
        assert m0.revenue_growth == pytest.approx(1 / 9)
        # Market cap from info only on most recent period
        assert m0.market_cap == pytest.approx(500_000)
        # Older period should not have market cap
        assert metrics[1].market_cap is None

    def test_info_only_fallback_when_no_statements(self):
        provider = YFinanceProvider()
        info = {
            "marketCap": 1_000_000,
            "trailingPE": 25.0,
            "grossMargins": 0.40,
            "currency": "USD",
        }
        mock_ticker = self._build_mock_ticker(info=info)

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            metrics = provider.get_financial_metrics("AAPL", "2024-01-01", period="annual")

        assert len(metrics) == 1
        m = metrics[0]
        assert m.market_cap == pytest.approx(1_000_000)
        assert m.price_to_earnings_ratio == pytest.approx(25.0)
        assert m.gross_margin == pytest.approx(0.40)

    def test_ttm_sums_quarterly(self):
        """TTM should sum flow-statement items across up to 4 quarters."""
        provider = YFinanceProvider()
        q_cols = pd.DatetimeIndex(["2023-12-31", "2023-09-30", "2023-06-30", "2023-03-31"])
        q_inc = pd.DataFrame(
            {
                "Total Revenue": [25_000, 24_000, 26_000, 27_000],
                "Net Income": [4_000, 3_800, 4_200, 4_100],
            },
            index=q_cols,
        ).T
        q_bs = pd.DataFrame(
            {
                "Total Assets": [200_000, 195_000, 190_000, 185_000],
                "Stockholders Equity": [100_000, 98_000, 97_000, 96_000],
            },
            index=q_cols,
        ).T
        q_cf = pd.DataFrame({"Free Cash Flow": [5_000, 4_500, 5_500, 5_200]}, index=q_cols).T
        info = {"currency": "USD"}
        mock_ticker = self._build_mock_ticker(info=info, quarterly_income_stmt=q_inc, quarterly_balance_sheet=q_bs, quarterly_cashflow=q_cf)

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            metrics = provider.get_financial_metrics("AAPL", "2024-01-01", period="ttm", limit=1)

        assert len(metrics) == 1
        m = metrics[0]
        # TTM revenue = sum of all 4 quarters
        assert m.net_margin == pytest.approx((4_000 + 3_800 + 4_200 + 4_100) / (25_000 + 24_000 + 26_000 + 27_000))


# ---------------------------------------------------------------------------
# search_line_items
# ---------------------------------------------------------------------------


class TestSearchLineItems:
    def test_returns_requested_fields(self):
        provider = YFinanceProvider()
        cols = pd.DatetimeIndex(["2023-12-31", "2022-12-31"])
        inc = pd.DataFrame({"Total Revenue": [100_000, 90_000], "Net Income": [15_000, 13_000]}, index=cols).T
        bs = pd.DataFrame({"Total Assets": [200_000, 180_000], "Total Debt": [50_000, 45_000]}, index=cols).T
        cf = pd.DataFrame({"Free Cash Flow": [18_000, 16_000]}, index=cols).T
        info = {"currency": "USD"}
        mock_ticker = MagicMock()
        mock_ticker.info = info
        mock_ticker.income_stmt = inc
        mock_ticker.balance_sheet = bs
        mock_ticker.cashflow = cf
        mock_ticker.quarterly_income_stmt = pd.DataFrame()
        mock_ticker.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker.quarterly_cashflow = pd.DataFrame()

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            results = provider.search_line_items("AAPL", ["revenue", "net_income", "free_cash_flow", "total_assets"], "2024-01-01", period="annual", limit=2)

        assert len(results) == 2
        assert all(isinstance(r, LineItem) for r in results)
        r0 = results[0]
        assert r0.ticker == "AAPL"
        assert r0.revenue == pytest.approx(100_000)
        assert r0.net_income == pytest.approx(15_000)
        assert r0.free_cash_flow == pytest.approx(18_000)
        assert r0.total_assets == pytest.approx(200_000)

    def test_unknown_field_returns_none(self):
        provider = YFinanceProvider()
        cols = pd.DatetimeIndex(["2023-12-31"])
        inc = pd.DataFrame({"Total Revenue": [100_000]}, index=cols).T
        bs = pd.DataFrame({"Total Assets": [200_000]}, index=cols).T
        cf = pd.DataFrame({"Free Cash Flow": [18_000]}, index=cols).T
        mock_ticker = MagicMock()
        mock_ticker.info = {"currency": "USD"}
        mock_ticker.income_stmt = inc
        mock_ticker.balance_sheet = bs
        mock_ticker.cashflow = cf
        mock_ticker.quarterly_income_stmt = pd.DataFrame()
        mock_ticker.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker.quarterly_cashflow = pd.DataFrame()

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            results = provider.search_line_items("AAPL", ["nonexistent_field"], "2024-01-01", period="annual", limit=1)

        assert len(results) == 1
        assert getattr(results[0], "nonexistent_field", None) is None

    def test_end_date_filters_future_periods(self):
        """Periods after end_date must be excluded."""
        provider = YFinanceProvider()
        cols = pd.DatetimeIndex(["2024-12-31", "2023-12-31"])
        inc = pd.DataFrame({"Total Revenue": [120_000, 100_000]}, index=cols).T
        bs = pd.DataFrame({"Total Assets": [220_000, 200_000]}, index=cols).T
        cf = pd.DataFrame({"Free Cash Flow": [20_000, 18_000]}, index=cols).T
        mock_ticker = MagicMock()
        mock_ticker.info = {"currency": "USD"}
        mock_ticker.income_stmt = inc
        mock_ticker.balance_sheet = bs
        mock_ticker.cashflow = cf
        mock_ticker.quarterly_income_stmt = pd.DataFrame()
        mock_ticker.quarterly_balance_sheet = pd.DataFrame()
        mock_ticker.quarterly_cashflow = pd.DataFrame()

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            results = provider.search_line_items("AAPL", ["revenue"], "2024-06-01", period="annual", limit=5)

        # 2024-12-31 is after end_date; only 2023-12-31 should be returned
        assert len(results) == 1
        assert results[0].report_period == "2023-12-31"


# ---------------------------------------------------------------------------
# get_insider_trades
# ---------------------------------------------------------------------------


class TestGetInsiderTrades:
    def test_returns_insider_trade_objects(self):
        provider = YFinanceProvider()
        df = pd.DataFrame(
            {
                "Insider": ["Tim Cook", "Jeff Williams"],
                "Position": ["CEO", "COO"],
                "Shares": [10_000, 5_000],
                "Value": [2_000_000, 1_000_000],
                "Start Date": ["2024-01-15", "2024-01-20"],
            }
        )
        mock_ticker = MagicMock()
        mock_ticker.insider_transactions = df

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            trades = provider.get_insider_trades("AAPL", "2024-06-01")

        assert len(trades) == 2
        assert all(isinstance(t, InsiderTrade) for t in trades)
        assert trades[0].name == "Tim Cook"
        assert trades[0].title == "CEO"
        assert trades[0].transaction_shares == pytest.approx(10_000)
        assert trades[0].transaction_value == pytest.approx(2_000_000)

    def test_filters_by_end_date(self):
        provider = YFinanceProvider()
        df = pd.DataFrame(
            {
                "Insider": ["Tim Cook", "Jeff Williams"],
                "Start Date": ["2024-01-15", "2025-01-20"],
                "Shares": [10_000, 5_000],
                "Value": [2_000_000, 1_000_000],
            }
        )
        mock_ticker = MagicMock()
        mock_ticker.insider_transactions = df

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            trades = provider.get_insider_trades("AAPL", "2024-12-31")

        # Only the 2024-01-15 trade should pass the end_date filter
        assert len(trades) == 1
        assert trades[0].name == "Tim Cook"

    def test_empty_df_returns_empty_list(self):
        provider = YFinanceProvider()
        mock_ticker = MagicMock()
        mock_ticker.insider_transactions = pd.DataFrame()

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            assert provider.get_insider_trades("AAPL", "2024-01-01") == []

    def test_exception_returns_empty_list(self):
        provider = YFinanceProvider()
        mock_ticker = MagicMock()
        mock_ticker.insider_transactions = None

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            assert provider.get_insider_trades("AAPL", "2024-01-01") == []


# ---------------------------------------------------------------------------
# get_company_news
# ---------------------------------------------------------------------------


class TestGetCompanyNews:
    def _news_item(self, title, pub_date_unix, source="Reuters", url="https://example.com"):
        return {
            "content": {
                "title": title,
                "pubDate": pub_date_unix,
                "provider": {"displayName": source},
                "authors": [{"name": "Jane Doe"}],
                "canonicalUrl": {"url": url},
            }
        }

    def test_returns_company_news_objects(self):
        provider = YFinanceProvider()
        # Unix timestamp for 2024-01-10
        ts = int(datetime(2024, 1, 10).timestamp())
        mock_ticker = MagicMock()
        mock_ticker.news = [self._news_item("Apple Q1 Results", ts)]

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            news = provider.get_company_news("AAPL", "2024-06-01")

        assert len(news) == 1
        assert isinstance(news[0], CompanyNews)
        assert news[0].title == "Apple Q1 Results"
        assert news[0].source == "Reuters"
        assert news[0].author == "Jane Doe"
        assert news[0].sentiment is None  # yfinance provides no sentiment

    def test_filters_by_end_date(self):
        provider = YFinanceProvider()
        ts_past = int(datetime(2023, 6, 1).timestamp())
        ts_future = int(datetime(2025, 1, 1).timestamp())
        mock_ticker = MagicMock()
        mock_ticker.news = [
            self._news_item("Old News", ts_past),
            self._news_item("Future News", ts_future),
        ]

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            news = provider.get_company_news("AAPL", "2024-01-01")

        assert len(news) == 1
        assert news[0].title == "Old News"

    def test_empty_news_returns_empty_list(self):
        provider = YFinanceProvider()
        mock_ticker = MagicMock()
        mock_ticker.news = []

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            assert provider.get_company_news("AAPL", "2024-01-01") == []


# ---------------------------------------------------------------------------
# get_market_cap
# ---------------------------------------------------------------------------


class TestGetMarketCap:
    def test_returns_market_cap_from_info(self):
        provider = YFinanceProvider()
        mock_ticker = MagicMock()
        mock_ticker.info = {"marketCap": 2_500_000_000_000}

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            result = provider.get_market_cap("AAPL", "2024-01-01")

        assert result == pytest.approx(2_500_000_000_000)

    def test_returns_none_when_missing(self):
        provider = YFinanceProvider()
        mock_ticker = MagicMock()
        mock_ticker.info = {}

        with patch("src.tools.providers.yfinance_provider.yf.Ticker", return_value=mock_ticker):
            assert provider.get_market_cap("AAPL", "2024-01-01") is None


# ---------------------------------------------------------------------------
# Provider delegation in api.py
# ---------------------------------------------------------------------------


class TestApiProviderDelegation:
    """Verify that api.py delegates to YFinanceProvider when _provider is set."""

    def test_get_prices_uses_provider(self):
        import src.tools.api as api_module
        from src.tools.providers.yfinance_provider import YFinanceProvider

        mock_provider = MagicMock(spec=YFinanceProvider)
        mock_provider.get_prices.return_value = [
            Price(open=100.0, close=101.0, high=102.0, low=99.0, volume=1000, time="2024-01-02T00:00:00")
        ]

        original_provider = api_module._provider
        original_cache = api_module._cache
        try:
            api_module._provider = mock_provider
            # Use a fresh cache mock so no stale data interferes
            mock_cache = MagicMock()
            mock_cache.get_prices.return_value = None
            api_module._cache = mock_cache

            result = api_module.get_prices("AAPL", "2024-01-01", "2024-01-05")
        finally:
            api_module._provider = original_provider
            api_module._cache = original_cache

        mock_provider.get_prices.assert_called_once_with("AAPL", "2024-01-01", "2024-01-05")
        assert len(result) == 1
        assert result[0].open == pytest.approx(100.0)

    def test_get_financial_metrics_uses_provider(self):
        import src.tools.api as api_module
        from src.tools.providers.yfinance_provider import YFinanceProvider

        mock_provider = MagicMock(spec=YFinanceProvider)
        mock_provider.get_financial_metrics.return_value = [
            FinancialMetrics(
                ticker="AAPL",
                report_period="2023-12-31",
                period="ttm",
                currency="USD",
                market_cap=3e12,
                enterprise_value=None,
                price_to_earnings_ratio=None,
                price_to_book_ratio=None,
                price_to_sales_ratio=None,
                enterprise_value_to_ebitda_ratio=None,
                enterprise_value_to_revenue_ratio=None,
                free_cash_flow_yield=None,
                peg_ratio=None,
                gross_margin=0.4,
                operating_margin=0.3,
                net_margin=0.25,
                return_on_equity=None,
                return_on_assets=None,
                return_on_invested_capital=None,
                asset_turnover=None,
                inventory_turnover=None,
                receivables_turnover=None,
                days_sales_outstanding=None,
                operating_cycle=None,
                working_capital_turnover=None,
                current_ratio=None,
                quick_ratio=None,
                cash_ratio=None,
                operating_cash_flow_ratio=None,
                debt_to_equity=None,
                debt_to_assets=None,
                interest_coverage=None,
                revenue_growth=None,
                earnings_growth=None,
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=None,
                earnings_per_share=None,
                book_value_per_share=None,
                free_cash_flow_per_share=None,
            )
        ]

        original_provider = api_module._provider
        original_cache = api_module._cache
        try:
            api_module._provider = mock_provider
            mock_cache = MagicMock()
            mock_cache.get_financial_metrics.return_value = None
            api_module._cache = mock_cache

            result = api_module.get_financial_metrics("AAPL", "2024-01-01", period="ttm", limit=1)
        finally:
            api_module._provider = original_provider
            api_module._cache = original_cache

        mock_provider.get_financial_metrics.assert_called_once_with("AAPL", "2024-01-01", "ttm", 1)
        assert len(result) == 1
        assert result[0].gross_margin == pytest.approx(0.4)
