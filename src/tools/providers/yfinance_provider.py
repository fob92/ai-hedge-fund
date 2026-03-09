"""yfinance-based implementation of FinancialDataProvider.

All six API surface functions are mapped to yfinance equivalents.  The
conversion to the existing Pydantic models is done in-process so that
every downstream agent and the backtesting engine continue to work
without modification.

Key decisions
-------------
* ``period="ttm"``   – computed by summing the last ≤4 quarters of the
                       quarterly income-statement / cash-flow statement, and
                       using the most-recent quarter for balance-sheet items.
* ``period="annual"``– maps directly to ``yf.Ticker.income_stmt`` (annual).
* ``period="quarterly"`` – maps to ``yf.Ticker.quarterly_income_stmt``.
* Market-cap / valuation ratios (PE, PB, EV …) are taken from
  ``yf.Ticker.info`` and are therefore current-price based.  They are only
  populated on the *most recent* historical period entry.
* ``CompanyNews.sentiment`` is always ``None`` because yfinance does not
  provide sentiment scores.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price
from src.tools.providers.base import FinancialDataProvider


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _safe_float(value) -> Optional[float]:
    """Convert *value* to ``float``; return ``None`` on failure or NaN/Inf."""
    if value is None:
        return None
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


class YFinanceProvider(FinancialDataProvider):
    """Financial data provider backed by the ``yfinance`` library."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_row(self, series: pd.Series, *keys: str) -> Optional[float]:
        """Return the first matching value from *series*, or ``None``."""
        for key in keys:
            try:
                if key in series.index:
                    return _safe_float(series[key])
            except Exception:
                pass
        return None

    def _filter_df(self, df: pd.DataFrame, end_date: str) -> pd.DataFrame:
        """Return columns whose date is on or before *end_date*."""
        if df is None or df.empty:
            return pd.DataFrame()
        end_ts = pd.Timestamp(end_date)
        cols = [c for c in df.columns if pd.Timestamp(c) <= end_ts]
        return df[cols] if cols else pd.DataFrame()

    def _get_stmt(self, t: yf.Ticker, period: str):
        """Return (income_stmt, balance_sheet, cashflow) for *period*."""
        if period == "quarterly":
            return t.quarterly_income_stmt, t.quarterly_balance_sheet, t.quarterly_cashflow
        return t.income_stmt, t.balance_sheet, t.cashflow

    def _compute_ebitda(self, inc: pd.Series, cf: pd.Series) -> Optional[float]:
        ebit = self._get_row(inc, "EBIT", "Operating Income")
        da = self._get_row(cf, "Depreciation And Amortization", "Depreciation Amortization Depletion")
        if ebit is not None and da is not None:
            return ebit + da
        return None

    def _compute_working_capital(self, bs: pd.Series) -> Optional[float]:
        ca = self._get_row(bs, "Current Assets")
        cl = self._get_row(bs, "Current Liabilities")
        if ca is not None and cl is not None:
            return ca - cl
        return None

    def _compute_bvps(self, bs: pd.Series) -> Optional[float]:
        equity = self._get_row(bs, "Stockholders Equity", "Total Equity Gross Minority Interest")
        shares = self._get_row(bs, "Ordinary Shares Number", "Share Issued")
        if equity is not None and shares and shares > 0:
            return equity / shares
        return None

    def _compute_margin(self, inc: pd.Series, numerator_key: str) -> Optional[float]:
        num = self._get_row(inc, numerator_key)
        rev = self._get_row(inc, "Total Revenue", "Revenue")
        if num is not None and rev is not None and rev != 0:
            return num / rev
        return None

    # ------------------------------------------------------------------
    # Core builder — converts statement rows + info dict → FinancialMetrics
    # ------------------------------------------------------------------

    def _build_metrics(
        self,
        ticker: str,
        report_period: str,
        period: str,
        info: dict,
        inc: pd.Series,
        bs: pd.Series,
        cf: pd.Series,
        prev_inc: Optional[pd.Series] = None,
        prev_bs: Optional[pd.Series] = None,
    ) -> FinancialMetrics:
        g = self._get_row

        # --- income statement ---
        revenue = g(inc, "Total Revenue", "Revenue")
        gross_profit = g(inc, "Gross Profit")
        operating_income = g(inc, "Operating Income", "EBIT")
        net_income = g(inc, "Net Income")
        ebitda = g(inc, "EBITDA") or self._compute_ebitda(inc, cf)
        interest_expense = g(inc, "Interest Expense")
        eps = g(inc, "Basic EPS", "Diluted EPS")

        # --- balance sheet ---
        total_assets = g(bs, "Total Assets")
        total_liabilities = g(bs, "Total Liabilities Net Minority Interest", "Total Liabilities")
        total_equity = g(bs, "Stockholders Equity", "Total Equity Gross Minority Interest")
        total_debt = g(bs, "Total Debt")
        current_assets = g(bs, "Current Assets")
        current_liabilities = g(bs, "Current Liabilities")
        cash = g(bs, "Cash And Cash Equivalents", "Cash")
        shares_outstanding = g(bs, "Ordinary Shares Number", "Share Issued")
        inventory = g(bs, "Inventory")
        accounts_receivable = g(bs, "Accounts Receivable", "Net Receivables")

        # --- cash flow ---
        fcf = g(cf, "Free Cash Flow")
        operating_cf = g(cf, "Operating Cash Flow", "Cash From Operations")

        # --- previous period for growth rates ---
        prev_revenue = g(prev_inc, "Total Revenue", "Revenue") if prev_inc is not None else None
        prev_net_income = g(prev_inc, "Net Income") if prev_inc is not None else None
        prev_equity = g(prev_bs, "Stockholders Equity", "Total Equity Gross Minority Interest") if prev_bs is not None else None

        # --- margins ---
        gross_margin = (gross_profit / revenue if (gross_profit is not None and revenue is not None and revenue != 0) else _safe_float(info.get("grossMargins")))
        operating_margin = (operating_income / revenue if (operating_income is not None and revenue is not None and revenue != 0) else _safe_float(info.get("operatingMargins")))
        net_margin = (net_income / revenue if (net_income is not None and revenue is not None and revenue != 0) else _safe_float(info.get("profitMargins")))

        # --- returns ---
        roe = (net_income / total_equity if (net_income is not None and total_equity is not None and total_equity != 0) else _safe_float(info.get("returnOnEquity")))
        roa = (net_income / total_assets if (net_income is not None and total_assets is not None and total_assets != 0) else _safe_float(info.get("returnOnAssets")))

        invested_capital = None
        if total_equity is not None and total_debt is not None:
            invested_capital = total_equity + total_debt - (cash or 0)
        roic = (net_income / invested_capital if (net_income is not None and invested_capital is not None and invested_capital != 0) else None)

        # --- liquidity ---
        current_ratio = (current_assets / current_liabilities if (current_assets is not None and current_liabilities is not None and current_liabilities != 0) else _safe_float(info.get("currentRatio")))
        quick_ratio = _safe_float(info.get("quickRatio"))
        cash_ratio = (cash / current_liabilities if (cash is not None and current_liabilities is not None and current_liabilities != 0) else None)
        ocf_ratio = (operating_cf / current_liabilities if (operating_cf is not None and current_liabilities is not None and current_liabilities != 0) else None)

        # --- leverage ---
        debt_to_equity = None
        if total_debt is not None and total_equity is not None and total_equity != 0:
            debt_to_equity = total_debt / total_equity
        elif info.get("debtToEquity") is not None:
            raw = _safe_float(info.get("debtToEquity"))
            debt_to_equity = raw / 100.0 if raw is not None else None

        debt_to_assets = (total_debt / total_assets if (total_debt is not None and total_assets is not None and total_assets != 0) else None)
        interest_coverage = None
        if operating_income is not None and interest_expense and interest_expense != 0:
            interest_coverage = operating_income / abs(interest_expense)

        # --- efficiency ---
        asset_turnover = (revenue / total_assets if (revenue is not None and total_assets is not None and total_assets != 0) else None)
        inventory_turnover = (revenue / inventory if (revenue is not None and inventory is not None and inventory != 0) else None)
        receivables_turnover = (revenue / accounts_receivable if (revenue is not None and accounts_receivable is not None and accounts_receivable != 0) else None)
        days_sales_outstanding = (365.0 / receivables_turnover if (receivables_turnover is not None and receivables_turnover != 0) else None)
        working_capital = self._compute_working_capital(bs)
        working_capital_turnover = (revenue / working_capital if (revenue is not None and working_capital is not None and working_capital != 0) else None)
        days_inventory = (365.0 / inventory_turnover if (inventory_turnover is not None and inventory_turnover != 0) else None)
        operating_cycle = ((days_sales_outstanding + days_inventory) if (days_sales_outstanding is not None and days_inventory is not None) else None)

        # --- growth ---
        revenue_growth = None
        if revenue is not None and prev_revenue is not None and prev_revenue != 0:
            revenue_growth = (revenue - prev_revenue) / abs(prev_revenue)
        elif info.get("revenueGrowth") is not None:
            revenue_growth = _safe_float(info.get("revenueGrowth"))

        earnings_growth = None
        if net_income is not None and prev_net_income is not None and prev_net_income != 0:
            earnings_growth = (net_income - prev_net_income) / abs(prev_net_income)
        elif info.get("earningsGrowth") is not None:
            earnings_growth = _safe_float(info.get("earningsGrowth"))

        # --- per-share ---
        bvps = self._compute_bvps(bs) or _safe_float(info.get("bookValue"))
        if eps is None:
            eps = _safe_float(info.get("trailingEps"))
        fcf_per_share = None
        if fcf is not None and shares_outstanding is not None and shares_outstanding != 0:
            fcf_per_share = fcf / shares_outstanding

        # --- market / valuation (current price based; only meaningful for the most recent period) ---
        market_cap = _safe_float(info.get("marketCap"))
        enterprise_value = _safe_float(info.get("enterpriseValue"))
        pe_ratio = _safe_float(info.get("trailingPE"))
        pb_ratio = _safe_float(info.get("priceToBook"))
        ps_ratio = _safe_float(info.get("priceToSalesTrailing12Months"))
        ev_ebitda = _safe_float(info.get("enterpriseToEbitda"))
        ev_revenue = _safe_float(info.get("enterpriseToRevenue"))
        peg_ratio = _safe_float(info.get("pegRatio"))
        payout_ratio = _safe_float(info.get("payoutRatio"))
        fcf_yield = (fcf / market_cap if (fcf is not None and market_cap is not None and market_cap != 0) else None)

        currency = info.get("currency", "USD") or "USD"

        return FinancialMetrics(
            ticker=ticker,
            report_period=report_period,
            period=period,
            currency=currency,
            market_cap=market_cap,
            enterprise_value=enterprise_value,
            price_to_earnings_ratio=pe_ratio,
            price_to_book_ratio=pb_ratio,
            price_to_sales_ratio=ps_ratio,
            enterprise_value_to_ebitda_ratio=ev_ebitda,
            enterprise_value_to_revenue_ratio=ev_revenue,
            free_cash_flow_yield=fcf_yield,
            peg_ratio=peg_ratio,
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            net_margin=net_margin,
            return_on_equity=roe,
            return_on_assets=roa,
            return_on_invested_capital=roic,
            asset_turnover=asset_turnover,
            inventory_turnover=inventory_turnover,
            receivables_turnover=receivables_turnover,
            days_sales_outstanding=days_sales_outstanding,
            operating_cycle=operating_cycle,
            working_capital_turnover=working_capital_turnover,
            current_ratio=current_ratio,
            quick_ratio=quick_ratio,
            cash_ratio=cash_ratio,
            operating_cash_flow_ratio=ocf_ratio,
            debt_to_equity=debt_to_equity,
            debt_to_assets=debt_to_assets,
            interest_coverage=interest_coverage,
            revenue_growth=revenue_growth,
            earnings_growth=earnings_growth,
            book_value_growth=None,
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=payout_ratio,
            earnings_per_share=eps,
            book_value_per_share=bvps,
            free_cash_flow_per_share=fcf_per_share,
        )

    def _metrics_from_info_only(self, ticker: str, report_period: str, period: str, info: dict) -> FinancialMetrics:
        """Build a FinancialMetrics when no financial statements are available."""
        g = _safe_float
        currency = info.get("currency", "USD") or "USD"
        raw_d2e = g(info.get("debtToEquity"))
        d2e = raw_d2e / 100.0 if raw_d2e is not None else None
        return FinancialMetrics(
            ticker=ticker,
            report_period=report_period,
            period=period,
            currency=currency,
            market_cap=g(info.get("marketCap")),
            enterprise_value=g(info.get("enterpriseValue")),
            price_to_earnings_ratio=g(info.get("trailingPE")),
            price_to_book_ratio=g(info.get("priceToBook")),
            price_to_sales_ratio=g(info.get("priceToSalesTrailing12Months")),
            enterprise_value_to_ebitda_ratio=g(info.get("enterpriseToEbitda")),
            enterprise_value_to_revenue_ratio=g(info.get("enterpriseToRevenue")),
            free_cash_flow_yield=None,
            peg_ratio=g(info.get("pegRatio")),
            gross_margin=g(info.get("grossMargins")),
            operating_margin=g(info.get("operatingMargins")),
            net_margin=g(info.get("profitMargins")),
            return_on_equity=g(info.get("returnOnEquity")),
            return_on_assets=g(info.get("returnOnAssets")),
            return_on_invested_capital=None,
            asset_turnover=None,
            inventory_turnover=None,
            receivables_turnover=None,
            days_sales_outstanding=None,
            operating_cycle=None,
            working_capital_turnover=None,
            current_ratio=g(info.get("currentRatio")),
            quick_ratio=g(info.get("quickRatio")),
            cash_ratio=None,
            operating_cash_flow_ratio=None,
            debt_to_equity=d2e,
            debt_to_assets=None,
            interest_coverage=None,
            revenue_growth=g(info.get("revenueGrowth")),
            earnings_growth=g(info.get("earningsGrowth")),
            book_value_growth=None,
            earnings_per_share_growth=None,
            free_cash_flow_growth=None,
            operating_income_growth=None,
            ebitda_growth=None,
            payout_ratio=g(info.get("payoutRatio")),
            earnings_per_share=g(info.get("trailingEps")),
            book_value_per_share=g(info.get("bookValue")),
            free_cash_flow_per_share=None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, end=end_date, interval="1d", auto_adjust=True)
        if df.empty:
            return []
        prices = []
        for date, row in df.iterrows():
            prices.append(
                Price(
                    open=float(row["Open"]),
                    close=float(row["Close"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    volume=int(row["Volume"]),
                    time=date.strftime("%Y-%m-%dT%H:%M:%S"),
                )
            )
        return prices

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        t = yf.Ticker(ticker)
        info = t.info or {}

        if period == "ttm":
            return self._ttm_metrics(ticker, t, info, end_date, limit)

        # Annual or quarterly
        inc_all, bs_all, cf_all = self._get_stmt(t, period)
        inc_df = self._filter_df(inc_all, end_date)
        bs_df = self._filter_df(bs_all, end_date)
        cf_df = self._filter_df(cf_all, end_date)

        if inc_df.empty:
            return [self._metrics_from_info_only(ticker, end_date, period, info)]

        results = []
        cols = list(inc_df.columns[:limit])
        for i, col in enumerate(cols):
            inc = inc_df[col]
            bs = bs_df[col] if (not bs_df.empty and col in bs_df.columns) else pd.Series(dtype=float)
            cf = cf_df[col] if (not cf_df.empty and col in cf_df.columns) else pd.Series(dtype=float)

            prev_col = cols[i + 1] if i + 1 < len(cols) else None
            prev_inc = inc_df[prev_col] if prev_col is not None else None
            prev_bs = (bs_df[prev_col] if (prev_col is not None and not bs_df.empty and prev_col in bs_df.columns) else None)

            report_period = col.strftime("%Y-%m-%d")
            results.append(self._build_metrics(ticker, report_period, period, info if i == 0 else {}, inc, bs, cf, prev_inc, prev_bs))
        return results

    def _ttm_metrics(
        self,
        ticker: str,
        t: yf.Ticker,
        info: dict,
        end_date: str,
        limit: int,
    ) -> list[FinancialMetrics]:
        """Compute rolling TTM FinancialMetrics from quarterly statements."""
        inc_q = self._filter_df(t.quarterly_income_stmt, end_date)
        bs_q = self._filter_df(t.quarterly_balance_sheet, end_date)
        cf_q = self._filter_df(t.quarterly_cashflow, end_date)

        if inc_q.empty:
            return [self._metrics_from_info_only(ticker, end_date, "ttm", info)]

        n_cols = len(inc_q.columns)
        results = []
        # Each TTM window requires 4 quarters; slide by 1 quarter per entry.
        max_windows = max(1, n_cols - 3) if n_cols >= 4 else 1
        n_windows = min(limit, max_windows)

        for i in range(n_windows):
            window = min(4, n_cols - i)
            w_cols = list(inc_q.columns[i : i + window])

            ttm_inc = inc_q[w_cols].sum(axis=1)
            ttm_cf = cf_q[w_cols].sum(axis=1) if (not cf_q.empty and all(c in cf_q.columns for c in w_cols)) else pd.Series(dtype=float)
            anchor = inc_q.columns[i]
            ttm_bs = bs_q[anchor] if (not bs_q.empty and anchor in bs_q.columns) else pd.Series(dtype=float)

            report_period = anchor.strftime("%Y-%m-%d")
            results.append(self._build_metrics(ticker, report_period, "ttm", info if i == 0 else {}, ttm_inc, ttm_bs, ttm_cf))
        return results

    # ------------------------------------------------------------------

    def search_line_items(
        self,
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[LineItem]:
        t = yf.Ticker(ticker)
        info = t.info or {}
        currency = info.get("currency", "USD") or "USD"

        if period == "ttm":
            return self._ttm_line_items(ticker, t, line_items, end_date, limit, currency)

        inc_all, bs_all, cf_all = self._get_stmt(t, period)
        inc_df = self._filter_df(inc_all, end_date)
        bs_df = self._filter_df(bs_all, end_date)
        cf_df = self._filter_df(cf_all, end_date)

        # Collect all available period-end dates
        all_dates: set = set()
        for df in (inc_df, bs_df, cf_df):
            if not df.empty:
                all_dates.update(df.columns)

        if not all_dates:
            return []

        sorted_dates = sorted(all_dates, reverse=True)[:limit]
        results = []
        for col in sorted_dates:
            inc = inc_df[col] if (not inc_df.empty and col in inc_df.columns) else pd.Series(dtype=float)
            bs = bs_df[col] if (not bs_df.empty and col in bs_df.columns) else pd.Series(dtype=float)
            cf = cf_df[col] if (not cf_df.empty and col in cf_df.columns) else pd.Series(dtype=float)
            results.append(self._build_line_item(ticker, col.strftime("%Y-%m-%d"), period, currency, line_items, inc, bs, cf))
        return results

    def _ttm_line_items(
        self,
        ticker: str,
        t: yf.Ticker,
        line_items: list[str],
        end_date: str,
        limit: int,
        currency: str,
    ) -> list[LineItem]:
        inc_q = self._filter_df(t.quarterly_income_stmt, end_date)
        bs_q = self._filter_df(t.quarterly_balance_sheet, end_date)
        cf_q = self._filter_df(t.quarterly_cashflow, end_date)

        if inc_q.empty:
            return []

        n_cols = len(inc_q.columns)
        max_windows = max(1, n_cols - 3) if n_cols >= 4 else 1
        n_windows = min(limit, max_windows)

        results = []
        for i in range(n_windows):
            window = min(4, n_cols - i)
            w_cols = list(inc_q.columns[i : i + window])

            ttm_inc = inc_q[w_cols].sum(axis=1)
            ttm_cf = cf_q[w_cols].sum(axis=1) if (not cf_q.empty and all(c in cf_q.columns for c in w_cols)) else pd.Series(dtype=float)
            anchor = inc_q.columns[i]
            ttm_bs = bs_q[anchor] if (not bs_q.empty and anchor in bs_q.columns) else pd.Series(dtype=float)

            results.append(self._build_line_item(ticker, anchor.strftime("%Y-%m-%d"), "ttm", currency, line_items, ttm_inc, ttm_bs, ttm_cf))
        return results

    def _build_line_item(
        self,
        ticker: str,
        report_period: str,
        period: str,
        currency: str,
        line_items: list[str],
        inc: pd.Series,
        bs: pd.Series,
        cf: pd.Series,
    ) -> LineItem:
        data: dict = {
            "ticker": ticker,
            "report_period": report_period,
            "period": period,
            "currency": currency,
        }
        for field in line_items:
            data[field] = self._get_line_item_value(field, inc, bs, cf)
        return LineItem(**data)

    def _get_line_item_value(
        self,
        field: str,
        inc: pd.Series,
        bs: pd.Series,
        cf: pd.Series,
    ) -> Optional[float]:
        """Map a logical field name to a yfinance statement row value."""
        g = self._get_row

        FIELD_MAP: dict[str, object] = {
            # Income statement
            "revenue": lambda: g(inc, "Total Revenue", "Revenue"),
            "gross_profit": lambda: g(inc, "Gross Profit"),
            "operating_income": lambda: g(inc, "Operating Income", "EBIT"),
            "net_income": lambda: g(inc, "Net Income"),
            "ebit": lambda: g(inc, "EBIT", "Operating Income"),
            "ebitda": lambda: (g(inc, "EBITDA") or self._compute_ebitda(inc, cf)),
            "research_and_development": lambda: g(inc, "Research And Development"),
            "interest_expense": lambda: g(inc, "Interest Expense"),
            "depreciation_and_amortization": lambda: g(cf, "Depreciation And Amortization", "Depreciation Amortization Depletion"),
            "income_tax_expense": lambda: g(inc, "Tax Provision", "Income Tax Expense Benefit"),
            "earnings_per_share": lambda: g(inc, "Basic EPS", "Diluted EPS"),
            # Balance sheet
            "total_assets": lambda: g(bs, "Total Assets"),
            "total_liabilities": lambda: g(bs, "Total Liabilities Net Minority Interest", "Total Liabilities"),
            "total_debt": lambda: g(bs, "Total Debt"),
            "current_assets": lambda: g(bs, "Current Assets"),
            "current_liabilities": lambda: g(bs, "Current Liabilities"),
            "cash_and_equivalents": lambda: g(bs, "Cash And Cash Equivalents", "Cash"),
            "shareholders_equity": lambda: g(bs, "Stockholders Equity", "Total Equity Gross Minority Interest"),
            "outstanding_shares": lambda: g(bs, "Ordinary Shares Number", "Share Issued"),
            "book_value_per_share": lambda: self._compute_bvps(bs),
            "working_capital": lambda: self._compute_working_capital(bs),
            "goodwill_and_intangible_assets": lambda: self._compute_goodwill_intangibles(bs),
            # Cash flow
            "free_cash_flow": lambda: g(cf, "Free Cash Flow"),
            "capital_expenditure": lambda: g(cf, "Capital Expenditure"),
            "operating_cash_flow": lambda: g(cf, "Operating Cash Flow", "Cash From Operations"),
            "dividends_and_other_cash_distributions": lambda: g(cf, "Common Stock Dividend Paid", "Payment Of Dividends"),
            # Computed margins
            "gross_margin": lambda: self._compute_margin(inc, "Gross Profit"),
            "operating_margin": lambda: self._compute_margin(inc, "Operating Income"),
            "net_margin": lambda: self._compute_margin(inc, "Net Income"),
        }

        mapper = FIELD_MAP.get(field)
        if mapper is None:
            return None
        try:
            return mapper()  # type: ignore[operator]
        except Exception:
            return None

    def _compute_goodwill_intangibles(self, bs: pd.Series) -> Optional[float]:
        goodwill = self._get_row(bs, "Goodwill")
        intangibles = self._get_row(bs, "Other Intangible Assets", "Intangible Assets")
        if goodwill is not None or intangibles is not None:
            return (goodwill or 0.0) + (intangibles or 0.0)
        return None

    # ------------------------------------------------------------------

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        t = yf.Ticker(ticker)
        try:
            df = t.insider_transactions
        except Exception:
            return []

        if df is None or df.empty:
            return []

        end_ts = pd.Timestamp(end_date)
        start_ts = pd.Timestamp(start_date) if start_date else None

        trades: list[InsiderTrade] = []
        for _, row in df.iterrows():
            # Parse the transaction date — yfinance column names vary by version
            trade_ts: Optional[pd.Timestamp] = None
            for date_col in ("Start Date", "Date", "startDate", "date"):
                raw = row.get(date_col)
                if raw is not None and not (isinstance(raw, float) and math.isnan(raw)):
                    try:
                        trade_ts = pd.Timestamp(raw)
                        break
                    except Exception:
                        pass

            if trade_ts is None:
                continue

            if trade_ts > end_ts:
                continue
            if start_ts is not None and trade_ts < start_ts:
                continue

            ts_str = trade_ts.strftime("%Y-%m-%dT%H:%M:%S")
            name_val = row.get("Insider", row.get("name", ""))
            title_val = row.get("Position", row.get("position", ""))

            trades.append(
                InsiderTrade(
                    ticker=ticker,
                    issuer=None,
                    name=str(name_val) if name_val else None,
                    title=str(title_val) if title_val else None,
                    is_board_director=None,
                    transaction_date=ts_str,
                    transaction_shares=_safe_float(row.get("Shares", row.get("shares"))),
                    transaction_price_per_share=None,
                    transaction_value=_safe_float(row.get("Value", row.get("value"))),
                    shares_owned_before_transaction=None,
                    shares_owned_after_transaction=None,
                    security_title=None,
                    filing_date=ts_str,
                )
            )

            if len(trades) >= limit:
                break

        return trades

    # ------------------------------------------------------------------

    def get_company_news(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        t = yf.Ticker(ticker)
        try:
            raw_news = t.news or []
        except Exception:
            return []

        end_ts = pd.Timestamp(end_date)
        start_ts = pd.Timestamp(start_date) if start_date else None

        result: list[CompanyNews] = []
        for item in raw_news:
            try:
                # yfinance ≥ 0.2.50 wraps fields inside a ``content`` dict;
                # older versions put them at the top level.
                content = item.get("content", item) if isinstance(item, dict) else {}
                if not isinstance(content, dict):
                    content = item

                title = content.get("title", "") or ""

                # Parse publication date
                pub_date: Optional[str] = None
                for date_key in ("pubDate", "providerPublishTime", "date"):
                    raw_date = content.get(date_key)
                    if raw_date is None:
                        raw_date = item.get(date_key) if isinstance(item, dict) else None
                    if raw_date is not None:
                        try:
                            if isinstance(raw_date, (int, float)):
                                pub_date = datetime.fromtimestamp(raw_date).strftime("%Y-%m-%dT%H:%M:%S")
                            else:
                                pub_date = pd.Timestamp(raw_date).strftime("%Y-%m-%dT%H:%M:%S")
                            break
                        except Exception:
                            pass

                if not pub_date:
                    continue

                news_ts = pd.Timestamp(pub_date)
                if news_ts > end_ts:
                    continue
                if start_ts is not None and news_ts < start_ts:
                    continue

                # Source / provider
                provider_info = content.get("provider", {})
                if isinstance(provider_info, dict):
                    source = provider_info.get("displayName", "") or ""
                else:
                    source = str(provider_info) if provider_info else ""

                # Author
                authors = content.get("authors", [])
                author = ""
                if authors:
                    a = authors[0]
                    author = (a.get("name", "") if isinstance(a, dict) else str(a)) or ""

                # URL
                canonical = content.get("canonicalUrl", {})
                if isinstance(canonical, dict):
                    url = canonical.get("url", "") or ""
                else:
                    url = content.get("url", item.get("link", "")) or ""

                result.append(
                    CompanyNews(
                        ticker=ticker,
                        title=title,
                        author=author,
                        source=source,
                        date=pub_date,
                        url=url,
                        sentiment=None,  # yfinance provides no sentiment scores
                    )
                )

                if len(result) >= limit:
                    break
            except Exception:
                continue

        return result

    # ------------------------------------------------------------------

    def get_market_cap(self, ticker: str, end_date: str) -> Optional[float]:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return _safe_float(info.get("marketCap"))
