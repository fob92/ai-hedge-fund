from abc import ABC, abstractmethod
from typing import Optional

from src.data.models import CompanyNews, FinancialMetrics, InsiderTrade, LineItem, Price


class FinancialDataProvider(ABC):
    """Abstract interface for financial data providers."""

    @abstractmethod
    def get_prices(self, ticker: str, start_date: str, end_date: str) -> list[Price]:
        """Fetch daily OHLCV price data for a ticker between start_date and end_date."""
        ...

    @abstractmethod
    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[FinancialMetrics]:
        """Fetch financial metrics for a ticker up to end_date."""
        ...

    @abstractmethod
    def search_line_items(
        self,
        ticker: str,
        line_items: list[str],
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list[LineItem]:
        """Search for specific financial statement line items."""
        ...

    @abstractmethod
    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000,
    ) -> list[InsiderTrade]:
        """Fetch insider transactions for a ticker."""
        ...

    @abstractmethod
    def get_company_news(
        self,
        ticker: str,
        end_date: str,
        start_date: Optional[str] = None,
        limit: int = 1000,
    ) -> list[CompanyNews]:
        """Fetch news articles for a ticker."""
        ...

    @abstractmethod
    def get_market_cap(self, ticker: str, end_date: str) -> Optional[float]:
        """Fetch market capitalisation for a ticker."""
        ...
