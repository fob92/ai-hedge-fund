import os

from .base import FinancialDataProvider


def get_provider(provider_name: str | None = None) -> FinancialDataProvider:
    """Return a FinancialDataProvider instance for the given provider name.

    The name is resolved from the argument, then the ``FINANCIAL_DATA_PROVIDER``
    environment variable.  Currently only ``"yfinance"`` is supported.
    """
    name = (provider_name or os.environ.get("FINANCIAL_DATA_PROVIDER", "yfinance")).lower()
    if name == "yfinance":
        from .yfinance_provider import YFinanceProvider

        return YFinanceProvider()
    raise ValueError(f"Unknown financial data provider: {name!r}")
