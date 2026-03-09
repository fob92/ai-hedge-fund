"""Quick data diagnostic — run with:  poetry run python -m src._diagnose_data MP"""
import os, sys
os.environ["FINANCIAL_DATA_PROVIDER"] = "yfinance"

import argparse
import yfinance as yf
from src.tools.providers.yfinance_provider import YFinanceProvider
from src.tools.api import search_line_items, get_financial_metrics

parser = argparse.ArgumentParser()
parser.add_argument("ticker", nargs="?", default="MP")
parser.add_argument("--end-date", default="2026-03-01")
args = parser.parse_args()

TICKER = args.ticker.upper()
END    = args.end_date

print(f"\n{'='*60}")
print(f"  Data diagnostic for {TICKER}  (end_date={END})")
print(f"{'='*60}\n")

t = yf.Ticker(TICKER)
info = t.info or {}
print(f"Company   : {info.get('longName', 'N/A')}")
print(f"Sector    : {info.get('sector', 'N/A')}")
print(f"Industry  : {info.get('industry', 'N/A')}")
print(f"Currency  : {info.get('currency', 'N/A')}")
print(f"MarketCap : {info.get('marketCap', 'N/A')}")

print("\n--- Quarterly statement coverage ---")
qi = t.quarterly_income_stmt
qb = t.quarterly_balance_sheet
qc = t.quarterly_cashflow
for label, df in [("income_stmt  ", qi), ("balance_sheet", qb), ("cashflow     ", qc)]:
    cols = list(df.columns) if df is not None and not df.empty else []
    print(f"  {label}: {len(cols)} quarters  {[str(c.date()) for c in cols]}")

print("\n--- Annual statement coverage ---")
ai = t.income_stmt
ab = t.balance_sheet
ac = t.cashflow
for label, df in [("income_stmt  ", ai), ("balance_sheet", ab), ("cashflow     ", ac)]:
    cols = list(df.columns) if df is not None and not df.empty else []
    print(f"  {label}: {len(cols)} years    {[str(c.date()) for c in cols]}")

print("\n--- TTM financial metrics (limit=10) ---")
provider = YFinanceProvider()
metrics = provider.get_financial_metrics(TICKER, END, period="ttm", limit=10)
print(f"  periods returned : {len(metrics)}")
for i, m in enumerate(metrics):
    print(f"  [{i}] {m.report_period}  ROE={m.return_on_equity}  op_margin={m.operating_margin}"
          f"  net_margin={m.net_margin}  d/e={m.debt_to_equity}")

print("\n--- TTM line items (limit=10) ---")
lis = search_line_items(
    TICKER,
    ["revenue", "net_income", "free_cash_flow", "total_assets",
     "shareholders_equity", "capital_expenditure", "depreciation_and_amortization",
     "outstanding_shares", "gross_profit", "operating_income"],
    END, period="ttm", limit=10
)
print(f"  periods returned : {len(lis)}")
for i, li in enumerate(lis):
    print(f"  [{i}] {li.report_period}  rev={li.revenue}  ni={li.net_income}"
          f"  fcf={li.free_cash_flow}  equity={li.shareholders_equity}")

# Highlight what the Buffett agent will see
print("\n--- Buffett agent gate checks (thresholds lowered to match yfinance depth) ---")
print(f"  analyze_fundamentals  needs: >=1 TTM metric  → {'✓ OK' if len(metrics) >= 1 else '✗ FAIL (no data)'}")
print(f"  analyze_consistency   needs: >=2 line items  → {'✓ OK' if len(lis) >= 2 else f'✗ FAIL (only {len(lis)})'}")
print(f"  analyze_moat          needs: >=2 TTM metrics → {'✓ OK' if len(metrics) >= 2 else f'✗ FAIL (only {len(metrics)})'}")
print(f"  analyze_pricing_power needs: >=2 line items  → {'✓ OK' if len(lis) >= 2 else f'✗ FAIL (only {len(lis)})'}")
print(f"  analyze_book_value    needs: >=2 line items  → {'✓ OK' if len(lis) >= 2 else f'✗ FAIL (only {len(lis)})'}")
print(f"  calculate_intrinsic_v needs: >=1 line item   → {'✓ OK' if len(lis) >= 1 else '✗ FAIL (no data)'}")

if metrics:
    m0 = metrics[0]
    nones = [f for f in ["return_on_equity","debt_to_equity","operating_margin","current_ratio","net_margin"]
             if getattr(m0, f, None) is None]
    if nones:
        print(f"\n  ⚠  Latest TTM metric has None values for: {nones}")
    else:
        print(f"\n  ✓  All key ratio fields populated in latest TTM metric")
