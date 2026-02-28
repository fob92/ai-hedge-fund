"""
Multi-Model Stock Analysis Script
==================================
Runs all analysts against one or more tickers using three models in parallel:
  - Grok 4          (xAI)
  - GPT-5.2         (OpenAI)
  - Gemini 3 Pro    (Google)

Usage:
    poetry run python -m src.multi_model_analysis --ticker AAPL
    poetry run python -m src.multi_model_analysis --ticker AAPL,MSFT,NVDA
    poetry run python -m src.multi_model_analysis --ticker AAPL --start-date 2024-01-01 --end-date 2024-06-01
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from colorama import init
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

from src.main import run_hedge_fund
from src.utils.analysts import ANALYST_ORDER

load_dotenv()
init(autoreset=True)

console = Console()

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = [
    {"label": "Grok 4",       "model_name": "grok-4-0709",           "model_provider": "xAI"},
    {"label": "GPT-5.2",      "model_name": "gpt-5.2",               "model_provider": "OpenAI"},
    {"label": "Gemini 3 Pro", "model_name": "gemini-3-pro-preview",   "model_provider": "Google"},
]

# ---------------------------------------------------------------------------
# Signal styling helpers
# ---------------------------------------------------------------------------
SIGNAL_COLORS = {
    "BULLISH": "green",
    "BUY":     "green",
    "BEARISH": "red",
    "SELL":    "red",
    "NEUTRAL": "yellow",
    "HOLD":    "yellow",
}


def _color_signal(signal: str) -> Text:
    upper = signal.upper()
    color = SIGNAL_COLORS.get(upper, "white")
    return Text(upper, style=f"bold {color}")


def _build_portfolio(tickers: list[str]) -> dict:
    return {
        "cash": 100_000.0,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            t: {"long": 0, "short": 0, "long_cost_basis": 0.0,
                "short_cost_basis": 0.0, "short_margin_used": 0.0}
            for t in tickers
        },
        "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers},
    }


# ---------------------------------------------------------------------------
# Run one model
# ---------------------------------------------------------------------------
def _run_model(model: dict, tickers: list[str], start_date: str, end_date: str) -> dict:
    """Run the full analyst pipeline for a single model. Returns the result dict."""
    console.print(f"  [cyan]→[/cyan] Starting [bold]{model['label']}[/bold] …")
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=_build_portfolio(tickers),
        show_reasoning=False,
        selected_analysts=[],          # empty = all analysts
        model_name=model["model_name"],
        model_provider=model["model_provider"],
    )
    console.print(f"  [green]✓[/green] [bold]{model['label']}[/bold] done.")
    return result


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _analyst_display_name(agent_key: str) -> str:
    mapping = {display: display for display, _ in ANALYST_ORDER}
    raw = agent_key.replace("_agent", "").replace("_", " ").title()
    return raw


def _analyst_sort_key(agent_key: str) -> int:
    name = _analyst_display_name(agent_key)
    for idx, (display, _) in enumerate(ANALYST_ORDER):
        if display.lower() in name.lower() or name.lower() in display.lower():
            return idx
    return 999


def print_analyst_signals_table(tickers: list[str], results: list[dict], model_labels: list[str]) -> None:
    """Print one table per ticker showing every analyst's signal across all models."""
    for ticker in tickers:
        table = Table(
            title=f"[bold white]Analyst Signals — {ticker}[/bold white]",
            box=box.ROUNDED,
            show_lines=True,
            header_style="bold cyan",
        )
        table.add_column("Analyst", style="cyan", min_width=24)
        for label in model_labels:
            table.add_column(f"{label}\nSignal",   justify="center", min_width=10)
            table.add_column(f"{label}\nConf %",   justify="right",  min_width=8)

        # Collect all agent keys across all results for this ticker
        all_agents: set[str] = set()
        for result in results:
            for agent_key, signals in result.get("analyst_signals", {}).items():
                if agent_key == "risk_management_agent":
                    continue
                if ticker in signals:
                    all_agents.add(agent_key)

        sorted_agents = sorted(all_agents, key=_analyst_sort_key)

        for agent_key in sorted_agents:
            row: list = [_analyst_display_name(agent_key)]
            for result in results:
                sig = result.get("analyst_signals", {}).get(agent_key, {}).get(ticker, {})
                signal  = sig.get("signal", "N/A").upper()
                conf    = sig.get("confidence", "—")
                conf_str = f"{conf}%" if isinstance(conf, (int, float)) else str(conf)
                row.append(_color_signal(signal) if signal != "N/A" else Text("N/A", style="dim"))
                row.append(Text(conf_str, style="dim white"))
            table.add_row(*row)

        console.print(table)
        console.print()


def print_portfolio_summary_table(tickers: list[str], results: list[dict], model_labels: list[str]) -> None:
    """Print the final portfolio decision table — one row per ticker, one column group per model."""
    table = Table(
        title="[bold white]Portfolio Decision Summary[/bold white]",
        box=box.DOUBLE_EDGE,
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("Ticker", style="bold cyan", min_width=8)
    for label in model_labels:
        table.add_column(f"{label}\nAction",   justify="center", min_width=10)
        table.add_column(f"{label}\nQty",      justify="right",  min_width=6)
        table.add_column(f"{label}\nConf %",   justify="right",  min_width=8)
    table.add_column("Consensus", justify="center", min_width=12)

    for ticker in tickers:
        row: list = [ticker]
        signals_for_consensus: list[str] = []

        for result in results:
            decisions = result.get("decisions") or {}
            decision  = decisions.get(ticker, {})
            action    = decision.get("action", "N/A").upper()
            qty       = decision.get("quantity", "—")
            conf      = decision.get("confidence", "—")
            conf_str  = f"{conf}%" if isinstance(conf, (int, float)) else str(conf)

            row.append(_color_signal(action) if action != "N/A" else Text("N/A", style="dim"))
            row.append(Text(str(qty), style="white"))
            row.append(Text(conf_str, style="dim white"))
            signals_for_consensus.append(action)

        # Simple majority-vote consensus
        from collections import Counter
        vote = Counter(signals_for_consensus)
        consensus_signal, _ = vote.most_common(1)[0] if vote else ("N/A", 0)
        row.append(_color_signal(consensus_signal))

        table.add_row(*row)

    console.print(table)
    console.print()


def print_risk_summary_table(tickers: list[str], results: list[dict], model_labels: list[str]) -> None:
    """Print risk management limits per ticker per model."""
    table = Table(
        title="[bold white]Risk Management Limits[/bold white]",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold yellow",
    )
    table.add_column("Ticker", style="bold cyan", min_width=8)
    for label in model_labels:
        table.add_column(f"{label}\nMax Long $",  justify="right", min_width=12)
        table.add_column(f"{label}\nMax Short $", justify="right", min_width=12)

    for ticker in tickers:
        row: list = [ticker]
        for result in results:
            risk_signals = result.get("analyst_signals", {}).get("risk_management_agent", {})
            risk = risk_signals.get(ticker, {})
            max_long  = risk.get("max_position_size", risk.get("max_long", "—"))
            max_short = risk.get("max_short_position_size", risk.get("max_short", "—"))
            fmt = lambda v: f"${v:,.0f}" if isinstance(v, (int, float)) else str(v)
            row.append(Text(fmt(max_long),  style="green"))
            row.append(Text(fmt(max_short), style="red"))
        table.add_row(*row)

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze stocks with Grok 4, GPT-5.2, and Gemini 3 Pro simultaneously."
    )
    parser.add_argument(
        "--ticker", required=True,
        help="Comma-separated list of tickers, e.g. AAPL,MSFT,NVDA",
    )
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end-date",   default=None, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run models one after another instead of in parallel (useful if rate-limited)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]

    end_date   = args.end_date   or datetime.today().strftime("%Y-%m-%d")
    start_date = args.start_date or (
        datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=3)
    ).strftime("%Y-%m-%d")

    console.rule("[bold cyan]Multi-Model Stock Analysis[/bold cyan]")
    console.print(f"[bold]Tickers :[/bold] {', '.join(tickers)}")
    console.print(f"[bold]Period  :[/bold] {start_date}  →  {end_date}")
    console.print(f"[bold]Models  :[/bold] {', '.join(m['label'] for m in MODELS)}")
    console.print()

    # ----- run models -----
    results: list[dict] = []

    if args.sequential:
        for model in MODELS:
            results.append(_run_model(model, tickers, start_date, end_date))
    else:
        console.print("[bold]Running all three models in parallel …[/bold]")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(_run_model, m, tickers, start_date, end_date): m["label"]
                for m in MODELS
            }
            ordered: dict[str, dict] = {}
            for future in concurrent.futures.as_completed(futures):
                label = futures[future]
                try:
                    ordered[label] = future.result()
                except Exception as exc:
                    console.print(f"[red]✗ {label} failed: {exc}[/red]")
                    ordered[label] = {"decisions": {}, "analyst_signals": {}}
        # preserve MODELS order
        results = [ordered[m["label"]] for m in MODELS]

    model_labels = [m["label"] for m in MODELS]

    console.print()
    console.rule("[bold cyan]Results[/bold cyan]")
    console.print()

    # 1. Analyst signals per ticker
    print_analyst_signals_table(tickers, results, model_labels)

    # 2. Risk limits
    print_risk_summary_table(tickers, results, model_labels)

    # 3. Portfolio decision summary
    print_portfolio_summary_table(tickers, results, model_labels)

    console.rule("[bold green]Analysis Complete[/bold green]")


if __name__ == "__main__":
    main()
