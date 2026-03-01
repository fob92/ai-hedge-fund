"""
Market Scanner – LLM-free, cost-efficient stock screener.

Each stock is scored across multiple analyst lenses using rule-based metrics
only.  No LLM calls are made, making this suitable for scanning large watchlists.

Usage examples
--------------
# Scan five tickers with default analysts
poetry run python -m src.scan --tickers AAPL,MSFT,NVDA,TSLA,AMD

# Use specific analyst lenses and return top 5
poetry run python -m src.scan --tickers AAPL,MSFT,NVDA --analysts fundamentals,growth,warren_buffett --top 5

# Filter low-scoring stocks and export
poetry run python -m src.scan --tickers AAPL,MSFT --min-score 55 --output results.csv

# Full analyst suite across a sector watchlist
poetry run python -m src.scan --tickers MP,VALE,FCX,NEM,AA --analysts fundamentals,valuation,ben_graham,warren_buffett

# Show available analysts
poetry run python -m src.scan --list-analysts
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from src.scanner.engine import (
    DEFAULT_ANALYSTS,
    SCORER_REGISTRY,
    run_scan,
)
from src.scanner.output import (
    export_csv,
    print_scan_legend,
    print_scan_table,
    print_summary_stats,
)
from src.utils.progress import progress as global_progress

load_dotenv()

console = Console()


# ────────────────────────────────────────────────────────────────────────────
# Built-in watchlists (extend as needed)
# ────────────────────────────────────────────────────────────────────────────

WATCHLISTS: dict[str, list[str]] = {
    "mag7": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
    "materials": ["FCX", "NEM", "VALE", "MP", "AA", "CLF", "X"],
    "energy": ["XOM", "CVX", "COP", "OXY", "SLB", "HAL", "PSX"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "C"],
    "healthcare": ["JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "UNH"],
    "tech": ["AAPL", "MSFT", "NVDA", "AVGO", "AMD", "INTC", "QCOM"],
    "ev": ["TSLA", "RIVN", "LCID", "NIO", "BYD", "GM", "F"],
}


# ────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    available_analysts = ", ".join(SCORER_REGISTRY.keys())
    available_watchlists = ", ".join(WATCHLISTS.keys())

    parser = argparse.ArgumentParser(
        prog="src.scan",
        description=(
            "LLM-free market scanner.\n"
            "Scores stocks rule-based across multiple analyst lenses."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
analyst keys  : {available_analysts}
watchlist keys: {available_watchlists}

score legend  : >=65% bullish  |  40-65% neutral  |  <=40% bearish
        """,
    )

    src_group = parser.add_mutually_exclusive_group(required=False)
    src_group.add_argument(
        "--tickers",
        metavar="T1,T2,...",
        help="Comma-separated ticker symbols (e.g. AAPL,MSFT,NVDA)",
    )
    src_group.add_argument(
        "--watchlist",
        metavar="NAME",
        choices=list(WATCHLISTS.keys()),
        help=f"Use a built-in watchlist: {available_watchlists}",
    )

    parser.add_argument(
        "--analysts",
        metavar="A1,A2,...",
        default=",".join(DEFAULT_ANALYSTS),
        help=(
            f"Comma-separated analyst scorers.  "
            f"Default: {','.join(DEFAULT_ANALYSTS)}"
        ),
    )
    parser.add_argument(
        "--end-date",
        metavar="YYYY-MM-DD",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis end date (default: today).",
    )
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=12,
        metavar="N",
        help="Months of price history for technical analysis (default 12).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Keep only the top-N results.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        metavar="S",
        help="Minimum composite score 0-100 to include (default 0).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help="Parallel fetch threads (default 4).",
    )
    parser.add_argument(
        "--output",
        metavar="FILE.csv",
        default=None,
        help="Export results to a CSV file.",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Print a legend of analyst descriptions after the table.",
    )
    parser.add_argument(
        "--list-analysts",
        action="store_true",
        help="List all available analyst scorers and exit.",
    )

    return parser


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _list_analysts() -> None:
    """Print a table of all available analyst scorers."""
    t = Table(title="Available Analyst Scorers", box=box.SIMPLE_HEAVY, expand=False)
    t.add_column("Key", style="cyan bold", width=18)
    t.add_column("Label", width=12)
    t.add_column("Style", width=12)
    t.add_column("Description")
    for key, cfg in SCORER_REGISTRY.items():
        t.add_row(
            key,
            cfg.get("label", ""),
            cfg.get("style", ""),
            cfg.get("description", ""),
        )
    console.print()
    console.print(t)
    console.print()


def _validate_analysts(keys: list[str]) -> list[str]:
    invalid = [k for k in keys if k not in SCORER_REGISTRY]
    if invalid:
        console.print(f"[bold red]Unknown analyst(s): {', '.join(invalid)}[/bold red]")
        console.print(
            f"[yellow]Available: {', '.join(SCORER_REGISTRY.keys())}[/yellow]"
        )
        sys.exit(1)
    return keys


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_analysts:
        _list_analysts()
        return

    # Resolve tickers
    if args.watchlist:
        tickers = WATCHLISTS[args.watchlist]
        watchlist_label = f"[{args.watchlist}]"
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        watchlist_label = ""
    else:
        parser.print_help()
        console.print("\n[red]Provide --tickers or --watchlist.[/red]")
        sys.exit(1)

    if not tickers:
        console.print("[red]No tickers to scan.[/red]")
        sys.exit(1)

    analyst_keys = _validate_analysts(
        [a.strip() for a in args.analysts.split(",") if a.strip()]
    )

    end_date: str = args.end_date
    start_date: str = (
        datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=args.lookback_months)
    ).strftime("%Y-%m-%d")

    api_key: str | None = os.environ.get("FINANCIAL_DATASETS_API_KEY")

    # ── Header ─────────────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Market Scanner[/bold cyan]  [dim]LLM-free / rule-based[/dim]\n"
            f"Tickers  : [bold]{', '.join(tickers)}[/bold] {watchlist_label}\n"
            f"Analysts : [bold]{', '.join(analyst_keys)}[/bold]\n"
            f"Period   : {start_date} to {end_date}",
            border_style="cyan",
        )
    )

    # Progress tracker (optional – piggybacks on existing infra)
    _completed: set[str] = set()

    def _on_progress(ticker: str, status: str) -> None:
        if status == "done" and ticker not in _completed:
            _completed.add(ticker)
            pct = round(len(_completed) / len(tickers) * 100)
            console.print(
                f"  [dim][{pct:3d}%][/dim] {ticker} done", end="\r"
            )

    # ── Run scan ───────────────────────────────────────────────────────────
    with console.status(
        f"[bold green]Scanning {len(tickers)} ticker(s)...[/bold green]",
        spinner="dots",
    ):
        results = run_scan(
            tickers=tickers,
            end_date=end_date,
            start_date=start_date,
            analyst_keys=analyst_keys,
            api_key=api_key,
            top_n=args.top,
            min_score=args.min_score,
            workers=args.workers,
            progress_callback=_on_progress,
        )

    if not results:
        console.print(
            f"[yellow]No results match the filters "
            f"(min_score={args.min_score}).[/yellow]"
        )
        return

    # ── Output ─────────────────────────────────────────────────────────────
    title = (
        f"Scan | {end_date}"
        + (f" | top {args.top}" if args.top else "")
        + (f" | {watchlist_label}" if watchlist_label else "")
    )
    print_scan_table(results, analyst_keys, console=console, title=title)
    print_summary_stats(results, console=console)

    if args.legend:
        print_scan_legend(analyst_keys, console=console)

    # ── CSV export ─────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        export_csv(results, analyst_keys, out_path)
        console.print(f"[green]Exported to {out_path}[/green]\n")


if __name__ == "__main__":
    main()
