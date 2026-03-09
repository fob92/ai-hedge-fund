"""
Rich console output and CSV export for scan results.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.scanner.engine import SCORER_REGISTRY, ScanResult

# ────────────────────────────────────────────────────────────────────────────
# Style maps
# ────────────────────────────────────────────────────────────────────────────

SIGNAL_STYLE: dict[str, str] = {
    "bullish": "bold green",
    "neutral": "yellow",
    "bearish": "bold red",
}

SIGNAL_EMOJI: dict[str, str] = {
    "bullish": "[+]",
    "neutral": "[-]",
    "bearish": "[!]",
}


def _score_color(score_pct: float) -> str:
    """Map a 0-100 score to a terminal colour."""
    if score_pct >= 65:
        return "green"
    if score_pct >= 50:
        return "yellow"
    return "red"


def _fmt_score(score_pct: float, confidence: int = 0) -> Text:
    """Format a score percentage as a coloured Rich Text."""
    color = _score_color(score_pct)
    return Text(f"{score_pct:.0f}%", style=color)


def _fmt_analyst_cell(scorer_result: Optional[dict]) -> Text:
    """Format a single analyst column cell."""
    if scorer_result is None:
        return Text("N/A", style="dim")
    conf = scorer_result.get("confidence", 0)
    details = scorer_result.get("details", {})
    if conf == 0 and details.get("note"):
        return Text("N/A", style="dim")
    score_pct = scorer_result["score"] * 100
    signal = scorer_result.get("signal", "neutral")
    prefix = "+" if signal == "bullish" else ("-" if signal == "bearish" else " ")
    color = _score_color(score_pct)
    return Text(f"{prefix}{score_pct:.0f}%", style=color)


# ────────────────────────────────────────────────────────────────────────────
# Main table renderer
# ────────────────────────────────────────────────────────────────────────────

def print_scan_table(
    results: list[ScanResult],
    analyst_keys: list[str],
    console: Optional[Console] = None,
    title: str = "Market Scan Results",
    show_details: bool = False,
) -> None:
    """Render scan results as a Rich table."""
    if console is None:
        console = Console()

    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
        padding=(0, 1),
        expand=False,
    )

    # Fixed columns
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Ticker", style="bold white", width=8)
    table.add_column("Signal", width=9)
    table.add_column("Score", width=7, justify="right")
    table.add_column("Bull/Bear", width=9, justify="center")

    # Per-analyst columns
    for key in analyst_keys:
        label = SCORER_REGISTRY.get(key, {}).get("label", key[:9])
        table.add_column(label, width=9, justify="right")

    # Rows
    for rank, result in enumerate(results, 1):
        signal_text = Text(
            result.signal.upper(),
            style=SIGNAL_STYLE.get(result.signal, ""),
        )
        score_text = _fmt_score(result.composite_score)
        consensus = f"+{result.bullish_count} -{result.bearish_count}"

        row: list = [
            str(rank),
            result.ticker,
            signal_text,
            score_text,
            Text(consensus, style="dim"),
        ]

        for key in analyst_keys:
            scorer_result = result.analyst_scores.get(key)
            row.append(_fmt_analyst_cell(scorer_result))

        # Error indicator
        if result.error:
            row[1] = Text(result.ticker, style="dim red")

        table.add_row(*row)

    console.print()
    console.print(table)
    console.print()

    # Optional error summary
    errors = [(r.ticker, r.error) for r in results if r.error]
    if errors:
        console.print("[bold red]Errors:[/bold red]")
        for ticker, err in errors:
            console.print(f"  [red]{ticker}[/red]: {err}")
        console.print()


def print_scan_legend(analyst_keys: list[str], console: Optional[Console] = None) -> None:
    """Print a legend explaining each analyst column."""
    if console is None:
        console = Console()

    console.print("[bold dim]Analysts used:[/bold dim]")
    for key in analyst_keys:
        cfg = SCORER_REGISTRY.get(key, {})
        label = cfg.get("label", key)
        desc = cfg.get("description", "")
        console.print(f"  [cyan]{label:10s}[/cyan] {desc}")
    console.print()


def print_summary_stats(results: list[ScanResult], console: Optional[Console] = None) -> None:
    """Print aggregate statistics after the table."""
    if console is None:
        console = Console()

    valid = [r for r in results if not r.error]
    if not valid:
        return

    bull = sum(1 for r in valid if r.signal == "bullish")
    neut = sum(1 for r in valid if r.signal == "neutral")
    bear = sum(1 for r in valid if r.signal == "bearish")
    avg = sum(r.composite_score for r in valid) / len(valid)

    console.print(
        f"[bold]Summary:[/bold]  "
        f"[green]{bull} bullish[/green]  "
        f"[yellow]{neut} neutral[/yellow]  "
        f"[red]{bear} bearish[/red]  "
        f"  [bold]avg score {avg:.1f}%[/bold]"
    )
    console.print()


# ────────────────────────────────────────────────────────────────────────────
# CSV export
# ────────────────────────────────────────────────────────────────────────────

def export_csv(
    results: list[ScanResult],
    analyst_keys: list[str],
    path: Path,
) -> None:
    """Export scan results to a CSV file."""
    labels = [SCORER_REGISTRY.get(k, {}).get("label", k) for k in analyst_keys]

    headers = (
        ["rank", "ticker", "signal", "composite_score"]
        + [f"{lbl}_score" for lbl in labels]
        + [f"{lbl}_signal" for lbl in labels]
        + [f"{lbl}_confidence" for lbl in labels]
    )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for rank, result in enumerate(results, 1):
            row = [rank, result.ticker, result.signal, result.composite_score]
            for key in analyst_keys:
                s = result.analyst_scores.get(key, {})
                row.append(round(s.get("score", 0) * 100, 1))
            for key in analyst_keys:
                s = result.analyst_scores.get(key, {})
                row.append(s.get("signal", "N/A"))
            for key in analyst_keys:
                s = result.analyst_scores.get(key, {})
                row.append(s.get("confidence", 0))
            writer.writerow(row)
