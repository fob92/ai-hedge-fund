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
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from colorama import init
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

from src.main import run_hedge_fund
from src.utils.analysts import ANALYST_ORDER, ANALYST_CONFIG

load_dotenv()
init(autoreset=True)

# Recording console captures everything printed for the text report
console = Console(record=True)

OUTPUT_ROOT = Path("outputs")

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = [
    {"label": "Grok 4",       "model_name": "grok-4-0709",           "model_provider": "xAI"},
    {"label": "GPT-5.2",      "model_name": "gpt-5.2",               "model_provider": "OpenAI"},
    {"label": "Gemini 3 Pro", "model_name": "gemini-3-pro-preview",   "model_provider": "Google"},
]

# Phrases in LLM reasoning that indicate the agent could not perform a real analysis
_INSUFFICIENT_PHRASES = (
    "insufficient data", "insufficient historical", "insufficient fundamental",
    "too little data", "not enough data", "unable to assess",
    "error in analysis", "defaulting to neutral", "parsing error",
    "using default", "no data available", "cannot assess",
    "circle of competence unclear", "insufficient data across",
)


def _coerce_error_signal(signal: str | None, reasoning: str | None) -> str:
    """Return 'error' if the signal or reasoning indicates missing/bad data."""
    s = (signal or "").lower()
    if s == "error":
        return "error"
    if reasoning:
        low = reasoning.lower()
        if any(phrase in low for phrase in _INSUFFICIENT_PHRASES):
            return "error"
    return signal or "N/A"


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
    "ERROR":   "orange1",
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
def _resolve_analysts(raw: str) -> list[str]:
    """Resolve a comma-separated string of analyst names/keys to canonical ANALYST_CONFIG keys."""
    tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
    resolved: list[str] = []
    for token in tokens:
        # exact key match
        if token in ANALYST_CONFIG:
            resolved.append(token)
            continue
        # fuzzy match on display_name
        match = next(
            (k for k, v in ANALYST_CONFIG.items() if token in v["display_name"].lower()),
            None,
        )
        if match:
            resolved.append(match)
        else:
            console.print(f"[yellow]Warning:[/yellow] unknown analyst '[bold]{token}[/bold]' — skipping.")
    return resolved


def _print_available_analysts() -> None:
    console.print("\n[bold cyan]Available analyst keys:[/bold cyan]")
    rows = sorted(ANALYST_CONFIG.items(), key=lambda x: x[1]["order"])
    for key, cfg in rows:
        console.print(f"  [cyan]{key:<30}[/cyan]  {cfg['display_name']}")
    console.print()


def _run_model(model: dict, tickers: list[str], start_date: str, end_date: str, selected_analysts: list[str]) -> dict:
    """Run the full analyst pipeline for a single model. Returns the result dict."""
    console.print(f"  [cyan]→[/cyan] Starting [bold]{model['label']}[/bold] …")
    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=_build_portfolio(tickers),
        show_reasoning=False,
        selected_analysts=selected_analysts,   # empty list = all analysts
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
    parser.add_argument(
        "--analysts", default=None,
        help=(
            "Comma-separated list of analyst keys or partial display names to include, "
            "e.g. 'warren_buffett,cathie_wood' or 'buffett,cathie'. "
            "Omit to use ALL analysts. Run --list-analysts to see all keys."
        ),
    )
    parser.add_argument(
        "--list-analysts", action="store_true",
        help="Print all available analyst keys and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_analysts:
        _print_available_analysts()
        return

    tickers = [t.strip().upper() for t in args.ticker.split(",") if t.strip()]

    end_date   = args.end_date   or datetime.today().strftime("%Y-%m-%d")
    start_date = args.start_date or (
        datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(years=1)
    ).strftime("%Y-%m-%d")

    # Resolve analyst selection
    selected_analysts: list[str] = []
    if args.analysts:
        selected_analysts = _resolve_analysts(args.analysts)
        if not selected_analysts:
            console.print("[red]No valid analysts matched. Run with --list-analysts to see options.[/red]")
            return
        analyst_labels = ", ".join(ANALYST_CONFIG[k]["display_name"] for k in selected_analysts)
    else:
        analyst_labels = "All analysts"

    console.rule("[bold cyan]Multi-Model Stock Analysis[/bold cyan]")
    console.print(f"[bold]Tickers  :[/bold] {', '.join(tickers)}")
    console.print(f"[bold]Period   :[/bold] {start_date}  →  {end_date}")
    console.print(f"[bold]Models   :[/bold] {', '.join(m['label'] for m in MODELS)}")
    console.print(f"[bold]Analysts :[/bold] {analyst_labels}")
    console.print()

    # ----- run models -----
    results: list[dict] = []

    if args.sequential:
        for model in MODELS:
            results.append(_run_model(model, tickers, start_date, end_date, selected_analysts))
    else:
        console.print("[bold]Running all three models in parallel …[/bold]")
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(_run_model, m, tickers, start_date, end_date, selected_analysts): m["label"]
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

    # ----- persist results -----
    out_dir = _make_output_dir(tickers, selected_analysts, end_date)
    payload = _save_json(out_dir, tickers, results, model_labels, selected_analysts, start_date, end_date)
    _save_text_report(out_dir)
    _save_html_report(out_dir, payload)
    console.print(f"\n[bold]Output saved to:[/bold] [underline]{out_dir}[/underline]")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """Convert display text to a safe filename slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _make_output_dir(tickers: list[str], selected_analysts: list[str], end_date: str) -> Path:
    """Create and return a timestamped output directory."""
    ticker_part   = "-".join(tickers)
    analyst_part  = (
        "-".join(_slugify(ANALYST_CONFIG[k]["display_name"]) for k in selected_analysts)
        if selected_analysts else "all-analysts"
    )
    ts            = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name   = f"{ticker_part}_{analyst_part}_{ts}"
    out_dir       = OUTPUT_ROOT / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_json_payload(
    tickers: list[str],
    results: list[dict],
    model_labels: list[str],
    selected_analysts: list[str],
    start_date: str,
    end_date: str,
) -> dict:
    """Build a fully-structured dict ready for JSON serialisation."""
    run_at = datetime.now().isoformat(timespec="seconds")

    analysts_meta = (
        [{"key": k, "display_name": ANALYST_CONFIG[k]["display_name"]} for k in selected_analysts]
        if selected_analysts
        else [{"key": k, "display_name": v["display_name"]} for k, v in ANALYST_CONFIG.items()]
    )

    payload: dict = {
        "meta": {
            "run_at":     run_at,
            "tickers":    tickers,
            "start_date": start_date,
            "end_date":   end_date,
            "models":     [{"label": m["label"], "model_name": m["model_name"], "provider": m["model_provider"]} for m in MODELS],
            "analysts":   analysts_meta,
        },
        "tickers": {},
    }

    for ticker in tickers:
        ticker_data: dict = {
            "consensus":           {},
            "portfolio_decisions": {},
            "risk_limits":         {},
            "analyst_signals":     {},
        }

        # --- portfolio decisions & consensus ---
        model_actions: list[str] = []
        for model, result in zip(MODELS, results):
            decisions = result.get("decisions") or {}
            decision  = decisions.get(ticker, {})
            action    = decision.get("action",   "N/A")
            qty       = decision.get("quantity",  None)
            conf      = decision.get("confidence", None)
            reasoning = decision.get("reasoning", None)
            ticker_data["portfolio_decisions"][model["label"]] = {
                "action":     action,
                "quantity":   qty,
                "confidence": conf,
                "reasoning":  reasoning,
            }
            model_actions.append(action.upper())

        vote              = Counter(model_actions)
        consensus_action, consensus_votes = vote.most_common(1)[0] if vote else ("N/A", 0)
        ticker_data["consensus"] = {
            "action":      consensus_action,
            "vote_count":  consensus_votes,
            "total_models": len(MODELS),
            "model_votes": {m["label"]: a for m, a in zip(MODELS, model_actions)},
        }

        # --- risk limits ---
        for model, result in zip(MODELS, results):
            risk = result.get("analyst_signals", {}).get("risk_management_agent", {}).get(ticker, {})
            ticker_data["risk_limits"][model["label"]] = {
                "max_long":  risk.get("max_position_size",       risk.get("max_long",  None)),
                "max_short": risk.get("max_short_position_size", risk.get("max_short", None)),
            }

        # --- analyst signals with full reasoning ---
        all_agent_keys: set[str] = set()
        for result in results:
            for agent_key, signals in result.get("analyst_signals", {}).items():
                if agent_key != "risk_management_agent" and ticker in signals:
                    all_agent_keys.add(agent_key)

        for agent_key in sorted(all_agent_keys, key=_analyst_sort_key):
            display = _analyst_display_name(agent_key)
            ticker_data["analyst_signals"][agent_key] = {"display_name": display, "models": {}}
            for model, result in zip(MODELS, results):
                sig = result.get("analyst_signals", {}).get(agent_key, {}).get(ticker, {})
                raw_signal = sig.get("signal",     "N/A")
                reasoning  = sig.get("reasoning",  None)
                ticker_data["analyst_signals"][agent_key]["models"][model["label"]] = {
                    "signal":     _coerce_error_signal(raw_signal, reasoning),
                    "confidence": sig.get("confidence", None),
                    "reasoning":  reasoning,
                }

        payload["tickers"][ticker] = ticker_data

    return payload


def _save_json(
    out_dir: Path,
    tickers: list[str],
    results: list[dict],
    model_labels: list[str],
    selected_analysts: list[str],
    start_date: str,
    end_date: str,
) -> None:
    payload = _build_json_payload(tickers, results, model_labels, selected_analysts, start_date, end_date)
    json_path = out_dir / "analysis.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    console.print(f"  [green]✓[/green] JSON  → [dim]{json_path}[/dim]")
    return payload


def _save_text_report(out_dir: Path) -> None:
    """Export the captured Rich console output as plain text."""
    txt_path = out_dir / "report.txt"
    txt_path.write_text(console.export_text(), encoding="utf-8")
    console.print(f"  [green]✓[/green] Text  → [dim]{txt_path}[/dim]")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def _sig_color(signal: str) -> str:
    s = (signal or "").lower()
    if s in ("bullish", "buy", "long"):    return "#22c55e"
    if s in ("bearish", "sell", "short"): return "#ef4444"
    if s == "error":                       return "#f97316"
    return "#eab308"


def _sig_bg(signal: str) -> str:
    s = (signal or "").lower()
    if s in ("bullish", "buy", "long"):    return "#052e16"
    if s in ("bearish", "sell", "short"): return "#450a0a"
    if s == "error":                       return "#431407"
    return "#422006"


def _action_color(action: str) -> str:
    a = (action or "").lower()
    if a in ("buy", "long"):   return "#22c55e"
    if a in ("sell", "short"): return "#ef4444"
    return "#94a3b8"


def _badge(text: str, color: str, bg: str) -> str:
    return (f'<span style="display:inline-block;padding:3px 10px;border-radius:4px;'
            f'font-weight:700;font-size:0.75rem;letter-spacing:.06em;'
            f'color:{color};background:{bg};border:1px solid {color}33;">'
            f'{text.upper()}</span>')


def _conf_bar(conf) -> str:
    if conf is None:
        return '<span style="color:#475569">—</span>'
    pct = float(conf)
    col = "#22c55e" if pct >= 70 else ("#eab308" if pct >= 40 else "#ef4444")
    return (f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="flex:1;height:6px;background:#1e293b;border-radius:3px;overflow:hidden;">'
            f'<div style="height:100%;width:{pct}%;background:{col};border-radius:3px;"></div></div>'
            f'<span style="color:{col};font-size:.8rem;min-width:36px;text-align:right;">{pct:.0f}%</span>'
            f'</div>')


def _save_html_report(out_dir: Path, payload: dict) -> None:
    """Render a self-contained HTML report from the analysis JSON payload."""
    meta    = payload["meta"]
    models  = [m["label"] for m in meta["models"]]
    tickers = meta["tickers"]
    model_providers = {m["label"]: m["provider"] for m in meta["models"]}

    run_at     = meta["run_at"].replace("T", " ")
    period     = f"{meta['start_date']}  →  {meta['end_date']}"
    ticker_str = ", ".join(tickers)

    # ── model header pills ──────────────────────────────────────────────────
    def model_pill(label: str) -> str:
        prov = model_providers.get(label, "")
        col  = {"xAI": "#7c3aed", "OpenAI": "#16a34a", "Google": "#2563eb"}.get(prov, "#475569")
        return (f'<span style="background:{col}22;color:{col};border:1px solid {col}44;'
                f'padding:2px 10px;border-radius:12px;font-size:.75rem;font-weight:600;">'
                f'{label}<span style="font-weight:400;opacity:.7;"> · {prov}</span></span>')

    # ── build per-ticker sections ────────────────────────────────────────────
    ticker_sections = ""
    for ticker in tickers:
        td       = payload["tickers"][ticker]
        cons     = td["consensus"]
        c_action = cons["action"]
        c_col    = _action_color(c_action)
        c_votes  = cons["vote_count"]
        c_total  = cons["total_models"]

        # consensus banner
        vote_pills = " ".join(
            f'<span style="font-size:.8rem;color:#94a3b8;">{ml}:&nbsp;'
            f'<span style="color:{_action_color(v)};font-weight:600;">{v}</span></span>'
            for ml, v in cons["model_votes"].items()
        )
        consensus_html = f"""
        <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;padding:20px 24px;
                    background:#0f172a;border:1px solid {c_col}44;border-radius:10px;margin-bottom:28px;">
          <div>
            <div style="font-size:.7rem;letter-spacing:.1em;color:#64748b;margin-bottom:4px;">CONSENSUS</div>
            <div style="font-size:2rem;font-weight:800;color:{c_col};">{c_action}</div>
            <div style="font-size:.75rem;color:#64748b;">{c_votes}/{c_total} models agree</div>
          </div>
          <div style="display:flex;gap:10px;flex-wrap:wrap;">{vote_pills}</div>
        </div>"""

        # ── signals table ────────────────────────────────────────────────────
        th_cells = "".join(
            f'<th colspan="2" style="text-align:center;padding:8px 12px;">{model_pill(m)}</th>'
            for m in models
        )
        sub_cells = "".join(
            '<th style="text-align:center;padding:4px 8px;font-weight:400;'
            'font-size:.7rem;color:#64748b;">Signal</th>'
            '<th style="text-align:center;padding:4px 8px;font-weight:400;'
            'font-size:.7rem;color:#64748b;">Conf</th>'
            for _ in models
        )
        signal_rows = ""
        for agent_key, agent_data in td["analyst_signals"].items():
            dname = agent_data["display_name"]
            cells = ""
            for m in models:
                sig = agent_data["models"].get(m, {})
                s   = sig.get("signal") or "N/A"
                c   = sig.get("confidence")
                cells += (f'<td style="text-align:center;padding:8px;">{_badge(s, _sig_color(s), _sig_bg(s))}</td>'
                          f'<td style="padding:8px 4px;">{_conf_bar(c)}</td>')
            signal_rows += (
                f'<tr style="border-top:1px solid #1e293b;">'
                f'<td style="padding:10px 14px;font-weight:600;white-space:nowrap;'  
                f'color:#e2e8f0;">{dname}</td>{cells}</tr>'
            )
        signals_table = f"""
        <section style="margin-bottom:40px;">
          <h2 style="font-size:1rem;font-weight:700;color:#94a3b8;letter-spacing:.07em;
                     text-transform:uppercase;margin-bottom:12px;">Analyst Signals</h2>
          <div style="overflow-x:auto;">
          <table style="width:100%;border-collapse:collapse;background:#0f172a;
                        border:1px solid #1e293b;border-radius:8px;">
            <thead>
              <tr style="background:#1e293b;">
                <th style="text-align:left;padding:10px 14px;color:#94a3b8;
                           font-size:.75rem;letter-spacing:.05em;">Analyst</th>{th_cells}
              </tr>
              <tr style="background:#172033;">
                <th></th>{sub_cells}
              </tr>
            </thead>
            <tbody>{signal_rows}</tbody>
          </table>
          </div>
        </section>"""

        # ── deep-dive reasoning cards ─────────────────────────────────────────
        dive_cards = ""
        for agent_key, agent_data in td["analyst_signals"].items():
            dname  = agent_data["display_name"]
            # summary row (one badge per model)
            badges = "".join(
                f'<span style="margin-right:8px;">{model_pill(m)}&nbsp;'
                f'{_badge(agent_data["models"].get(m,{}).get("signal","N/A"), _sig_color(agent_data["models"].get(m,{}).get("signal","")), _sig_bg(agent_data["models"].get(m,{}).get("signal","")))}</span>'
                for m in models
            )
            # per-model reasoning blocks
            reason_blocks = ""
            for m in models:
                sig  = agent_data["models"].get(m, {})
                s    = sig.get("signal") or "N/A"
                c    = sig.get("confidence")
                text = sig.get("reasoning") or "No reasoning provided."
                c_col_m = _sig_color(s)
                reason_blocks += f"""
                <div style="margin-top:16px;padding:16px;background:#0a0f1a;
                            border:1px solid #1e293b;border-radius:8px;">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    {model_pill(m)}
                    {_badge(s, c_col_m, _sig_bg(s))}
                    {_conf_bar(c)}
                  </div>
                  <p style="margin:0;font-size:.875rem;line-height:1.7;color:#94a3b8;">{text}</p>
                </div>"""
            dive_cards += f"""
            <details style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;
                            margin-bottom:12px;">
              <summary style="padding:14px 18px;cursor:pointer;list-style:none;
                             display:flex;align-items:center;justify-content:space-between;
                             gap:12px;">
                <span style="font-weight:700;color:#e2e8f0;font-size:.95rem;">{dname}</span>
                <span style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;">{badges}</span>
              </summary>
              <div style="padding:4px 18px 18px;">{reason_blocks}</div>
            </details>"""

        deep_dive = f"""
        <section style="margin-bottom:40px;">
          <h2 style="font-size:1rem;font-weight:700;color:#94a3b8;letter-spacing:.07em;
                     text-transform:uppercase;margin-bottom:12px;">Analyst Deep Dive</h2>
          {dive_cards}
        </section>"""

        # ── portfolio decisions ───────────────────────────────────────────────
        pd_cols = "".join(
            f'<th style="padding:8px 14px;text-align:center;">{model_pill(m)}</th>'
            for m in models
        )
        pd_rows = ""
        for row_label, key in [("Action","action"),("Qty","quantity"),("Confidence","confidence"),("Note","reasoning")]:
            cells = ""
            for m in models:
                val = td["portfolio_decisions"].get(m, {}).get(key)
                if key == "action":
                    cells += f'<td style="text-align:center;padding:8px;">{_badge(str(val or "—"), _action_color(str(val)), _sig_bg(str(val)))}</td>'
                elif key == "confidence":
                    cells += f'<td style="padding:8px 14px;">{_conf_bar(val)}</td>'
                else:
                    display_val = str(val) if val is not None else "—"
                    cells += f'<td style="padding:8px 14px;color:#94a3b8;font-size:.85rem;">{display_val}</td>'
            pd_rows += f'<tr style="border-top:1px solid #1e293b;"><td style="padding:8px 14px;color:#64748b;font-weight:600;font-size:.8rem;letter-spacing:.05em;text-transform:uppercase;">{row_label}</td>{cells}</tr>'

        portfolio_section = f"""
        <section style="margin-bottom:40px;">
          <h2 style="font-size:1rem;font-weight:700;color:#94a3b8;letter-spacing:.07em;
                     text-transform:uppercase;margin-bottom:12px;">Portfolio Decisions</h2>
          <div style="overflow-x:auto;">
          <table style="width:100%;border-collapse:collapse;background:#0f172a;
                        border:1px solid #1e293b;border-radius:8px;">
            <thead><tr style="background:#1e293b;">
              <th style="text-align:left;padding:10px 14px;color:#64748b;
                         font-size:.7rem;letter-spacing:.05em;">Metric</th>{pd_cols}
            </tr></thead>
            <tbody>{pd_rows}</tbody>
          </table>
          </div>
        </section>"""

        ticker_sections += f"""
        <div style="margin-bottom:56px;">
          <h1 style="font-size:2.2rem;font-weight:800;color:#f1f5f9;margin:0 0 6px;">
            {ticker}
            <span style="font-size:1rem;font-weight:400;color:#64748b;margin-left:12px;">Analysis Report</span>
          </h1>
          <p style="color:#64748b;font-size:.85rem;margin:0 0 24px;">Period: {period}</p>
          {consensus_html}
          {signals_table}
          {deep_dive}
          {portfolio_section}
        </div>"""

    # ── full page ─────────────────────────────────────────────────────────────
    model_pills_str = " ".join(model_pill(m) for m in models)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Hedge Fund · {ticker_str}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
      background: #020617;
      color: #e2e8f0;
      min-height: 100vh;
    }}
    a {{ color: inherit; }}
    details > summary::-webkit-details-marker {{ display: none; }}
    details > summary::marker {{ display: none; }}
    details[open] > summary {{
      border-bottom: 1px solid #1e293b;
    }}
    summary:hover {{ background: #172033 !important; border-radius: 8px 8px 0 0; }}
    ::-webkit-scrollbar {{ height: 6px; width: 6px; background: #0f172a; }}
    ::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 3px; }}
  </style>
</head>
<body>
  <!-- top bar -->
  <header style="background:#0f172a;border-bottom:1px solid #1e293b;padding:14px 32px;
                 display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
    <div style="display:flex;align-items:center;gap:12px;">
      <span style="font-size:1.1rem;font-weight:800;letter-spacing:-.02em;color:#f1f5f9;">AI Hedge Fund</span>
      <span style="color:#334155;">|</span>
      <span style="color:#64748b;font-size:.85rem;">Multi-Model Analysis</span>
    </div>
    <div style="display:flex;gap:8px;flex-wrap:wrap;">{model_pills_str}</div>
  </header>

  <!-- meta bar -->
  <div style="background:#0c1322;border-bottom:1px solid #1e293b;padding:10px 32px;
              display:flex;gap:24px;flex-wrap:wrap;">
    <span style="font-size:.78rem;color:#475569;">Run: <strong style="color:#94a3b8;">{run_at}</strong></span>
    <span style="font-size:.78rem;color:#475569;">Period: <strong style="color:#94a3b8;">{period}</strong></span>
    <span style="font-size:.78rem;color:#475569;">Ticker(s): <strong style="color:#f1f5f9;">{ticker_str}</strong></span>
  </div>

  <!-- content -->
  <main style="max-width:1100px;margin:0 auto;padding:40px 24px;">
    {ticker_sections}
  </main>

  <!-- footer -->
  <footer style="border-top:1px solid #1e293b;padding:20px 32px;text-align:center;
                 color:#334155;font-size:.75rem;">
    Generated by AI Hedge Fund · {run_at} · For informational purposes only. Not financial advice.
  </footer>
</body>
</html>"""

    html_path = out_dir / "report.html"
    html_path.write_text(html, encoding="utf-8")
    console.print(f"  [green]✓[/green] HTML  → [dim]{html_path}[/dim]")


if __name__ == "__main__":
    main()
