"""
Per-analyst, LLM-free rule-based scorers for scan mode.

Each scorer accepts a TickerData object and returns a dict:

    {
        "score":      float,   # 0.0 – 1.0  (0 = very bearish, 1 = very bullish)
        "signal":     str,     # "bullish" | "neutral" | "bearish"
        "confidence": int,     # 0 – 100
        "details":    dict,    # analyst-specific sub-scores / key metrics
    }

No LLM is invoked; existing quantitative sub-analysis functions are
imported directly from the analyst modules.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.scanner.data import TickerData


# ────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ────────────────────────────────────────────────────────────────────────────

def _no_data(reason: str = "insufficient data") -> dict:
    """Return a neutral placeholder when data is unavailable."""
    return {
        "score": 0.5,
        "signal": "neutral",
        "confidence": 0,
        "details": {"note": reason},
    }


def _signal_from_score(
    score: float,
    bullish_threshold: float = 0.60,
    bearish_threshold: float = 0.40,
) -> str:
    if score >= bullish_threshold:
        return "bullish"
    if score <= bearish_threshold:
        return "bearish"
    return "neutral"


def _confidence(score: float) -> int:
    """Map a 0-1 score to a 0-100 confidence (distance from neutral 0.5)."""
    return round(abs(score - 0.5) * 2 * 100)


# ════════════════════════════════════════════════════════════════════════════
# 1. Fundamentals Scorer
#    Mirrors fundamentals_analyst_agent (pure rule-based, inlined for speed)
# ════════════════════════════════════════════════════════════════════════════

def score_fundamentals(td: TickerData) -> dict:
    """Rule-based fundamental quality scoring (no LLM)."""
    if not td.ttm_metrics:
        return _no_data("no TTM metrics")

    m = td.ttm_metrics[0]
    signals: list[str] = []
    details: dict[str, Any] = {}

    # 1. Profitability
    prof = sum([
        bool(m.return_on_equity and m.return_on_equity > 0.15),
        bool(m.net_margin and m.net_margin > 0.20),
        bool(m.operating_margin and m.operating_margin > 0.15),
    ])
    sig = "bullish" if prof >= 2 else ("bearish" if prof == 0 else "neutral")
    signals.append(sig)
    details["profitability"] = {
        "signal": sig,
        "roe_pct": round(m.return_on_equity * 100, 1) if m.return_on_equity else None,
        "net_margin_pct": round(m.net_margin * 100, 1) if m.net_margin else None,
        "op_margin_pct": round(m.operating_margin * 100, 1) if m.operating_margin else None,
    }

    # 2. Growth
    growth = sum([
        bool(m.revenue_growth and m.revenue_growth > 0.10),
        bool(m.earnings_growth and m.earnings_growth > 0.10),
        bool(m.book_value_growth and m.book_value_growth > 0.10),
    ])
    sig = "bullish" if growth >= 2 else ("bearish" if growth == 0 else "neutral")
    signals.append(sig)
    details["growth"] = {
        "signal": sig,
        "rev_growth_pct": round(m.revenue_growth * 100, 1) if m.revenue_growth else None,
        "earn_growth_pct": round(m.earnings_growth * 100, 1) if m.earnings_growth else None,
        "bv_growth_pct": round(m.book_value_growth * 100, 1) if m.book_value_growth else None,
    }

    # 3. Financial health
    health = sum([
        bool(m.current_ratio and m.current_ratio > 1.5),
        bool(m.debt_to_equity is not None and m.debt_to_equity < 0.5),
        bool(
            m.free_cash_flow_per_share
            and m.earnings_per_share
            and m.free_cash_flow_per_share > m.earnings_per_share * 0.8
        ),
    ])
    sig = "bullish" if health >= 2 else ("bearish" if health == 0 else "neutral")
    signals.append(sig)
    details["health"] = {
        "signal": sig,
        "current_ratio": round(m.current_ratio, 2) if m.current_ratio else None,
        "debt_to_equity": round(m.debt_to_equity, 2) if m.debt_to_equity is not None else None,
    }

    # 4. Valuation (expensive = bearish)
    val_expensive = sum([
        bool(m.price_to_earnings_ratio and m.price_to_earnings_ratio > 25),
        bool(m.price_to_book_ratio and m.price_to_book_ratio > 3),
        bool(m.price_to_sales_ratio and m.price_to_sales_ratio > 5),
    ])
    sig = "bearish" if val_expensive >= 2 else ("bullish" if val_expensive == 0 else "neutral")
    signals.append(sig)
    details["valuation"] = {
        "signal": sig,
        "pe": round(m.price_to_earnings_ratio, 1) if m.price_to_earnings_ratio else None,
        "pb": round(m.price_to_book_ratio, 1) if m.price_to_book_ratio else None,
        "ps": round(m.price_to_sales_ratio, 1) if m.price_to_sales_ratio else None,
    }

    bullish_count = signals.count("bullish")
    bearish_count = signals.count("bearish")
    total = len(signals)
    score = bullish_count / total
    signal = "bullish" if bullish_count > bearish_count else ("bearish" if bearish_count > bullish_count else "neutral")

    return {
        "score": score,
        "signal": signal,
        "confidence": _confidence(score),
        "details": details,
    }


# ════════════════════════════════════════════════════════════════════════════
# 2. Growth Scorer
#    Imports sub-functions from growth_analyst_agent (pure rule-based)
# ════════════════════════════════════════════════════════════════════════════

def score_growth(td: TickerData) -> dict:
    """Growth-focused scoring using growth_analyst_agent sub-functions (no LLM)."""
    try:
        from src.agents.growth_agent import (
            analyze_growth_trends,
            analyze_valuation as growth_analyze_valuation,
            analyze_margin_trends,
            analyze_insider_conviction,
            check_financial_health,
        )
    except ImportError:
        return _no_data("growth_agent unavailable")

    if not td.ttm_metrics or len(td.ttm_metrics) < 2:
        return _no_data("need >=2 TTM metrics periods")

    most_recent = td.ttm_metrics[0]

    try:
        growth_result = analyze_growth_trends(td.ttm_metrics)
        val_result = growth_analyze_valuation(most_recent)
        margin_result = analyze_margin_trends(td.ttm_metrics)
        insider_result = analyze_insider_conviction(td.insider_trades)
        health_result = check_financial_health(most_recent)

        weights = {
            "growth": 0.40,
            "valuation": 0.25,
            "margins": 0.15,
            "insider": 0.10,
            "health": 0.10,
        }
        weighted_score = (
            growth_result["score"] * weights["growth"]
            + val_result["score"] * weights["valuation"]
            + margin_result["score"] * weights["margins"]
            + insider_result["score"] * weights["insider"]
            + health_result["score"] * weights["health"]
        )

        signal = (
            "bullish" if weighted_score > 0.6
            else "bearish" if weighted_score < 0.4
            else "neutral"
        )

        return {
            "score": weighted_score,
            "signal": signal,
            "confidence": _confidence(weighted_score),
            "details": {
                "growth_score": round(growth_result["score"], 2),
                "valuation_score": round(val_result["score"], 2),
                "margin_score": round(margin_result["score"], 2),
                "insider_score": round(insider_result["score"], 2),
                "health_score": round(health_result["score"], 2),
                "weighted_score": round(weighted_score, 2),
            },
        }
    except Exception as exc:
        return _no_data(f"growth scoring error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# 3. Valuation Scorer
#    Uses FCF-yield, P/E, EV/EBITDA, PEG, and P/B from pre-fetched metrics.
#    For deep DCF, imports valuation_analyst sub-functions.
# ════════════════════════════════════════════════════════════════════════════

def score_valuation(td: TickerData) -> dict:
    """Multi-method valuation scoring without LLM."""
    if not td.ttm_metrics:
        return _no_data("no TTM metrics")

    m = td.ttm_metrics[0]

    # Start neutral and adjust based on valuation signals
    score = 0.5
    details: dict[str, Any] = {}
    signals_hit = 0

    def _add(key: str, value: float | None, fmt: str = ".1f") -> None:
        if value is not None:
            details[key] = round(value, 2)

    _add("fcf_yield_pct", (m.free_cash_flow_yield or 0) * 100)
    _add("pe_ratio", m.price_to_earnings_ratio)
    _add("pb_ratio", m.price_to_book_ratio)
    _add("ps_ratio", m.price_to_sales_ratio)
    _add("ev_ebitda", m.enterprise_value_to_ebitda_ratio)
    _add("peg_ratio", m.peg_ratio)
    _add("ev_revenue", m.enterprise_value_to_revenue_ratio)

    # FCF Yield: higher = cheaper = bullish
    if m.free_cash_flow_yield is not None:
        signals_hit += 1
        if m.free_cash_flow_yield > 0.06:
            score += 0.12
        elif m.free_cash_flow_yield > 0.03:
            score += 0.05
        elif m.free_cash_flow_yield < 0.01:
            score -= 0.12

    # P/E Ratio
    if m.price_to_earnings_ratio is not None and m.price_to_earnings_ratio > 0:
        signals_hit += 1
        if m.price_to_earnings_ratio < 15:
            score += 0.12
        elif m.price_to_earnings_ratio < 25:
            score += 0.05
        elif m.price_to_earnings_ratio > 40:
            score -= 0.12
        elif m.price_to_earnings_ratio > 30:
            score -= 0.06

    # EV/EBITDA
    if m.enterprise_value_to_ebitda_ratio is not None and m.enterprise_value_to_ebitda_ratio > 0:
        signals_hit += 1
        if m.enterprise_value_to_ebitda_ratio < 10:
            score += 0.10
        elif m.enterprise_value_to_ebitda_ratio < 18:
            score += 0.04
        elif m.enterprise_value_to_ebitda_ratio > 30:
            score -= 0.10
        elif m.enterprise_value_to_ebitda_ratio > 22:
            score -= 0.05

    # P/B Ratio
    if m.price_to_book_ratio is not None and m.price_to_book_ratio > 0:
        signals_hit += 1
        if m.price_to_book_ratio < 1.5:
            score += 0.08
        elif m.price_to_book_ratio < 3:
            score += 0.03
        elif m.price_to_book_ratio > 6:
            score -= 0.08

    # PEG Ratio
    if m.peg_ratio is not None and m.peg_ratio > 0:
        signals_hit += 1
        if m.peg_ratio < 1.0:
            score += 0.10
        elif m.peg_ratio < 1.5:
            score += 0.04
        elif m.peg_ratio > 2.5:
            score -= 0.08

    if signals_hit == 0:
        return _no_data("no valuation ratios available")

    score = max(0.0, min(1.0, score))
    signal = _signal_from_score(score)

    return {
        "score": score,
        "signal": signal,
        "confidence": _confidence(score),
        "details": details,
    }


# ════════════════════════════════════════════════════════════════════════════
# 4. Technical Scorer
#    Imports sub-functions from technical_analyst_agent (pure rule-based)
# ════════════════════════════════════════════════════════════════════════════

def score_technical(td: TickerData) -> dict:
    """Technical analysis scoring without LLM."""
    try:
        from src.agents.technicals import (
            calculate_mean_reversion_signals,
            calculate_momentum_signals,
            calculate_stat_arb_signals,
            calculate_trend_signals,
            calculate_volatility_signals,
            weighted_signal_combination,
        )
    except ImportError:
        return _no_data("technicals module unavailable")

    if td.prices_df is None or len(td.prices_df) < 55:
        return _no_data("need ≥55 trading days of price data")

    try:
        trend = calculate_trend_signals(td.prices_df)
        mean_rev = calculate_mean_reversion_signals(td.prices_df)
        momentum = calculate_momentum_signals(td.prices_df)
        volatility = calculate_volatility_signals(td.prices_df)
        stat_arb = calculate_stat_arb_signals(td.prices_df)

        strategy_weights = {
            "trend": 0.25,
            "mean_reversion": 0.20,
            "momentum": 0.25,
            "volatility": 0.15,
            "stat_arb": 0.15,
        }
        combined = weighted_signal_combination(
            {
                "trend": trend,
                "mean_reversion": mean_rev,
                "momentum": momentum,
                "volatility": volatility,
                "stat_arb": stat_arb,
            },
            strategy_weights,
        )

        raw_signal = combined["signal"]
        raw_conf = combined["confidence"]  # 0–1 float

        # Map to 0-1 score
        if raw_signal == "bullish":
            score = 0.5 + raw_conf * 0.5
        elif raw_signal == "bearish":
            score = 0.5 - raw_conf * 0.5
        else:
            score = 0.5

        return {
            "score": round(score, 4),
            "signal": raw_signal,
            "confidence": round(raw_conf * 100),
            "details": {
                "trend": trend["signal"],
                "momentum": momentum["signal"],
                "mean_reversion": mean_rev["signal"],
                "volatility": volatility["signal"],
            },
        }
    except Exception as exc:
        return _no_data(f"technical scoring error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# 5. Sentiment Scorer
#    Pre-tagged news sentiment + insider trade direction – NO LLM calls.
#    Articles without a pre-tagged sentiment are simply skipped.
# ════════════════════════════════════════════════════════════════════════════

def score_sentiment(td: TickerData) -> dict:
    """News + insider sentiment without LLM (pre-tagged articles only)."""
    # ── Insider trades ──────────────────────────────────────────────────────
    insider_buys = sum(
        1 for t in td.insider_trades
        if t.transaction_shares is not None and t.transaction_shares > 0
    )
    insider_sells = sum(
        1 for t in td.insider_trades
        if t.transaction_shares is not None and t.transaction_shares < 0
    )
    insider_total = insider_buys + insider_sells

    # ── Pre-tagged news ─────────────────────────────────────────────────────
    tagged_articles = [n for n in td.news if n.sentiment is not None]
    news_positive = sum(1 for n in tagged_articles if n.sentiment == "positive")
    news_negative = sum(1 for n in tagged_articles if n.sentiment == "negative")
    news_total = len(tagged_articles)

    if insider_total == 0 and news_total == 0:
        return _no_data("no insider trades or tagged news")

    # Weighted combination (news-heavy since it's more timely)
    insider_weight = 0.30
    news_weight = 0.70

    insider_score = (insider_buys / insider_total) if insider_total > 0 else 0.5
    news_score = (news_positive / news_total) if news_total > 0 else 0.5

    # Blend proportionally by available data volume
    if insider_total == 0:
        score = news_score
    elif news_total == 0:
        score = insider_score
    else:
        denom = insider_total * insider_weight + news_total * news_weight
        score = (
            insider_score * insider_total * insider_weight
            + news_score * news_total * news_weight
        ) / denom

    signal = "bullish" if score > 0.55 else ("bearish" if score < 0.45 else "neutral")

    return {
        "score": round(score, 4),
        "signal": signal,
        "confidence": _confidence(score),
        "details": {
            "insider_buys": insider_buys,
            "insider_sells": insider_sells,
            "news_positive": news_positive,
            "news_negative": news_negative,
            "news_tagged": news_total,
            "news_coverage_days": 90,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# 6. Warren Buffett Scorer
#    Imports pre-LLM sub-analysis functions from warren_buffett_agent
# ════════════════════════════════════════════════════════════════════════════

def score_warren_buffett(td: TickerData) -> dict:
    """Buffett-style quality + value scoring without LLM."""
    try:
        from src.agents.warren_buffett import (
            analyze_book_value_growth,
            analyze_consistency,
            analyze_fundamentals,
            analyze_management_quality,
            analyze_moat,
            analyze_pricing_power,
            calculate_intrinsic_value,
        )
    except ImportError:
        return _no_data("warren_buffett module unavailable")

    if not td.ttm_metrics or not td.ttm_line_items:
        return _no_data("no TTM metrics or line items")

    try:
        fundamental = analyze_fundamentals(td.ttm_metrics)
        consistency = analyze_consistency(td.ttm_line_items)
        moat = analyze_moat(td.ttm_metrics)
        pricing = analyze_pricing_power(td.ttm_line_items, td.ttm_metrics)
        book_val = analyze_book_value_growth(td.ttm_line_items)
        mgmt = analyze_management_quality(td.ttm_line_items)

        total = (
            fundamental["score"]
            + consistency["score"]
            + moat["score"]
            + pricing["score"]
            + book_val["score"]
            + mgmt["score"]
        )
        max_possible = (
            10  # fundamentals
            + moat.get("max_score", 5)
            + mgmt.get("max_score", 2)
            + 5  # pricing power
            + 5  # book value growth
        )

        # Margin of safety
        margin_of_safety: float | None = None
        if td.market_cap and td.ttm_line_items:
            iv_analysis = calculate_intrinsic_value(td.ttm_line_items)
            iv = iv_analysis.get("intrinsic_value")
            if iv and iv > 0 and td.market_cap > 0:
                margin_of_safety = (iv - td.market_cap) / td.market_cap

        score = max(0.0, min(1.0, total / max_possible)) if max_possible > 0 else 0.5
        signal = "bullish" if score >= 0.65 else ("bearish" if score <= 0.35 else "neutral")

        return {
            "score": round(score, 4),
            "signal": signal,
            "confidence": _confidence(score),
            "details": {
                "fundamental": fundamental["score"],
                "consistency": consistency["score"],
                "moat": moat["score"],
                "pricing_power": pricing["score"],
                "book_value_growth": book_val["score"],
                "management": mgmt["score"],
                "total": round(total, 1),
                "max": max_possible,
                "margin_of_safety_pct": (
                    round(margin_of_safety * 100, 1) if margin_of_safety is not None else None
                ),
            },
        }
    except Exception as exc:
        return _no_data(f"Buffett scoring error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# 7. Ben Graham Scorer
#    Imports pre-LLM sub-analysis functions from ben_graham_agent
# ════════════════════════════════════════════════════════════════════════════

def score_ben_graham(td: TickerData) -> dict:
    """Graham deep-value scoring without LLM."""
    try:
        from src.agents.ben_graham import (
            analyze_earnings_stability,
            analyze_financial_strength,
            analyze_valuation_graham,
        )
    except ImportError:
        return _no_data("ben_graham module unavailable")

    if not td.annual_metrics or not td.annual_line_items:
        return _no_data("no annual metrics or line items")

    try:
        earnings = analyze_earnings_stability(td.annual_metrics, td.annual_line_items)
        strength = analyze_financial_strength(td.annual_line_items)
        valuation = analyze_valuation_graham(td.annual_line_items, td.market_cap)

        total = earnings["score"] + strength["score"] + valuation["score"]
        max_possible = 15
        score = max(0.0, min(1.0, total / max_possible))
        signal = "bullish" if score >= 0.65 else ("bearish" if score <= 0.35 else "neutral")

        return {
            "score": round(score, 4),
            "signal": signal,
            "confidence": _confidence(score),
            "details": {
                "earnings_stability": earnings["score"],
                "financial_strength": strength["score"],
                "graham_valuation": valuation["score"],
                "total": round(total, 1),
                "max": max_possible,
            },
        }
    except Exception as exc:
        return _no_data(f"Graham scoring error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# 8. Cathie Wood Scorer
#    Imports pre-LLM sub-analysis functions from cathie_wood_agent
# ════════════════════════════════════════════════════════════════════════════

def score_cathie_wood(td: TickerData) -> dict:
    """Cathie Wood / ARK disruptive-growth scoring without LLM."""
    try:
        from src.agents.cathie_wood import (
            analyze_cathie_wood_valuation,
            analyze_disruptive_potential,
            analyze_innovation_growth,
        )
    except ImportError:
        return _no_data("cathie_wood module unavailable")

    if not td.annual_metrics or not td.annual_line_items:
        return _no_data("no annual metrics or line items")

    try:
        disruptive = analyze_disruptive_potential(td.annual_metrics, td.annual_line_items)
        innovation = analyze_innovation_growth(td.annual_metrics, td.annual_line_items)
        valuation = analyze_cathie_wood_valuation(td.annual_line_items, td.market_cap)

        total = disruptive["score"] + innovation["score"] + valuation["score"]
        max_possible = 15
        score = max(0.0, min(1.0, total / max_possible))
        signal = "bullish" if score >= 0.65 else ("bearish" if score <= 0.35 else "neutral")

        return {
            "score": round(score, 4),
            "signal": signal,
            "confidence": _confidence(score),
            "details": {
                "disruptive_potential": disruptive["score"],
                "innovation_growth": innovation["score"],
                "cw_valuation": valuation["score"],
                "total": round(total, 1),
                "max": max_possible,
            },
        }
    except Exception as exc:
        return _no_data(f"Cathie Wood scoring error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# 9. Bill Ackman Scorer
#    Imports pre-LLM sub-analysis functions from bill_ackman_agent
# ════════════════════════════════════════════════════════════════════════════

def score_bill_ackman(td: TickerData) -> dict:
    """Ackman activist / quality scoring without LLM."""
    try:
        from src.agents.bill_ackman import (
            analyze_activism_potential,
            analyze_business_quality,
            analyze_financial_discipline,
            analyze_valuation as ackman_valuation,
        )
    except ImportError:
        return _no_data("bill_ackman module unavailable")

    if not td.annual_metrics or not td.annual_line_items:
        return _no_data("no annual metrics or line items")

    try:
        quality = analyze_business_quality(td.annual_metrics, td.annual_line_items)
        discipline = analyze_financial_discipline(td.annual_metrics, td.annual_line_items)
        activism = analyze_activism_potential(td.annual_line_items)
        val = ackman_valuation(td.annual_line_items, td.market_cap or 0)

        total = (
            quality["score"]
            + discipline["score"]
            + activism["score"]
            + val["score"]
        )
        max_possible = 20
        score = max(0.0, min(1.0, total / max_possible))
        signal = "bullish" if score >= 0.65 else ("bearish" if score <= 0.35 else "neutral")

        return {
            "score": round(score, 4),
            "signal": signal,
            "confidence": _confidence(score),
            "details": {
                "business_quality": quality["score"],
                "financial_discipline": discipline["score"],
                "activism_potential": activism["score"],
                "valuation": val["score"],
                "total": round(total, 1),
                "max": max_possible,
            },
        }
    except Exception as exc:
        return _no_data(f"Ackman scoring error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# 10. Michael Burry Scorer
#     Imports pre-LLM sub-analysis functions from michael_burry_agent
# ════════════════════════════════════════════════════════════════════════════

def score_michael_burry(td: TickerData) -> dict:
    """Burry deep-value / contrarian scoring without LLM."""
    try:
        from src.agents.michael_burry import (
            _analyze_balance_sheet,
            _analyze_contrarian_sentiment,
            _analyze_insider_activity,
            _analyze_value,
        )
    except ImportError:
        return _no_data("michael_burry module unavailable")

    if not td.ttm_metrics or not td.ttm_line_items:
        return _no_data("no TTM metrics or line items")

    try:
        value = _analyze_value(td.ttm_metrics, td.ttm_line_items, td.market_cap)
        balance = _analyze_balance_sheet(td.ttm_metrics, td.ttm_line_items)
        insider_act = _analyze_insider_activity(td.insider_trades)
        contrarian = _analyze_contrarian_sentiment(td.news)

        total = (
            value["score"]
            + balance["score"]
            + insider_act["score"]
            + contrarian["score"]
        )
        max_possible = (
            value.get("max_score", 5)
            + balance.get("max_score", 5)
            + insider_act.get("max_score", 3)
            + contrarian.get("max_score", 3)
        )
        if max_possible == 0:
            max_possible = 16

        score = max(0.0, min(1.0, total / max_possible))
        signal = "bullish" if score >= 0.65 else ("bearish" if score <= 0.35 else "neutral")

        return {
            "score": round(score, 4),
            "signal": signal,
            "confidence": _confidence(score),
            "details": {
                "deep_value": value["score"],
                "balance_sheet": balance["score"],
                "insider_activity": insider_act["score"],
                "contrarian_sentiment": contrarian["score"],
                "total": round(total, 1),
                "max": max_possible,
            },
        }
    except Exception as exc:
        return _no_data(f"Burry scoring error: {exc}")
