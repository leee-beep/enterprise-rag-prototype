"""Bounded deterministic query expansion for competitor annual reports."""

from __future__ import annotations

import re


MAX_QUERIES_PER_COMPANY = 2

_INTENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:ai[ -]?servers?|servers?|data[ -]?cent(?:er|re)s?|infrastructure|enterprise)\b", re.I), "server"),
    (re.compile(r"\b(?:ai[ -]?pcs?|notebooks?|laptops?)\b", re.I), "ai_pc"),
    (re.compile(r"\b(?:products?|business areas?|business segments?|product portfolios?|scope of business)\b", re.I), "products"),
    (re.compile(r"\b(?:growth drivers?|growth momentum|growth engines?)\b", re.I), "growth"),
    (re.compile(r"\b(?:ai|artificial intelligence)\b.*\b(?:strateg(?:y|ies|ic)|approach|priorit(?:y|ies))\b|\b(?:strateg(?:y|ies|ic))\b.*\b(?:ai|artificial intelligence)\b", re.I), "ai_strategy"),
)

_ENGLISH_TERMS = {
    "server": "AI server infrastructure data center server products",
    "ai_pc": "AI PC notebook laptop products",
    "products": "main products business scope product portfolio business areas",
    "growth": "growth drivers growth momentum market demand expansion",
    "ai_strategy": "AI strategy products infrastructure applications",
}

_ZH_TW_TERMS = {
    "server": "AI 伺服器 資料中心 伺服器產品 基礎設施",
    "ai_pc": "AI PC 人工智慧個人電腦 筆記型電腦",
    "products": "主要產品 業務內容 業務範圍 產品發展趨勢",
    "growth": "成長動能 成長驅動 市場需求 業務成長",
    "ai_strategy": "人工智慧 AI 策略 產品 基礎設施",
}

# Shared, generic bilingual concepts for expansion and local reranking. Values
# are terminology only: they intentionally contain no company facts.
CONTROLLED_CONCEPTS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "server": (
        ("ai_server", ("ai server", "ai servers", "ai伺服器", "ai 伺服器")),
        ("server", ("server", "servers", "伺服器")),
        ("data_center", ("data center", "data centers", "資料中心")),
        ("infrastructure", ("infrastructure", "基礎設施", "基礎方案")),
    ),
    "ai_pc": (
        ("ai_pc", ("ai pc", "ai pcs", "人工智慧個人電腦")),
        ("notebook", ("notebook", "notebooks", "laptop", "laptops", "筆記型電腦", "筆電")),
    ),
    "products": (
        ("products", ("product", "products", "產品", "產品線", "主要產品")),
        ("business_scope", ("business scope", "business areas", "business segments", "業務內容", "業務範圍", "事業群")),
        ("portfolio", ("product portfolio", "產品組合")),
    ),
    "growth": (
        ("growth", ("growth", "成長", "增長")),
        ("driver", ("growth driver", "growth momentum", "growth engine", "成長動能", "成長引擎", "成長驅動")),
        ("demand", ("market demand", "需求", "市場需求")),
    ),
    "ai_strategy": (
        ("ai", ("artificial intelligence", "人工智慧", "ai")),
        ("strategy", ("strategy", "strategies", "strategic", "策略", "佈局", "布局")),
        ("applications", ("applications", "應用", "產品", "infrastructure", "基礎設施")),
    ),
}


def detect_competitor_intent(question: str) -> str | None:
    """Return the first narrowly recognized controlled retrieval intent."""
    return next((name for pattern, name in _INTENTS if pattern.search(question)), None)


class CompetitorQueryExpander:
    """Return the original query and at most one controlled terminology variant."""

    def expand(self, question: str, company_id: str) -> tuple[str, ...]:
        original = question.strip()
        company = company_id.strip().casefold()
        intent = detect_competitor_intent(original)
        if intent is None:
            return (original,)
        terms = _ENGLISH_TERMS if company == "gigabyte" else _ZH_TW_TERMS
        variant = terms.get(intent)
        if not variant or variant.casefold() == original.casefold():
            return (original,)
        return (original, variant)[:MAX_QUERIES_PER_COMPANY]
