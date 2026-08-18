"""Deterministic routing and structured planning for competitor questions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


COMPANY_ORDER = ("gigabyte", "asus", "msi")
COMPANY_ALIASES = {
    "gigabyte": ("gigabyte", "技嘉"),
    "asus": ("asus", "華碩"),
    "msi": ("msi", "微星"),
}
SUPPORTED_YEARS = frozenset({2024, 2025})

FINANCIAL_TERMS = (
    ("revenue_yoy_growth", ("revenue growth", "營收成長", "營業收入成長")),
    ("gross_margin", ("gross margin", "毛利率")),
    ("operating_margin", ("operating margin", "營業利益率")),
    ("net_margin", ("net margin", "淨利率")),
    ("gross_profit", ("gross profit", "毛利")),
    ("operating_income", ("operating income", "operating profit", "營業利益")),
    ("net_income", ("net income", "net profit", "淨利")),
    ("revenue", ("operating revenue", "revenue", "營業收入", "營收")),
    ("eps", ("eps", "earnings per share", "每股盈餘")),
    ("financial_performance", ("financial performance", "財務表現")),
)
QUALITATIVE_TERMS = (
    ("ai_strategy", ("ai strategy", "ai strategies", "人工智慧策略", "ai 策略")),
    ("ai_infrastructure", ("ai infrastructure", "人工智慧基礎設施", "ai 基礎設施")),
    ("ai_server", ("ai server", "ai servers", "ai 伺服器")),
    ("ai_pc", ("ai pc", "ai pcs", "ai 電腦")),
    (
        "enterprise_server",
        ("enterprise strategy", "server strategy", "enterprise server", "企業策略", "伺服器策略"),
    ),
    (
        "products_business",
        ("product strategy", "products", "business areas", "business strategy", "產品策略", "產品", "業務領域", "事業策略"),
    ),
    ("growth_drivers", ("growth driver", "growth drivers", "成長動能", "成長驅動")),
    ("positioning", ("positioning", "market position", "市場定位")),
)
UNSUPPORTED_FINANCIAL_TERMS = (
    ("roe", ("roe", "return on equity", "股東權益報酬率")),
    ("ebitda", ("ebitda",)),
    ("free_cash_flow", ("free cash flow", "自由現金流")),
    ("stock_price", ("stock price", "share price", "股價")),
    ("valuation_multiple", ("valuation multiple", "valuation multiples", "p/e", "price earnings", "估值倍數", "本益比")),
    ("currency_conversion", ("currency conversion", "convert currency", "匯率換算", "貨幣換算")),
    ("forecast", ("forecast", "projection", "預測", "預估")),
    ("quarterly", ("quarterly", "quarter", "q1", "q2", "q3", "q4", "季度", "季營收")),
)

_RANK_TERMS = ("rank", "ranking", "排名")
_CHANGE_TERMS = (
    "increase",
    "decrease",
    "change",
    "increased",
    "decreased",
    "changed",
    "成長",
    "變化",
    "增加",
    "減少",
)
_COMPARE_TERMS = ("compare", "comparison", "versus", " vs ", "比較", "相較")
_ALL_COMPANY_TERMS = ("the companies", "all companies", "three companies", "各公司", "三家公司", "公司間")
_YEAR_PATTERN = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


class AnalysisRoute(str, Enum):
    QUALITATIVE = "qualitative"
    FINANCIAL = "financial"
    COMBINED = "combined"


class PlanStatus(str, Enum):
    READY = "ready"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED = "unsupported"


class FinancialOperation(str, Enum):
    METRIC_LOOKUP = "metric_lookup"
    COMPARISON = "comparison"
    RANKING = "ranking"
    YEAR_CHANGE = "year_change"


@dataclass(frozen=True)
class FinancialRequest:
    """Structured financial work requested by a question, never executed here."""

    metrics: tuple[str, ...]
    operation: FinancialOperation
    fiscal_years: tuple[int, ...]


@dataclass(frozen=True)
class AnalysisPlan:
    """Immutable orchestration metadata produced without retrieval or models."""

    question: str
    route: AnalysisRoute | None
    status: PlanStatus
    requested_companies: tuple[str, ...]
    qualitative_intents: tuple[str, ...]
    financial_intents: tuple[str, ...]
    financial_request: FinancialRequest | None
    fiscal_years: tuple[int, ...]
    unsupported_intents: tuple[str, ...]
    reasons: tuple[str, ...]


class DeterministicQuestionRouter:
    """Map supported competitor questions to explainable analysis plans."""

    def plan(self, question: str) -> AnalysisPlan:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")
        original = question.strip()
        normalized = original.casefold()

        qualitative = _detect_terms(normalized, QUALITATIVE_TERMS)
        financial = _detect_financial_terms(normalized)
        unsupported = list(_detect_terms(normalized, UNSUPPORTED_FINANCIAL_TERMS))
        years, unsupported_years = _detect_years(normalized)
        unsupported.extend(f"year:{year}" for year in unsupported_years)

        explicit_companies = tuple(
            company
            for company in COMPANY_ORDER
            if any(_contains(normalized, alias) for alias in COMPANY_ALIASES[company])
        )
        comparative = _contains_any(normalized, _COMPARE_TERMS + _RANK_TERMS + _ALL_COMPANY_TERMS)
        has_intent = bool(qualitative or financial or unsupported)
        defaulted_companies = not explicit_companies and comparative and has_intent
        companies = COMPANY_ORDER if defaulted_companies else explicit_companies

        has_financial_family = bool(financial or unsupported)
        route = _route(bool(qualitative), has_financial_family)
        financial_request = (
            FinancialRequest(
                metrics=financial,
                operation=_financial_operation(normalized, years),
                fiscal_years=years,
            )
            if financial
            else None
        )

        reasons: list[str] = []
        reasons.extend(f"detected_company:{company}" for company in companies)
        if defaulted_companies:
            reasons.append("defaulted_comparative_companies:all")
        reasons.extend(f"detected_year:{year}" for year in years)
        reasons.extend(f"detected_qualitative_intent:{intent}" for intent in qualitative)
        reasons.extend(f"detected_financial_metric:{intent}" for intent in financial)
        reasons.extend(f"unsupported_intent:{intent}" for intent in unsupported)
        if qualitative and has_financial_family:
            reasons.append("both_intent_families_present")

        missing_requirements: list[str] = []
        if route is not None and not companies:
            missing_requirements.append("missing_required_company")
        if financial_request is not None:
            missing_requirements.extend(
                _financial_operation_requirements(financial_request, companies)
            )
        if financial == ("financial_performance",):
            missing_requirements.append("unresolved_financial_metric:financial_performance")

        if unsupported:
            status = PlanStatus.UNSUPPORTED
        elif route is None:
            status = PlanStatus.AMBIGUOUS
            reasons.append("unresolved_intent")
        elif missing_requirements:
            status = PlanStatus.AMBIGUOUS
            reasons.extend(missing_requirements)
        else:
            status = PlanStatus.READY

        return AnalysisPlan(
            question=original,
            route=route,
            status=status,
            requested_companies=companies,
            qualitative_intents=qualitative,
            financial_intents=financial,
            financial_request=financial_request,
            fiscal_years=years,
            unsupported_intents=tuple(unsupported),
            reasons=tuple(reasons),
        )


def _detect_financial_terms(text: str) -> tuple[str, ...]:
    detected: list[str] = []
    for intent, phrases in FINANCIAL_TERMS:
        if not _contains_any(text, phrases):
            continue
        # Prefer the supported calculated metric over its overlapping raw label.
        if intent == "gross_profit" and "gross_margin" in detected:
            continue
        if intent == "operating_income" and "operating_margin" in detected:
            continue
        if intent == "net_income" and "net_margin" in detected:
            continue
        if intent == "revenue" and "revenue_yoy_growth" in detected:
            continue
        detected.append(intent)
    return tuple(detected)


def _detect_terms(
    text: str, vocabulary: tuple[tuple[str, tuple[str, ...]], ...]
) -> tuple[str, ...]:
    return tuple(intent for intent, phrases in vocabulary if _contains_any(text, phrases))


def _detect_years(text: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    mentioned = {int(value) for value in _YEAR_PATTERN.findall(text)}
    supported = tuple(sorted(mentioned & SUPPORTED_YEARS))
    unsupported = tuple(sorted(mentioned - SUPPORTED_YEARS))
    return supported, unsupported


def _financial_operation(text: str, years: tuple[int, ...]) -> FinancialOperation:
    if _contains_any(text, _RANK_TERMS):
        return FinancialOperation.RANKING
    if _contains_any(text, _CHANGE_TERMS):
        return FinancialOperation.YEAR_CHANGE
    if _contains_any(text, _COMPARE_TERMS):
        return FinancialOperation.COMPARISON
    return FinancialOperation.METRIC_LOOKUP


def _financial_operation_requirements(
    request: FinancialRequest, companies: tuple[str, ...]
) -> tuple[str, ...]:
    """Return deterministic reasons for an incomplete financial operation."""
    reasons: list[str] = []
    operation = request.operation

    if operation is FinancialOperation.METRIC_LOOKUP:
        if len(request.fiscal_years) != 1:
            reasons.append("missing_required_year")
    elif operation is FinancialOperation.COMPARISON:
        if len(companies) < 2:
            reasons.append("insufficient_companies_for_comparison")
        if len(request.fiscal_years) != 1:
            reasons.append("missing_required_year")
    elif operation is FinancialOperation.RANKING:
        if len(companies) < 2:
            reasons.append("insufficient_companies_for_ranking")
        if len(request.fiscal_years) != 1:
            reasons.append("missing_required_year")
    else:
        if len(companies) != 1:
            reasons.append("invalid_company_count_for_year_change")
        if len(request.fiscal_years) != 2:
            reasons.append("insufficient_years_for_year_change")
        margin_metrics = {"gross_margin", "operating_margin", "net_margin"}
        if any(metric not in margin_metrics for metric in request.metrics):
            reasons.append("unsupported_metric_for_year_change")

    return tuple(reasons)


def _route(qualitative: bool, financial: bool) -> AnalysisRoute | None:
    if qualitative and financial:
        return AnalysisRoute.COMBINED
    if qualitative:
        return AnalysisRoute.QUALITATIVE
    if financial:
        return AnalysisRoute.FINANCIAL
    return None


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains(text, phrase) for phrase in phrases)


def _contains(text: str, phrase: str) -> bool:
    if phrase.isascii() and all(character.isalnum() or character.isspace() for character in phrase):
        plural = "(?:s)?" if phrase[-1].isalpha() and not phrase.endswith("s") else ""
        return re.search(rf"(?<!\w){re.escape(phrase)}{plural}(?!\w)", text) is not None
    return phrase in text
