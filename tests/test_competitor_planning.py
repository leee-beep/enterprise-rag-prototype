"""Offline tests for deterministic competitor question planning."""

from __future__ import annotations

import pytest

from enterprise_rag.competitor_planning import (
    AnalysisRoute,
    DeterministicQuestionRouter,
    FinancialOperation,
    PlanStatus,
)


ROUTER = DeterministicQuestionRouter()


@pytest.mark.parametrize(
    "question,intent",
    (
        ("What is ASUS's AI strategy?", "ai_strategy"),
        ("What does MSI say about AI servers?", "ai_server"),
        ("Explain ASUS AI PCs.", "ai_pc"),
        ("Compare the companies' product strategy.", "products_business"),
        ("華碩的人工智慧策略是什麼？", "ai_strategy"),
        ("微星的 AI 伺服器策略是什麼？", "ai_server"),
        ("技嘉有哪些成長動能？", "growth_drivers"),
    ),
)
def test_qualitative_routing(question: str, intent: str) -> None:
    plan = ROUTER.plan(question)
    assert plan.route is AnalysisRoute.QUALITATIVE
    assert plan.status is PlanStatus.READY
    assert intent in plan.qualitative_intents
    assert not plan.financial_intents


@pytest.mark.parametrize(
    "question,metric,operation",
    (
        ("What was Gigabyte's 2025 revenue?", "revenue", FinancialOperation.METRIC_LOOKUP),
        ("What was ASUS's 2025 gross profit?", "gross_profit", FinancialOperation.METRIC_LOOKUP),
        ("What was MSI's 2025 operating income?", "operating_income", FinancialOperation.METRIC_LOOKUP),
        ("What was Gigabyte's 2025 net income?", "net_income", FinancialOperation.METRIC_LOOKUP),
        ("What was Gigabyte's 2025 revenue growth?", "revenue_yoy_growth", FinancialOperation.METRIC_LOOKUP),
        ("Rank the companies by 2025 gross margin.", "gross_margin", FinancialOperation.RANKING),
        ("Compare ASUS and MSI 2025 operating margin.", "operating_margin", FinancialOperation.COMPARISON),
        ("Did ASUS net margin increase from 2024 to 2025?", "net_margin", FinancialOperation.YEAR_CHANGE),
        ("What was MSI 2025 EPS?", "eps", FinancialOperation.METRIC_LOOKUP),
        ("比較各公司 2025 年毛利率。", "gross_margin", FinancialOperation.COMPARISON),
        ("華碩 2024 到 2025 年營業利益率有增加嗎？", "operating_margin", FinancialOperation.YEAR_CHANGE),
        ("微星 2025 年每股盈餘是多少？", "eps", FinancialOperation.METRIC_LOOKUP),
        ("華碩 2025 年營業收入是多少？", "revenue", FinancialOperation.METRIC_LOOKUP),
        ("技嘉 2025 年淨利是多少？", "net_income", FinancialOperation.METRIC_LOOKUP),
    ),
)
def test_financial_routing(
    question: str, metric: str, operation: FinancialOperation
) -> None:
    plan = ROUTER.plan(question)
    assert plan.route is AnalysisRoute.FINANCIAL
    assert plan.status is PlanStatus.READY
    assert metric in plan.financial_intents
    assert plan.financial_request is not None
    assert plan.financial_request.operation is operation


@pytest.mark.parametrize(
    "question,qualitative,financial",
    (
        ("Compare ASUS, Gigabyte and MSI's AI strategies and 2025 gross margins.", "ai_strategy", "gross_margin"),
        ("Compare product strategy and 2025 revenue growth for the companies.", "products_business", "revenue_yoy_growth"),
        ("比較華碩、技嘉和微星 2025 年的 AI 策略與淨利率。", "ai_strategy", "net_margin"),
    ),
)
def test_combined_routing(question: str, qualitative: str, financial: str) -> None:
    plan = ROUTER.plan(question)
    assert plan.route is AnalysisRoute.COMBINED
    assert plan.status is PlanStatus.READY
    assert qualitative in plan.qualitative_intents
    assert financial in plan.financial_intents
    assert "both_intent_families_present" in plan.reasons


@pytest.mark.parametrize(
    "question,companies",
    (
        ("What is GIGABYTE's 2025 revenue?", ("gigabyte",)),
        ("Compare ASUS and MSI 2025 net margin.", ("asus", "msi")),
        ("比較技嘉、華碩與微星 2025 年毛利率。", ("gigabyte", "asus", "msi")),
        ("ASUS 華碩 2025 gross margin", ("asus",)),
    ),
)
def test_company_alias_detection(question: str, companies: tuple[str, ...]) -> None:
    assert ROUTER.plan(question).requested_companies == companies


def test_comparative_default_selects_all_companies() -> None:
    plan = ROUTER.plan("Compare the companies' 2025 gross margins.")
    assert plan.requested_companies == ("gigabyte", "asus", "msi")
    assert "defaulted_comparative_companies:all" in plan.reasons


@pytest.mark.parametrize(
    "question,operation,reason",
    (
        (
            "Compare ASUS 2025 gross margin.",
            FinancialOperation.COMPARISON,
            "insufficient_companies_for_comparison",
        ),
        (
            "Rank ASUS by 2025 gross margin.",
            FinancialOperation.RANKING,
            "insufficient_companies_for_ranking",
        ),
        (
            "Did MSI net margin change in 2025?",
            FinancialOperation.YEAR_CHANGE,
            "insufficient_years_for_year_change",
        ),
    ),
)
def test_incomplete_requested_operation_is_ambiguous_without_downgrade(
    question: str, operation: FinancialOperation, reason: str
) -> None:
    plan = ROUTER.plan(question)
    assert plan.status is PlanStatus.AMBIGUOUS
    assert plan.financial_request is not None
    assert plan.financial_request.operation is operation
    assert reason in plan.reasons


@pytest.mark.parametrize(
    "question,operation",
    (
        ("Compare ASUS and MSI 2025 gross margin.", FinancialOperation.COMPARISON),
        ("Rank ASUS and MSI by 2025 gross margin.", FinancialOperation.RANKING),
        (
            "Did MSI net margin increase from 2024 to 2025?",
            FinancialOperation.YEAR_CHANGE,
        ),
    ),
)
def test_complete_requested_operation_remains_ready(
    question: str, operation: FinancialOperation
) -> None:
    plan = ROUTER.plan(question)
    assert plan.status is PlanStatus.READY
    assert plan.financial_request is not None
    assert plan.financial_request.operation is operation


@pytest.mark.parametrize(
    "question,reason",
    (
        ("Compare ASUS 2025 gross margin.", "insufficient_companies_for_comparison"),
        ("Rank ASUS by 2025 gross margin.", "insufficient_companies_for_ranking"),
    ),
)
def test_explicit_single_company_is_not_replaced_by_comparative_default(
    question: str, reason: str
) -> None:
    plan = ROUTER.plan(question)
    assert plan.requested_companies == ("asus",)
    assert "defaulted_comparative_companies:all" not in plan.reasons
    assert reason in plan.reasons


def test_unsupported_request_takes_precedence_over_incomplete_operation() -> None:
    plan = ROUTER.plan("Compare ASUS 2025 gross margin and ROE.")
    assert plan.status is PlanStatus.UNSUPPORTED
    assert plan.unsupported_intents == ("roe",)


def test_incomplete_combined_financial_operation_is_ambiguous() -> None:
    plan = ROUTER.plan("Compare ASUS AI strategy and 2025 gross margin.")
    assert plan.route is AnalysisRoute.COMBINED
    assert plan.status is PlanStatus.AMBIGUOUS
    assert plan.qualitative_intents == ("ai_strategy",)
    assert plan.financial_intents == ("gross_margin",)
    assert "insufficient_companies_for_comparison" in plan.reasons


def test_incomplete_operation_reason_codes_are_deterministic() -> None:
    question = "Did MSI net margin change in 2025?"
    assert ROUTER.plan(question).reasons == ROUTER.plan(question).reasons


def test_arbitrary_question_does_not_default_companies() -> None:
    plan = ROUTER.plan("What is an AI strategy?")
    assert plan.requested_companies == ()
    assert plan.status is PlanStatus.AMBIGUOUS
    assert "missing_required_company" in plan.reasons


@pytest.mark.parametrize(
    "question",
    (
        "What is gross margin?",
        "What is AI strategy?",
        "Explain revenue growth.",
        "Tell me about AI servers.",
    ),
)
def test_non_comparative_question_does_not_default_companies(question: str) -> None:
    plan = ROUTER.plan(question)
    assert plan.requested_companies == ()
    assert plan.status is PlanStatus.AMBIGUOUS
    assert "defaulted_comparative_companies:all" not in plan.reasons


def test_unresolved_comparison_does_not_fabricate_an_intent() -> None:
    plan = ROUTER.plan("Compare them.")
    assert plan.route is None
    assert plan.status is PlanStatus.AMBIGUOUS
    assert plan.requested_companies == ()


def test_chinese_broad_financial_performance_is_ambiguous() -> None:
    plan = ROUTER.plan("比較三家公司 2025 財務表現")
    assert plan.route is AnalysisRoute.FINANCIAL
    assert plan.status is PlanStatus.AMBIGUOUS
    assert "unresolved_financial_metric:financial_performance" in plan.reasons


def test_lexical_collision_preserves_current_combined_ambiguity() -> None:
    plan = ROUTER.plan("ASUS AI server revenue strategy.")
    assert plan.route is AnalysisRoute.COMBINED
    assert plan.status is PlanStatus.AMBIGUOUS
    assert plan.qualitative_intents == ("ai_server",)
    assert plan.financial_intents == ("revenue",)


@pytest.mark.parametrize(
    "question,years",
    (
        ("ASUS 2024 revenue", (2024,)),
        ("ASUS 2025 revenue", (2025,)),
        ("ASUS operating margin change from 2024 to 2025", (2024, 2025)),
    ),
)
def test_supported_year_detection(question: str, years: tuple[int, ...]) -> None:
    assert ROUTER.plan(question).fiscal_years == years


def test_financial_question_without_year_is_ambiguous() -> None:
    plan = ROUTER.plan("Compare the companies' financial performance.")
    assert plan.route is AnalysisRoute.FINANCIAL
    assert plan.status is PlanStatus.AMBIGUOUS
    assert "missing_required_year" in plan.reasons


def test_non_comparative_financial_question_without_company_is_ambiguous() -> None:
    plan = ROUTER.plan("What was 2025 revenue?")
    assert plan.route is AnalysisRoute.FINANCIAL
    assert plan.status is PlanStatus.AMBIGUOUS
    assert "missing_required_company" in plan.reasons


def test_broad_financial_performance_requires_metric_resolution() -> None:
    plan = ROUTER.plan("Compare the companies' financial performance in 2025.")
    assert plan.route is AnalysisRoute.FINANCIAL
    assert plan.status is PlanStatus.AMBIGUOUS
    assert "unresolved_financial_metric:financial_performance" in plan.reasons


@pytest.mark.parametrize(
    "question,unsupported",
    (
        ("Compare 2025 ROE.", "roe"),
        ("Compare 2025 EBITDA.", "ebitda"),
        ("Compare quarterly revenue in 2025.", "quarterly"),
        ("Forecast 2025 revenue.", "forecast"),
        ("Compare 2025 stock prices.", "stock_price"),
        ("Compare 2025 free cash flow.", "free_cash_flow"),
        ("Compare 2023 revenue growth.", "year:2023"),
    ),
)
def test_unsupported_financial_requests(question: str, unsupported: str) -> None:
    plan = ROUTER.plan(question)
    assert plan.status is PlanStatus.UNSUPPORTED
    assert unsupported in plan.unsupported_intents
    assert plan.route in {AnalysisRoute.FINANCIAL, AnalysisRoute.COMBINED}


def test_unrecognized_question_is_ambiguous() -> None:
    plan = ROUTER.plan("Tell me something interesting.")
    assert plan.route is None
    assert plan.status is PlanStatus.AMBIGUOUS
    assert plan.reasons == ("unresolved_intent",)


def test_qualitative_year_does_not_create_financial_route() -> None:
    plan = ROUTER.plan("What was ASUS's AI PC strategy in 2025?")
    assert plan.route is AnalysisRoute.QUALITATIVE
    assert plan.fiscal_years == (2025,)


def test_reason_codes_and_repeated_plans_are_deterministic() -> None:
    question = "Compare ASUS and MSI's AI strategies and 2025 revenue growth."
    first = ROUTER.plan(question)
    second = ROUTER.plan(question)
    assert first == second
    assert first.requested_companies == ("asus", "msi")
    assert first.reasons == (
        "detected_company:asus",
        "detected_company:msi",
        "detected_year:2025",
        "detected_qualitative_intent:ai_strategy",
        "detected_financial_metric:revenue_yoy_growth",
        "both_intent_families_present",
    )


@pytest.mark.parametrize("question", ("", "   ", None))
def test_empty_or_invalid_question_is_rejected(question: object) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ROUTER.plan(question)
