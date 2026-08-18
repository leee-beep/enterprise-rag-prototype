"""Deterministic, auditable structured financial facts and CSV loading."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable


COMPANIES = {
    "gigabyte": ("Gigabyte", "2376"),
    "asus": ("ASUS", "2357"),
    "msi": ("MSI", "2377"),
}
SUPPORTED_FISCAL_YEARS = frozenset({2024, 2025})
SUPPORTED_METRICS = frozenset(
    {"revenue", "gross_profit", "operating_income", "net_income", "eps"}
)
SUPPORTED_PERIODS = frozenset({"FY"})
SUPPORTED_REPORTING_SCOPES = frozenset({"consolidated"})
SUPPORTED_DOCUMENT_TYPES = frozenset({"consolidated_financial_report"})
SUPPORTED_CURRENCIES = frozenset({"TWD"})
UNIT_MULTIPLIERS = {
    "TWD": Decimal("1"),
    "thousand_TWD": Decimal("1000"),
    "million_TWD": Decimal("1000000"),
}

REQUIRED_CSV_COLUMNS = (
    "company_id",
    "company_name",
    "ticker",
    "fiscal_year",
    "period",
    "metric",
    "value",
    "currency",
    "unit",
    "reporting_scope",
    "document_type",
    "source_document_id",
    "source_title",
    "page_number",
)
OPTIONAL_CSV_COLUMNS = ("source_label", "source_sha256", "notes")
CSV_COLUMNS = REQUIRED_CSV_COLUMNS + OPTIONAL_CSV_COLUMNS

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class FinancialFactValidationError(ValueError):
    """Raised when a structured financial fact or dataset row is invalid."""


class FinancialFactConflictError(FinancialFactValidationError):
    """Raised when one logical fact key has conflicting canonical values."""


class FinancialFactNotFoundError(LookupError):
    """Raised when an exact requested fact is absent from a collection."""


def normalize_to_twd(value: Decimal | str | int, unit: str) -> Decimal:
    """Normalize a finite source value to canonical TWD using exact decimals."""
    number = _decimal_value("value", value)
    normalized_unit = _required_text("unit", unit)
    try:
        multiplier = UNIT_MULTIPLIERS[normalized_unit]
    except KeyError as exc:
        allowed = ", ".join(UNIT_MULTIPLIERS)
        raise FinancialFactValidationError(
            f"unit must be one of: {allowed}."
        ) from exc
    return number * multiplier


@dataclass(frozen=True)
class FinancialFact:
    """One source financial value with canonical TWD normalization.

    ``value`` and ``unit`` preserve the report's presentation. Calculations must
    use ``canonical_value_twd``, which is derived with :class:`Decimal` and is
    therefore independent of binary floating-point rounding.
    """

    company_id: str
    company_name: str
    ticker: str
    fiscal_year: int
    period: str
    metric: str
    value: Decimal
    currency: str
    unit: str
    reporting_scope: str
    document_type: str
    source_document_id: str
    source_title: str
    page_number: int
    source_label: str | None = None
    source_sha256: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        company_id = _required_text("company_id", self.company_id).casefold()
        if company_id not in COMPANIES:
            raise FinancialFactValidationError(
                "company_id must be one of: asus, gigabyte, msi."
            )
        expected_name, expected_ticker = COMPANIES[company_id]
        company_name = _required_text("company_name", self.company_name)
        ticker = _required_text("ticker", str(self.ticker))
        if company_name != expected_name or ticker != expected_ticker:
            raise FinancialFactValidationError(
                f"company_name or ticker is inconsistent with company_id {company_id!r}."
            )
        if isinstance(self.fiscal_year, bool) or self.fiscal_year not in SUPPORTED_FISCAL_YEARS:
            raise FinancialFactValidationError("fiscal_year must be 2024 or 2025.")
        period = _choice("period", self.period.upper(), SUPPORTED_PERIODS)
        metric = _choice("metric", self.metric.casefold(), SUPPORTED_METRICS)
        currency = _choice("currency", self.currency.upper(), SUPPORTED_CURRENCIES)
        unit = _choice("unit", self.unit, frozenset(UNIT_MULTIPLIERS))
        scope = _choice(
            "reporting_scope", self.reporting_scope.casefold(), SUPPORTED_REPORTING_SCOPES
        )
        document_type = _choice(
            "document_type", self.document_type.casefold(), SUPPORTED_DOCUMENT_TYPES
        )
        value = _decimal_value("value", self.value)
        if metric == "eps" and unit != "TWD":
            raise FinancialFactValidationError(
                "eps must use unit TWD in the initial financial fact vocabulary."
            )
        if isinstance(self.page_number, bool) or not isinstance(self.page_number, int) or self.page_number <= 0:
            raise FinancialFactValidationError("page_number must be a positive integer.")
        source_document_id = _safe_identity(self.source_document_id)
        source_sha256 = _optional_text(self.source_sha256)
        if source_sha256 is not None:
            source_sha256 = source_sha256.casefold()
            if not _SHA256_PATTERN.fullmatch(source_sha256):
                raise FinancialFactValidationError(
                    "source_sha256 must be a 64-character hexadecimal digest."
                )

        object.__setattr__(self, "company_id", company_id)
        object.__setattr__(self, "company_name", company_name)
        object.__setattr__(self, "ticker", ticker)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "currency", currency)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "reporting_scope", scope)
        object.__setattr__(self, "document_type", document_type)
        object.__setattr__(self, "source_document_id", source_document_id)
        object.__setattr__(self, "source_title", _safe_title(self.source_title))
        object.__setattr__(self, "source_label", _optional_text(self.source_label))
        object.__setattr__(self, "source_sha256", source_sha256)
        object.__setattr__(self, "notes", _optional_text(self.notes))

    @property
    def canonical_value_twd(self) -> Decimal:
        """Return the exact value normalized to the canonical TWD base unit."""
        return normalize_to_twd(self.value, self.unit)

    @property
    def logical_key(self) -> tuple[str, int, str, str, str]:
        """Return the unique first-version fact identity."""
        return (
            self.company_id,
            self.fiscal_year,
            self.period,
            self.metric,
            self.reporting_scope,
        )


class FinancialFactCollection:
    """An immutable, deterministically ordered set of canonical source facts."""

    def __init__(self, facts: Iterable[FinancialFact]) -> None:
        unique: dict[tuple[str, int, str, str, str], FinancialFact] = {}
        for fact in facts:
            if not isinstance(fact, FinancialFact):
                raise FinancialFactValidationError(
                    "FinancialFactCollection accepts only FinancialFact values."
                )
            existing = unique.get(fact.logical_key)
            if existing is None:
                unique[fact.logical_key] = fact
            elif existing.canonical_value_twd != fact.canonical_value_twd:
                raise FinancialFactConflictError(
                    "Conflicting values found for financial fact "
                    f"{_display_key(fact.logical_key)}."
                )
            # Exact normalized duplicates are intentionally represented once.
        self._facts = tuple(unique[key] for key in sorted(unique))
        self._by_key = {fact.logical_key: fact for fact in self._facts}

    def __iter__(self):
        return iter(self._facts)

    def __len__(self) -> int:
        return len(self._facts)

    @property
    def facts(self) -> tuple[FinancialFact, ...]:
        return self._facts

    def get_fact(
        self,
        company_id: str,
        fiscal_year: int,
        metric: str,
        period: str = "FY",
        reporting_scope: str = "consolidated",
    ) -> FinancialFact:
        key = (
            company_id.strip().casefold(),
            fiscal_year,
            period.strip().upper(),
            metric.strip().casefold(),
            reporting_scope.strip().casefold(),
        )
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise FinancialFactNotFoundError(
                f"Financial fact was not found for {_display_key(key)}."
            ) from exc

    def list_company_facts(
        self, company_id: str, fiscal_year: int | None = None
    ) -> tuple[FinancialFact, ...]:
        normalized = company_id.strip().casefold()
        return tuple(
            fact
            for fact in self._facts
            if fact.company_id == normalized
            and (fiscal_year is None or fact.fiscal_year == fiscal_year)
        )

    def list_metric_facts(self, metric: str) -> tuple[FinancialFact, ...]:
        normalized = metric.strip().casefold()
        return tuple(fact for fact in self._facts if fact.metric == normalized)


def load_financial_facts_csv(path: str | Path) -> FinancialFactCollection:
    """Load and validate a private or synthetic curated financial-facts CSV."""
    candidate = Path(path)
    safe_name = candidate.name or "financial facts dataset"
    if not candidate.exists() or not candidate.is_file():
        raise FinancialFactValidationError(f"Financial facts file is unavailable: {safe_name}.")
    try:
        with candidate.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = tuple(reader.fieldnames or ())
            missing = [name for name in REQUIRED_CSV_COLUMNS if name not in columns]
            unknown = [name for name in columns if name not in CSV_COLUMNS]
            if missing or unknown:
                details = []
                if missing:
                    details.append("missing columns: " + ", ".join(missing))
                if unknown:
                    details.append("unknown columns: " + ", ".join(unknown))
                raise FinancialFactValidationError(
                    "Invalid financial facts CSV schema (" + "; ".join(details) + ")."
                )
            facts = [_fact_from_row(row, number) for number, row in enumerate(reader, start=2)]
    except FinancialFactValidationError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise FinancialFactValidationError(
            f"Could not read financial facts file: {safe_name}."
        ) from exc
    return FinancialFactCollection(facts)


def _fact_from_row(row: dict[str, str | None], row_number: int) -> FinancialFact:
    try:
        fiscal_year = _integer_value("fiscal_year", row.get("fiscal_year"))
        page_number = _integer_value("page_number", row.get("page_number"))
        return FinancialFact(
            company_id=row.get("company_id") or "",
            company_name=row.get("company_name") or "",
            ticker=row.get("ticker") or "",
            fiscal_year=fiscal_year,
            period=row.get("period") or "",
            metric=row.get("metric") or "",
            value=_decimal_value("value", row.get("value")),
            currency=row.get("currency") or "",
            unit=row.get("unit") or "",
            reporting_scope=row.get("reporting_scope") or "",
            document_type=row.get("document_type") or "",
            source_document_id=row.get("source_document_id") or "",
            source_title=row.get("source_title") or "",
            page_number=page_number,
            source_label=row.get("source_label"),
            source_sha256=row.get("source_sha256"),
            notes=row.get("notes"),
        )
    except FinancialFactValidationError as exc:
        raise FinancialFactValidationError(
            f"Invalid financial fact at CSV row {row_number}: {exc}"
        ) from exc


def _decimal_value(name: str, value: object) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise FinancialFactValidationError(f"{name} must be a finite decimal number.")
    try:
        number = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise FinancialFactValidationError(
            f"{name} must be a finite decimal number."
        ) from exc
    if not number.is_finite():
        raise FinancialFactValidationError(f"{name} must be a finite decimal number.")
    return number


def _integer_value(name: str, value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise FinancialFactValidationError(f"{name} must be an integer.")
    text = str(value).strip()
    try:
        number = int(text)
    except ValueError as exc:
        raise FinancialFactValidationError(f"{name} must be an integer.") from exc
    if str(number) != text:
        raise FinancialFactValidationError(f"{name} must be an integer.")
    return number


def _choice(name: str, value: str, supported: frozenset[str]) -> str:
    normalized = _required_text(name, value)
    if normalized not in supported:
        raise FinancialFactValidationError(
            f"{name} must be one of: {', '.join(sorted(supported))}."
        )
    return normalized


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinancialFactValidationError(f"{name} must be a non-empty string.")
    return value.strip()


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FinancialFactValidationError("Optional text fields must be strings.")
    return value.strip() or None


def _safe_identity(value: object) -> str:
    identity = _required_text("source_document_id", value)
    if "/" in identity or "\\" in identity or re.match(r"^[A-Za-z]:", identity):
        raise FinancialFactValidationError(
            "source_document_id must be a safe identity, not a local path."
        )
    return identity


def _safe_title(value: object) -> str:
    title = _required_text("source_title", value)
    if PureWindowsPath(title).is_absolute() or PurePosixPath(title).is_absolute():
        raise FinancialFactValidationError(
            "source_title must be a safe display title, not a local absolute path."
        )
    return title


def _display_key(key: tuple[object, ...]) -> str:
    return "/".join(str(part) for part in key)
