const claimLabels: Record<string, string> = {
  reported_fact: "Reported Fact",
  calculated_metric: "Calculated Metric",
  comparison_entry: "Comparison Entry",
  financial_change_value: "Financial Change",
};

const roleLabels: Record<string, string> = {
  earlier_value: "Earlier value",
  later_value: "Later value",
  percentage_point_change: "Percentage-point change",
  calculated_value: "Calculated value",
  reported_value: "Reported value",
};

const reasonLabels = [
  ["qualitative_evidence_unavailable", "reasonQualitativeUnavailable"],
  ["missing_required_company", "reasonCompany"],
  ["missing_required_year", "reasonYear"],
  ["insufficient_companies_for_comparison", "reasonCompanies"],
  ["unresolved_financial_metric", "reasonMetric"],
  ["unresolved_intent", "reasonIntent"],
] as const;

export function claimTypeLabel(value: string): string {
  return claimLabels[value] ?? "Validated Financial Claim";
}

export function claimRoleLabel(value: string): string {
  return roleLabels[value] ?? "Evidence value";
}

export function safeReasonLabel(value: string, locale: Locale = "en"): string {
  const key = reasonLabels.find(([prefix]) => value === prefix || value.startsWith(`${prefix}:`))?.[1];
  return translate(locale, key ?? "reasonFallback");
}
import { translate, type Locale } from "./i18n";
