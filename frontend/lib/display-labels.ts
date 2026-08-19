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

const reasonLabels: Array<[string, string]> = [
  ["qualitative_evidence_unavailable", "Trusted qualitative evidence is unavailable for part of this request."],
  ["missing_required_company", "Name the company or companies you want to analyze."],
  ["missing_required_year", "Add a fiscal year to make the request precise."],
  ["insufficient_companies_for_comparison", "Choose at least two companies for a comparison."],
  ["unresolved_financial_metric", "Specify the financial metric you want to compare."],
  ["unresolved_intent", "Clarify the type of competitor analysis you need."],
];

export function claimTypeLabel(value: string): string {
  return claimLabels[value] ?? "Validated Financial Claim";
}

export function claimRoleLabel(value: string): string {
  return roleLabels[value] ?? "Evidence value";
}

export function safeReasonLabel(value: string): string {
  return reasonLabels.find(([prefix]) => value === prefix || value.startsWith(`${prefix}:`))?.[1]
    ?? "Additional trusted evidence or a more specific question may be required.";
}
