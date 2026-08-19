import { claimRoleLabel, claimTypeLabel, safeReasonLabel } from "@/lib/display-labels";

const financialLabels = [
  claimTypeLabel("reported_fact"),
  claimTypeLabel("calculated_metric"),
  claimTypeLabel("comparison_entry"),
  claimTypeLabel("financial_change_value"),
  claimTypeLabel("unknown_claim"),
] satisfies string[];

const roleLabels = [
  claimRoleLabel("earlier_value"),
  claimRoleLabel("later_value"),
  claimRoleLabel("percentage_point_change"),
  claimRoleLabel("unknown_role"),
] satisfies string[];

const safeReasons = [
  safeReasonLabel("missing_required_company"),
  safeReasonLabel("qualitative_evidence_unavailable:msi"),
  safeReasonLabel("unknown_internal_reason"),
] satisfies string[];

void financialLabels;
void roleLabels;
void safeReasons;
