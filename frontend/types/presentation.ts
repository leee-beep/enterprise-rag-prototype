export type EvidenceKind = "qualitative" | "financial";

export interface CompanyOption {
  id: string;
  name: string;
  ticker: string;
}

export interface TopicShortcut {
  label: string;
  query: string;
}

export interface EvidenceViewModel {
  id: string;
  kind: EvidenceKind;
  companyId: string | null;
  companyName: string | null;
  fiscalYear: string | null;
  sources: EvidenceSourceViewModel[];
}

export interface EvidenceSourceViewModel {
  sourceTitle: string;
  pageNumber: string | null;
  sourceMetric: string | null;
  fiscalYear: string | null;
  documentType: string | null;
}

export interface FinancialDisplayItem {
  evidenceId: string;
  companyId: string | null;
  claimType: string;
  role: string;
  displayValue: string;
  rankLabel: string | null;
}

export interface WorkspaceResultViewModel {
  question: string;
  status: import("./api").DomainStatus;
  answerText: string | null;
  reasons: string[];
  evidence: EvidenceViewModel[];
  financialItems: FinancialDisplayItem[];
  generation: { provider: string | null; model: string | null } | null;
}

// Numeric values are permitted only for the isolated synthetic preview chart.
export interface SyntheticChartDatum {
  company: string;
  value: number;
}
