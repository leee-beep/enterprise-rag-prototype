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
  company: string;
  ticker: string;
  year: number;
  page: number;
  title: string;
  excerpt: string;
}

export interface FinancialDisplayItem {
  company: string;
  ticker: string;
  fiscalYear: string;
  metric: string;
  displayValue: string;
  rankLabel: string;
  tied: boolean;
}

// Numeric values are permitted only for the isolated synthetic preview chart.
export interface SyntheticChartDatum {
  company: string;
  value: number;
}
