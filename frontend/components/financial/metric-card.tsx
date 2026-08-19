import type { FinancialDisplayItem } from "@/types/presentation";

export function MetricCard({ item }: { item: FinancialDisplayItem }) {
  return <article className="metric-card"><div><span>{item.companyId ?? "Unspecified company"}</span><small>{item.evidenceId}</small></div><strong>{item.displayValue}</strong><p>{item.claimType} · {item.role}</p>{item.rankLabel && <em>Rank {item.rankLabel}</em>}</article>;
}
