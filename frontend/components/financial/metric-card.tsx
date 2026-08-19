import type { FinancialDisplayItem } from "@/types/presentation";

export function MetricCard({ item }: { item: FinancialDisplayItem }) {
  return (
    <article className="metric-card">
      <div><span>{item.company}</span><small>{item.ticker}</small></div>
      <strong>{item.displayValue}</strong>
      <p>Synthetic {item.metric.toLowerCase()}</p>
      {item.tied && <em>Tied rank · {item.rankLabel}</em>}
    </article>
  );
}
