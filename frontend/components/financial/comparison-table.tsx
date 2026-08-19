import type { FinancialDisplayItem } from "@/types/presentation";

export function ComparisonTable({ items }: { items: FinancialDisplayItem[] }) {
  return (
    <div className="table-wrap">
      <table>
        <caption className="sr-only">Synthetic operating margin comparison</caption>
        <thead>
          <tr><th>Company</th><th>FY</th><th>Metric</th><th>Value</th><th>Rank</th></tr>
        </thead>
        <tbody>
          {items.map((item) => (
            <tr key={item.company}>
              <td>{item.company}</td>
              <td>{item.fiscalYear}</td>
              <td>{item.metric}</td>
              <td>{item.displayValue}</td>
              <td>{item.rankLabel}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
