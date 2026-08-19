import type { FinancialDisplayItem } from "@/types/presentation";

export function ComparisonTable({ items }: { items: FinancialDisplayItem[] }) {
  return <div className="table-wrap"><table><caption className="sr-only">Validated financial claims</caption><thead><tr><th>Company</th><th>Evidence</th><th>Claim</th><th>Value</th><th>Rank</th></tr></thead><tbody>{items.map((item) => <tr key={item.evidenceId}><td>{item.companyId ?? "—"}</td><td>{item.evidenceId}</td><td>{item.claimType}</td><td>{item.displayValue}</td><td>{item.rankLabel ?? "—"}</td></tr>)}</tbody></table></div>;
}
