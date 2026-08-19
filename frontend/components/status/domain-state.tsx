import type { DomainStatus } from "@/types/api";
const copy: Record<DomainStatus, string> = {
  completed: "Analysis completed",
  partial: "Partial company coverage",
  ambiguous: "Refine the question for a more precise comparison",
  unsupported: "This request is outside the supported analysis scope",
  insufficient: "Available evidence is insufficient for a grounded conclusion",
};
export function DomainState({ status }: { status: DomainStatus }) {
  return <div className={`domain-state domain-${status}`} role="status"><strong>{status.replace("_", " ")}</strong><span>{copy[status]}</span></div>;
}
