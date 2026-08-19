import type { DomainStatus } from "@/types/api";

const copy: Record<DomainStatus, { label: string; detail: string }> = {
  completed: { label: "Completed", detail: "Grounded analysis is ready for review." },
  partial: { label: "Partial coverage", detail: "The available result is usable, with evidence limitations noted below." },
  ambiguous: { label: "Needs refinement", detail: "Make the company, year, or analysis goal more specific." },
  unsupported: { label: "Outside scope", detail: "This request is outside the workspace's current analysis capabilities." },
  insufficient: { label: "Insufficient evidence", detail: "There is not enough trusted evidence for a grounded conclusion." },
};

export function DomainState({ status }: { status: DomainStatus }) {
  const state = copy[status];
  return <div className={`domain-state domain-${status}`} role="status"><span className="state-mark" aria-hidden="true" /><div><strong>{state.label}</strong><span>{state.detail}</span></div></div>;
}
