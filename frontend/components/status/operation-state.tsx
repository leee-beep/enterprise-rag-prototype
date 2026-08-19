import type { UiOperationState } from "@/types/api";

const copy: Record<UiOperationState, string> = {
  idle: "Ready for a research question",
  loading: "Preparing grounded analysis",
  unavailable: "The local analysis API is unavailable",
  generation_failure: "Evidence was found, but synthesis could not be completed",
};

export function OperationState({ state }: { state: UiOperationState }) {
  return (
    <div className={`domain-state operation-${state}`} role="status">
      <strong>{state.replace("_", " ")}</strong>
      <span>{copy[state]}</span>
    </div>
  );
}
