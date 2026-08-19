import { RotateCcw } from "lucide-react";
import type { UiOperationState } from "@/types/api";

const copy: Record<UiOperationState, { label: string; detail: string }> = {
  idle: { label: "Ready", detail: "Enter a research question to begin." },
  submitting: { label: "Analyzing", detail: "Preparing a grounded response." },
  unavailable: { label: "Local service unavailable", detail: "Check the local API and model service, then retry when ready." },
  server_failure: { label: "Analysis interrupted", detail: "The request could not be completed. Your question has been retained." },
};

export function OperationState({ state, onRetry }: { state: UiOperationState; onRetry?: () => void }) {
  const content = copy[state];
  return <div className={`domain-state operation-${state}`} role="alert"><span className="state-mark" aria-hidden="true" /><div><strong>{content.label}</strong><span>{content.detail}</span></div>{onRetry && <button type="button" onClick={onRetry}><RotateCcw size={14} /> Retry</button>}</div>;
}
