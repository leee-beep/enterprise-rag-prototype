import { RotateCcw } from "lucide-react";
import type { UiOperationState } from "@/types/api";
import { useI18n } from "@/lib/i18n-context";

export function OperationState({ state, onRetry }: { state: UiOperationState; onRetry?: () => void }) {
  const { t } = useI18n();
  const copy: Record<UiOperationState, { label: string; detail: string }> = {
    idle: { label: t("ready"), detail: t("researchQuestion") }, submitting: { label: t("analyzing"), detail: t("buildingBrief") }, unavailable: { label: t("serviceUnavailable"), detail: t("serviceUnavailableDetail") }, server_failure: { label: t("interrupted"), detail: t("interruptedDetail") },
  };
  const content = copy[state];
  return <div className={`domain-state operation-${state}`} role="alert"><span className="state-mark" aria-hidden="true" /><div><strong>{content.label}</strong><span>{content.detail}</span></div>{onRetry && <button type="button" onClick={onRetry}><RotateCcw size={14} /> {t("retry")}</button>}</div>;
}
