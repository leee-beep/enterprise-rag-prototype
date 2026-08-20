import type { DomainStatus } from "@/types/api";
import { useI18n } from "@/lib/i18n-context";

export function DomainState({ status, compact = false }: { status: DomainStatus; compact?: boolean }) {
  const { t } = useI18n();
  const copy: Record<DomainStatus, { label: string; detail: string }> = {
    completed: { label: t("completed"), detail: t("completedDetail") }, partial: { label: t("partial"), detail: t("partialDetail") }, ambiguous: { label: t("ambiguous"), detail: t("ambiguousDetail") }, unsupported: { label: t("unsupported"), detail: t("unsupportedDetail") }, insufficient: { label: t("insufficient"), detail: t("insufficientDetail") },
  };
  const state = copy[status];
  return <div className={`domain-state domain-${status} ${compact ? "domain-compact" : ""}`} role="status"><span className="state-mark" aria-hidden="true" /><div><strong>{state.label}</strong>{!compact && <span>{state.detail}</span>}</div></div>;
}
