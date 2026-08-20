import { RefreshCw } from "lucide-react";
import type { UiBackendStatus } from "@/types/api";
import { useI18n } from "@/lib/i18n-context";

export function StatusBadge({ status = "unknown", onRetry }: { status?: UiBackendStatus; onRetry?: () => void }) {
  const { t } = useI18n();
  const labels: Record<UiBackendStatus, ReturnType<typeof t>> = { ready: t("ready"), loading: t("connecting"), unavailable: t("unavailable"), unknown: t("unknown") };
  return <div className={`status-badge status-${status}`} role="status" aria-live="polite"><span className="status-dot" aria-hidden="true" /><span>{labels[status]}</span>{status === "unavailable" && onRetry && <button type="button" onClick={onRetry} aria-label={t("retry")}><RefreshCw size={12} /></button>}</div>;
}
