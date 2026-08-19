import { RefreshCw } from "lucide-react";
import type { UiBackendStatus } from "@/types/api";

const labels: Record<UiBackendStatus, string> = {
  ready: "Ready",
  loading: "Connecting",
  unavailable: "Unavailable",
  unknown: "Unknown",
};

export function StatusBadge({ status = "unknown", onRetry }: { status?: UiBackendStatus; onRetry?: () => void }) {
  return <div className={`status-badge status-${status}`} role="status" aria-live="polite"><span className="status-dot" aria-hidden="true" /><span>{labels[status]}</span>{status === "unavailable" && onRetry && <button type="button" onClick={onRetry} aria-label="Retry backend readiness"><RefreshCw size={12} /></button>}</div>;
}
