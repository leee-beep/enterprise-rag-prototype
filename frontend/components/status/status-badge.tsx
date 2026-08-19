import type { UiBackendStatus } from "@/types/api";
const labels: Record<UiBackendStatus, string> = { ready: "Backend ready", loading: "Checking backend", unavailable: "Backend unavailable", unknown: "Backend unknown" };
export function StatusBadge({ status = "unknown" }: { status?: UiBackendStatus }) {
  return <span className={`status-badge status-${status}`}><span aria-hidden="true" />{labels[status]}</span>;
}
