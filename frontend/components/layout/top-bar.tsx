import { Menu, PanelRightOpen } from "lucide-react";
import type { UiBackendStatus } from "@/types/api";
import { StatusBadge } from "@/components/status/status-badge";

export function TopBar({ status, onRetry, onMenu, onEvidence }: {
  status: UiBackendStatus;
  onRetry: () => void;
  onMenu: () => void;
  onEvidence: () => void;
}) {
  return <header className="topbar"><button className="mobile-icon" type="button" onClick={onMenu} aria-label="Open research navigation"><Menu size={19} /></button><div className="topbar-title"><p className="eyebrow">Local research environment</p><h1>Competitor Intelligence</h1></div><div className="topbar-actions"><span className="privacy-label">Private · Local</span><StatusBadge status={status} onRetry={onRetry} /><button className="mobile-icon" type="button" onClick={onEvidence} aria-label="Open evidence panel"><PanelRightOpen size={19} /></button></div></header>;
}
