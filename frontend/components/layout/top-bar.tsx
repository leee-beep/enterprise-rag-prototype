import { Menu, PanelRightOpen } from "lucide-react";
import type { UiBackendStatus } from "@/types/api";
import { StatusBadge } from "@/components/status/status-badge";
export function TopBar({ status, onMenu, onEvidence }: { status: UiBackendStatus; onMenu: () => void; onEvidence: () => void }) {
  return <header className="topbar"><button className="mobile-icon" type="button" onClick={onMenu} aria-label="Open navigation"><Menu size={19} /></button><div className="topbar-title"><p className="eyebrow">Local intelligence system</p><h1>Competitor Intelligence Workspace</h1></div><div className="topbar-actions"><StatusBadge status={status} /><button className="mobile-icon" type="button" onClick={onEvidence} aria-label="Open evidence"><PanelRightOpen size={19} /></button></div></header>;
}
