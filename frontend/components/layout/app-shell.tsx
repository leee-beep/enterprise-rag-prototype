"use client";

import { useEffect, type ReactNode } from "react";
import type { UiBackendStatus } from "@/types/api";
import type { CompanyOption, EvidenceViewModel, TopicShortcut } from "@/types/presentation";
import { EvidencePanel } from "./evidence-panel";
import { Sidebar, type HistoryEntry } from "./sidebar";
import { TopBar } from "./top-bar";

export function AppShell(props: {
  children: ReactNode;
  status: UiBackendStatus;
  companies: CompanyOption[];
  topics: TopicShortcut[];
  selectedCompanies: string[];
  history: HistoryEntry[];
  evidence: EvidenceViewModel[];
  selectedEvidenceId: string | null;
  menuOpen: boolean;
  evidenceOpen: boolean;
  onRetryReadiness: () => void;
  onMenu: () => void;
  onEvidence: () => void;
  onCloseDrawers: () => void;
  onClearHistory: () => void;
  onCompany: (id: string) => void;
  onTopic: (query: string) => void;
  onSelectEvidence: (id: string) => void;
}) {
  const { menuOpen, evidenceOpen, onCloseDrawers } = props;
  useEffect(() => {
    if (!menuOpen && !evidenceOpen) return;
    const opener = menuOpen ? "Open research navigation" : "Open evidence panel";
    requestAnimationFrame(() => document.querySelector<HTMLElement>(".drawer-open .drawer-close")?.focus());
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key !== "Escape") return;
      onCloseDrawers();
      requestAnimationFrame(() => document.querySelector<HTMLElement>(`button[aria-label="${opener}"]`)?.focus());
    }
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [menuOpen, evidenceOpen, onCloseDrawers]);

  return (
    <main className="workspace-shell">
      <TopBar status={props.status} onRetry={props.onRetryReadiness} onMenu={props.onMenu} onEvidence={props.onEvidence} />
      <Sidebar companies={props.companies} topics={props.topics} selected={props.selectedCompanies} history={props.history} open={props.menuOpen} onClose={props.onCloseDrawers} onClearHistory={props.onClearHistory} onCompany={props.onCompany} onTopic={props.onTopic} />
      {props.children}
      <EvidencePanel evidence={props.evidence} selectedId={props.selectedEvidenceId} open={props.evidenceOpen} onClose={props.onCloseDrawers} onSelect={props.onSelectEvidence} />
      {(props.menuOpen || props.evidenceOpen) && <button className="drawer-scrim" aria-label="Close open panel" onClick={props.onCloseDrawers} />}
    </main>
  );
}
