import type { ReactNode } from "react";
import type { UiBackendStatus } from "@/types/api";
import type {
  CompanyOption,
  EvidenceViewModel,
  TopicShortcut,
} from "@/types/presentation";
import { TopBar } from "./top-bar";
import { Sidebar, type HistoryEntry } from "./sidebar";
import { EvidencePanel } from "./evidence-panel";

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
  onMenu: () => void;
  onEvidence: () => void;
  onCloseDrawers: () => void;
  onCompany: (id: string) => void;
  onTopic: (query: string) => void;
  onSelectEvidence: (id: string) => void;
}) {
  return (
    <main className="workspace-shell">
      <TopBar
        status={props.status}
        onMenu={props.onMenu}
        onEvidence={props.onEvidence}
      />
      <Sidebar
        companies={props.companies}
        topics={props.topics}
        selected={props.selectedCompanies}
        history={props.history}
        open={props.menuOpen}
        onClose={props.onCloseDrawers}
        onCompany={props.onCompany}
        onTopic={props.onTopic}
      />
      {props.children}
      <EvidencePanel
        evidence={props.evidence}
        selectedId={props.selectedEvidenceId}
        open={props.evidenceOpen}
        onClose={props.onCloseDrawers}
        onSelect={props.onSelectEvidence}
      />
      {(props.menuOpen || props.evidenceOpen) && (
        <button
          className="drawer-scrim"
          aria-label="Close panel"
          onClick={props.onCloseDrawers}
        />
      )}
    </main>
  );
}
