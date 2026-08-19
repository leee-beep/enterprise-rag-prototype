"use client";

import { useState } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { ResearchWorkspace } from "@/components/research/research-workspace";
import {
  companies,
  previewEvidence,
  previewFinancialItems,
  previewPrompts,
  syntheticMarginChartData,
  topics,
} from "@/lib/mock-data";
import { useSessionHistory } from "@/hooks/use-session-history";

export default function WorkspaceClient() {
  const [query, setQuery] = useState("");
  const [selectedCompanies, setSelectedCompanies] = useState([
    "asus",
    "gigabyte",
    "msi",
  ]);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const [view, setView] = useState<"empty" | "loading" | "preview">("empty");
  const { history, addHistory } = useSessionHistory();
  const [menuOpen, setMenuOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);

  function selectCompany(id: string) {
    setSelectedCompanies((current) =>
      current.includes(id)
        ? current.length > 1
          ? current.filter((item) => item !== id)
          : current
        : [...current, id],
    );
  }

  function selectEvidence(id: string) {
    setSelectedEvidenceId(id);
    setEvidenceOpen(true);
  }

  function analyzePreview() {
    if (!query.trim()) return;
    setView("loading");
    window.setTimeout(() => {
      addHistory(query.trim());
      setView("preview");
    }, 550);
  }

  function closeDrawers() {
    setMenuOpen(false);
    setEvidenceOpen(false);
  }

  return (
    <AppShell
      status="unknown"
      companies={companies}
      topics={topics}
      selectedCompanies={selectedCompanies}
      history={history}
      evidence={view === "preview" ? previewEvidence : []}
      selectedEvidenceId={selectedEvidenceId}
      menuOpen={menuOpen}
      evidenceOpen={evidenceOpen}
      onMenu={() => setMenuOpen(true)}
      onEvidence={() => setEvidenceOpen(true)}
      onCloseDrawers={closeDrawers}
      onCompany={selectCompany}
      onTopic={(value) => {
        setQuery(value);
        closeDrawers();
      }}
      onSelectEvidence={selectEvidence}
    >
      <ResearchWorkspace
        query={query}
        view={view}
        prompts={previewPrompts}
        evidence={previewEvidence}
        financialItems={previewFinancialItems}
        chartData={syntheticMarginChartData}
        onQuery={setQuery}
        onAnalyze={analyzePreview}
        onEvidence={selectEvidence}
      />
    </AppShell>
  );
}
