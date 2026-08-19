"use client";

import { useEffect, useRef, useState } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { ResearchWorkspace } from "@/components/research/research-workspace";
import { ApiClientError, competitorApi } from "@/lib/api";
import { mapAnalyzeResponse } from "@/lib/mappers";
import { companies, previewPrompts, topics } from "@/lib/mock-data";
import { useSessionHistory } from "@/hooks/use-session-history";
import type { UiBackendStatus, UiOperationState } from "@/types/api";
import type { WorkspaceResultViewModel } from "@/types/presentation";

export default function WorkspaceClient() {
  const [query, setQuery] = useState("");
  const [selectedCompanies, setSelectedCompanies] = useState(["asus", "gigabyte", "msi"]);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<UiBackendStatus>("loading");
  const [operation, setOperation] = useState<UiOperationState>("idle");
  const [result, setResult] = useState<WorkspaceResultViewModel | null>(null);
  const { history, addHistory } = useSessionHistory();
  const [menuOpen, setMenuOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const readinessRequested = useRef(false);

  useEffect(() => {
    if (readinessRequested.current) return;
    readinessRequested.current = true;
    competitorApi.readiness().then(() => setBackendStatus("ready")).catch(() => setBackendStatus("unavailable"));
  }, []);

  function selectCompany(id: string) {
    setSelectedCompanies((current) => current.includes(id) ? (current.length > 1 ? current.filter((item) => item !== id) : current) : [...current, id]);
  }

  function selectEvidence(id: string) {
    setSelectedEvidenceId(id);
    setEvidenceOpen(true);
  }

  async function analyze() {
    const question = query.trim();
    if (!question || operation === "submitting") return;
    setOperation("submitting");
    setSelectedEvidenceId(null);
    try {
      const mapped = mapAnalyzeResponse(await competitorApi.analyzeCompetitor(question));
      setResult(mapped);
      addHistory(question, mapped.status);
      setOperation("idle");
    } catch (error) {
      setResult(null);
      setOperation(error instanceof ApiClientError && (error.status === 503 || error.status === null) ? "unavailable" : "server_failure");
    }
  }

  function closeDrawers() {
    setMenuOpen(false);
    setEvidenceOpen(false);
  }

  return (
    <AppShell status={backendStatus} companies={companies} topics={topics} selectedCompanies={selectedCompanies} history={history} evidence={result?.evidence ?? []} selectedEvidenceId={selectedEvidenceId} menuOpen={menuOpen} evidenceOpen={evidenceOpen} onMenu={() => setMenuOpen(true)} onEvidence={() => setEvidenceOpen(true)} onCloseDrawers={closeDrawers} onCompany={selectCompany} onTopic={(value) => { setQuery(value); closeDrawers(); }} onSelectEvidence={selectEvidence}>
      <ResearchWorkspace query={query} operation={operation} result={result} prompts={previewPrompts} onQuery={setQuery} onAnalyze={analyze} onEvidence={selectEvidence} />
    </AppShell>
  );
}
