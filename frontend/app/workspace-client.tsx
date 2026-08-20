"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AppShell } from "@/components/layout/app-shell";
import { ResearchWorkspace } from "@/components/research/research-workspace";
import { ApiClientError, competitorApi } from "@/lib/api";
import { mapAnalyzeResponse } from "@/lib/mappers";
import { companies } from "@/lib/mock-data";
import { localizedPrompts, localizedTopics } from "@/lib/i18n";
import { useI18n } from "@/lib/i18n-context";
import { useSessionHistory } from "@/hooks/use-session-history";
import type { UiBackendStatus, UiOperationState } from "@/types/api";
import type { WorkspaceResultViewModel } from "@/types/presentation";

interface RequestMeta { completedAt: string; durationSeconds: number; }

export default function WorkspaceClient() {
  const { locale } = useI18n();
  const [query, setQuery] = useState("");
  const [selectedCompanies, setSelectedCompanies] = useState(["asus", "gigabyte", "msi"]);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<UiBackendStatus>("loading");
  const [operation, setOperation] = useState<UiOperationState>("idle");
  const [result, setResult] = useState<WorkspaceResultViewModel | null>(null);
  const [requestMeta, setRequestMeta] = useState<RequestMeta | null>(null);
  const { history, addHistory, clearHistory } = useSessionHistory();
  const [menuOpen, setMenuOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const readinessRequested = useRef(false);

  const checkReadiness = useCallback(async () => {
    setBackendStatus("loading");
    try {
      await competitorApi.readiness();
      setBackendStatus("ready");
    } catch {
      setBackendStatus("unavailable");
    }
  }, []);

  useEffect(() => {
    if (readinessRequested.current) return;
    readinessRequested.current = true;
    void checkReadiness();
  }, [checkReadiness]);

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
    const startedAt = performance.now();
    setOperation("submitting");
    setSelectedEvidenceId(null);
    try {
      const mapped = mapAnalyzeResponse(await competitorApi.analyzeCompetitor(question));
      setResult(mapped);
      setRequestMeta({ completedAt: new Date().toISOString(), durationSeconds: (performance.now() - startedAt) / 1000 });
      addHistory(question, mapped.status);
      setOperation("idle");
    } catch (error) {
      setResult(null);
      setRequestMeta(null);
      setOperation(error instanceof ApiClientError && (error.status === 503 || error.status === null) ? "unavailable" : "server_failure");
    }
  }

  const closeDrawers = useCallback(() => {
    setMenuOpen(false);
    setEvidenceOpen(false);
  }, []);

  const selectedContext = companies.filter((company) => selectedCompanies.includes(company.id)).map((company) => company.name).join(" · ");

  return (
    <AppShell status={backendStatus} companies={companies} topics={localizedTopics[locale]} selectedCompanies={selectedCompanies} history={history} evidence={result?.evidence ?? []} selectedEvidenceId={selectedEvidenceId} menuOpen={menuOpen} evidenceOpen={evidenceOpen} onRetryReadiness={checkReadiness} onMenu={() => setMenuOpen(true)} onEvidence={() => setEvidenceOpen(true)} onCloseDrawers={closeDrawers} onClearHistory={clearHistory} onCompany={selectCompany} onTopic={(value) => { setQuery(value); closeDrawers(); }} onSelectEvidence={selectEvidence}>
      <ResearchWorkspace query={query} context={selectedContext} operation={operation} result={result} requestMeta={requestMeta} prompts={localizedPrompts[locale]} onQuery={setQuery} onAnalyze={analyze} onEvidence={selectEvidence} />
    </AppShell>
  );
}
