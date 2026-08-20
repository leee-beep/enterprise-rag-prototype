import { BarChart3, Building2, FileSearch2, Layers3, Sparkles } from "lucide-react";
import type { UiOperationState } from "@/types/api";
import type { WorkspaceResultViewModel } from "@/types/presentation";
import { claimRoleLabel, claimTypeLabel, safeReasonLabel } from "@/lib/display-labels";
import { DomainState } from "@/components/status/domain-state";
import { OperationState } from "@/components/status/operation-state";
import { EmptyState } from "./empty-state";
import { ExecutiveComparison } from "./executive-comparison";
import { LoadingState } from "./loading-state";
import { QueryComposer } from "./query-composer";
import { useI18n } from "@/lib/i18n-context";

const promptIcons = [Building2, FileSearch2, BarChart3, Layers3];
const durationFormat = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });

function resultCompanies(result: WorkspaceResultViewModel): string[] {
  const names: string[] = [];
  for (const item of result.evidence) {
    const name = item.companyName ?? item.companyId;
    if (name && !names.includes(name)) names.push(name);
  }
  return names;
}

export function ResearchWorkspace({ query, context, operation, result, requestMeta, prompts, onQuery, onAnalyze, onEvidence }: {
  query: string;
  context: string;
  operation: UiOperationState;
  result: WorkspaceResultViewModel | null;
  requestMeta: { completedAt: string; durationSeconds: number } | null;
  prompts: string[];
  onQuery: (value: string) => void;
  onAnalyze: () => void;
  onEvidence: (id: string) => void;
}) {
  const { locale, t } = useI18n();
  const submitting = operation === "submitting";
  const companies = result ? resultCompanies(result) : [];
  const qualitativeCompanyCount = result
    ? new Set(result.evidence.filter((item) => item.kind === "qualitative").map((item) => item.companyId ?? item.companyName).filter(Boolean)).size
    : 0;
  const isComparison = qualitativeCompanyCount > 1;
  const coverageNotes = result
    ? [...new Set(result.reasons.map((reason) => safeReasonLabel(reason, locale)))]
    : [];
  return (
    <section className={`research-pane ${result ? "has-result" : ""}`}>
      <div className="hero">
        <p className="eyebrow accent"><Sparkles size={13} /> {t("evidenceDecisionSupport")}</p>
        <h2>{t("heroTitle")}</h2>
        <p>{t("heroDescription")}</p>
      </div>
      <QueryComposer value={query} context={context} submitting={submitting} onChange={onQuery} onAnalyze={onAnalyze} />
      {!result && !submitting && operation === "idle" && <div className="prompt-grid" aria-label="Example research questions">{prompts.map((prompt, index) => { const Icon = promptIcons[index % promptIcons.length]; return <button key={prompt} type="button" onClick={() => onQuery(prompt)}><Icon size={16} /><span>{prompt}</span></button>; })}</div>}
      {!result && operation === "idle" && <EmptyState />}
      {submitting && <LoadingState />}
      {!result && (operation === "unavailable" || operation === "server_failure") && <div className="analysis-results"><OperationState state={operation} onRetry={onAnalyze} /></div>}
      {result && !submitting && (
        <article className="analysis-results" aria-label="Competitor intelligence result">
          <header className="result-header">
            <div>
              <p className="section-kicker">{t("liveAnalysis")}</p>
              <h3>{result.question}</h3>
              <div className="result-context">
                {companies.length > 0 && <span>{companies.join(" · ")}</span>}
                <DomainState status={result.status} compact />
                {requestMeta && <span>{new Date(requestMeta.completedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>}
                {requestMeta && <span>{durationFormat.format(requestMeta.durationSeconds)}s</span>}
              </div>
            </div>
          </header>

          {coverageNotes.length > 0 && <section className="coverage-note"><strong>{t("coverageNote")}</strong><ul>{coverageNotes.map((note) => <li key={note}>{note}</li>)}</ul></section>}

          <ExecutiveComparison comparison={result.comparison} evidence={result.evidence} onEvidence={onEvidence} />

          {result.answerText && <section className="analysis-section answer-section" data-answer><div className="section-heading"><span>{isComparison ? "02" : "01"}</span><div><p className="section-kicker">{t("supportingNarrative")}</p><h3>{t("detailedAnalysis")}</h3></div></div><p className="answer-text">{result.answerText}</p></section>}

          {result.financialItems.length > 0 && <section className="analysis-section financial-section"><div className="section-heading"><span>{isComparison ? "03" : "02"}</span><div><p className="section-kicker">{t("deterministicAuthority")}</p><h3>{t("financialComparison")}</h3></div></div><p className="section-description">{t("financialDisclosure")}</p><div className="financial-claims">{result.financialItems.map((item, index) => <article className="financial-claim" data-financial-claim key={`${item.evidenceId}-${index}`}><div className="claim-top"><span>{claimTypeLabel(item.claimType)}</span><button type="button" onClick={() => onEvidence(item.evidenceId)} aria-label={`${t("evidence")} ${item.evidenceId}`}>{item.evidenceId}</button></div><strong>{item.displayValue}</strong><dl><div><dt>{t("company")}</dt><dd>{item.companyId ?? t("notAvailable")}</dd></div><div><dt>{t("role")}</dt><dd>{claimRoleLabel(item.role)}</dd></div>{item.rankLabel && <div><dt>{t("rank")}</dt><dd><span className="rank-badge">{item.rankLabel}</span></dd></div>}</dl></article>)}</div></section>}

          {result.evidence.length > 0 && <section className="analysis-section citation-section"><div className="section-heading"><span>{isComparison ? (result.financialItems.length > 0 ? "04" : "03") : (result.answerText || result.financialItems.length ? "03" : "01")}</span><div><p className="section-kicker">{t("structuredProvenance")}</p><h3>{t("sourcesEvidence")}</h3></div></div><div className="citation-list">{result.evidence.map((item) => <button data-citation-item key={item.id} type="button" onClick={() => onEvidence(item.id)} aria-label={`${t("evidence")} ${item.id}`}><span className="citation-id">{item.id}</span><span className="citation-copy"><strong>{item.companyName ?? item.companyId ?? t("notAvailable")}</strong><small>{item.sources[0]?.sourceTitle ?? t("sourceDetailsUnavailable")}</small></span><span className="citation-count">{item.sources.length} {t("sources")}</span></button>)}</div></section>}

          {result.generation && (result.generation.provider || result.generation.model) && <footer className="generation-meta"><span>{t("localAI")}</span>{result.generation.provider && <span>{result.generation.provider}</span>}{result.generation.model && <span>{result.generation.model}</span>}</footer>}
        </article>
      )}
    </section>
  );
}
