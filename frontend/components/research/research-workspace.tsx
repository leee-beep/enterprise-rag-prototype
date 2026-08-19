import { BarChart3, Building2, FileSearch2, Layers3, Sparkles } from "lucide-react";
import type { UiOperationState } from "@/types/api";
import type { WorkspaceResultViewModel } from "@/types/presentation";
import { claimRoleLabel, claimTypeLabel, safeReasonLabel } from "@/lib/display-labels";
import { DomainState } from "@/components/status/domain-state";
import { OperationState } from "@/components/status/operation-state";
import { EmptyState } from "./empty-state";
import { LoadingState } from "./loading-state";
import { QueryComposer } from "./query-composer";

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
  const submitting = operation === "submitting";
  const companies = result ? resultCompanies(result) : [];
  const coverageNotes = result
    ? [...new Set(result.reasons.map((reason) => safeReasonLabel(reason)))]
    : [];
  return (
    <section className={`research-pane ${result ? "has-result" : ""}`}>
      <div className="hero">
        <p className="eyebrow accent"><Sparkles size={13} /> Evidence-backed decision support</p>
        <h2>Research competitors.<br />Verify every conclusion.</h2>
        <p>Compare strategy and validated financial performance across ASUS, Gigabyte, and MSI using local annual-report evidence.</p>
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
              <p className="section-kicker">Live analysis</p>
              <h3>{result.question}</h3>
              <div className="result-context">
                {companies.length > 0 && <span>{companies.join(" · ")}</span>}
                {requestMeta && <span>{new Date(requestMeta.completedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>}
                {requestMeta && <span>Request time {durationFormat.format(requestMeta.durationSeconds)}s</span>}
              </div>
            </div>
            <DomainState status={result.status} />
          </header>

          {coverageNotes.length > 0 && <section className="coverage-note"><strong>Coverage note</strong><ul>{coverageNotes.map((note) => <li key={note}>{note}</li>)}</ul></section>}

          {result.answerText && <section className="analysis-section answer-section" data-answer><div className="section-heading"><span>01</span><div><p className="section-kicker">Grounded analysis</p><h3>Research brief</h3></div></div><p className="answer-text">{result.answerText}</p></section>}

          {result.financialItems.length > 0 && <section className="analysis-section financial-section"><div className="section-heading"><span>02</span><div><p className="section-kicker">Deterministic authority</p><h3>Financial insights</h3></div></div><p className="section-description">Values and ranks below are displayed exactly as supplied by the validated backend contract.</p><div className="financial-claims">{result.financialItems.map((item, index) => <article className="financial-claim" data-financial-claim key={`${item.evidenceId}-${index}`}><div className="claim-top"><span>{claimTypeLabel(item.claimType)}</span><button type="button" onClick={() => onEvidence(item.evidenceId)} aria-label={`Open provenance for ${item.evidenceId}`}>{item.evidenceId}</button></div><strong>{item.displayValue}</strong><dl><div><dt>Company</dt><dd>{item.companyId ?? "Not available"}</dd></div><div><dt>Role</dt><dd>{claimRoleLabel(item.role)}</dd></div>{item.rankLabel && <div><dt>Rank</dt><dd><span className="rank-badge">{item.rankLabel}</span></dd></div>}</dl></article>)}</div></section>}

          {result.evidence.length > 0 && <section className="analysis-section citation-section"><div className="section-heading"><span>{result.answerText || result.financialItems.length ? "03" : "01"}</span><div><p className="section-kicker">Structured provenance</p><h3>Sources and evidence</h3></div></div><div className="citation-list">{result.evidence.map((item) => <button data-citation-item key={item.id} type="button" onClick={() => onEvidence(item.id)} aria-label={`Inspect evidence ${item.id}`}><span className="citation-id">{item.id}</span><span className="citation-copy"><strong>{item.companyName ?? item.companyId ?? "Not available"}</strong><small>{item.sources[0]?.sourceTitle ?? "Source details not available"}</small></span><span className="citation-count">{item.sources.length} {item.sources.length === 1 ? "source" : "sources"}</span></button>)}</div></section>}

          {result.generation && (result.generation.provider || result.generation.model) && <footer className="generation-meta"><span>Local AI</span>{result.generation.provider && <span>{result.generation.provider}</span>}{result.generation.model && <span>{result.generation.model}</span>}</footer>}
        </article>
      )}
    </section>
  );
}
