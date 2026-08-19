import { Building2, FileSearch2, Layers3, Sparkles } from "lucide-react";
import type { UiOperationState } from "@/types/api";
import type { WorkspaceResultViewModel } from "@/types/presentation";
import { DomainState } from "@/components/status/domain-state";
import { OperationState } from "@/components/status/operation-state";
import { EmptyState } from "./empty-state";
import { LoadingState } from "./loading-state";
import { QueryComposer } from "./query-composer";

const promptIcons = [Building2, FileSearch2, Layers3];

export function ResearchWorkspace({ query, operation, result, prompts, onQuery, onAnalyze, onEvidence }: {
  query: string;
  operation: UiOperationState;
  result: WorkspaceResultViewModel | null;
  prompts: string[];
  onQuery: (value: string) => void;
  onAnalyze: () => void;
  onEvidence: (id: string) => void;
}) {
  const submitting = operation === "submitting";
  return (
    <section className="research-pane">
      <div className="hero">
        <p className="eyebrow accent"><Sparkles size={14} /> Grounded local research</p>
        <h2>Ask sharper questions.<br />Trace every conclusion.</h2>
        <p>Explore annual-report evidence and deterministic financial comparisons without sending private documents outside your machine.</p>
      </div>
      <QueryComposer value={query} submitting={submitting} onChange={onQuery} onAnalyze={onAnalyze} />
      <div className="prompt-grid">
        {prompts.map((prompt, index) => {
          const Icon = promptIcons[index % promptIcons.length];
          return <button key={prompt} type="button" disabled={submitting} onClick={() => onQuery(prompt)}><Icon size={17} /><span>{prompt}</span></button>;
        })}
      </div>
      {!result && operation === "idle" && <EmptyState />}
      {submitting && <LoadingState />}
      {!result && (operation === "unavailable" || operation === "server_failure") && <div className="analysis-results"><OperationState state={operation} /></div>}
      {result && !submitting && (
        <div className="analysis-results">
          <DomainState status={result.status} />
          {result.reasons.length > 0 && <section className="analysis-section"><div className="section-kicker">Status details</div><ul className="reason-list">{result.reasons.map((reason, index) => <li key={`${reason}-${index}`}>{reason}</li>)}</ul></section>}
          {result.answerText && <section className="analysis-section"><div className="section-kicker">Grounded answer</div><p className="answer-text">{result.answerText}</p></section>}
          {result.evidence.length > 0 && <section className="analysis-section"><div className="section-kicker">Citations</div><div className="citation-list">{result.evidence.map((item) => <button key={item.id} type="button" onClick={() => onEvidence(item.id)}>{item.id}<span>{item.companyName ?? item.companyId ?? "Source"}</span></button>)}</div></section>}
          {result.financialItems.length > 0 && <section className="analysis-section"><div className="section-kicker">Validated financial claims</div><div className="table-wrap"><table><caption className="sr-only">Financial claims returned by the analysis API</caption><thead><tr><th>Evidence</th><th>Company</th><th>Claim</th><th>Role</th><th>Value</th><th>Rank</th></tr></thead><tbody>{result.financialItems.map((item, index) => <tr key={`${item.evidenceId}-${index}`}><td><button className="table-evidence" type="button" onClick={() => onEvidence(item.evidenceId)}>{item.evidenceId}</button></td><td>{item.companyId ?? "—"}</td><td>{item.claimType}</td><td>{item.role}</td><td>{item.displayValue}</td><td>{item.rankLabel ?? "—"}</td></tr>)}</tbody></table></div></section>}
          {result.generation && (result.generation.provider || result.generation.model) && <p className="generation-meta">Generated locally{result.generation.provider ? ` with ${result.generation.provider}` : ""}{result.generation.model ? ` · ${result.generation.model}` : ""}</p>}
        </div>
      )}
    </section>
  );
}
