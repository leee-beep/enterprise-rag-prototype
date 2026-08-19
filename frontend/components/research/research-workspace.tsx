import { Building2, FileSearch2, Layers3, Sparkles } from "lucide-react";
import type {
  EvidenceViewModel,
  FinancialDisplayItem,
  SyntheticChartDatum,
} from "@/types/presentation";
import { QueryComposer } from "./query-composer";
import { EmptyState } from "./empty-state";
import { LoadingState } from "./loading-state";
import { ExecutiveSummary } from "./executive-summary";
import { QualitativeInsight } from "./qualitative-insight";
import { MetricCard } from "@/components/financial/metric-card";
import { ComparisonTable } from "@/components/financial/comparison-table";
import { MarginChart } from "@/components/financial/margin-chart";

const promptIcons = [Building2, FileSearch2, Layers3];

export function ResearchWorkspace({
  query,
  view,
  prompts,
  evidence,
  financialItems,
  chartData,
  onQuery,
  onAnalyze,
  onEvidence,
}: {
  query: string;
  view: "empty" | "loading" | "preview";
  prompts: string[];
  evidence: EvidenceViewModel[];
  financialItems: FinancialDisplayItem[];
  chartData: SyntheticChartDatum[];
  onQuery: (value: string) => void;
  onAnalyze: () => void;
  onEvidence: (id: string) => void;
}) {
  return (
    <section className="research-pane">
      <div className="hero">
        <p className="eyebrow accent"><Sparkles size={14} /> Grounded local research</p>
        <h2>Ask sharper questions.<br />Trace every conclusion.</h2>
        <p>Explore annual-report evidence and deterministic financial comparisons without sending private documents outside your machine.</p>
      </div>
      <QueryComposer value={query} onChange={onQuery} onAnalyze={onAnalyze} />
      <div className="prompt-grid">
        {prompts.map((prompt, index) => {
          const Icon = promptIcons[index % promptIcons.length];
          return (
            <button key={prompt} type="button" onClick={() => onQuery(prompt)}>
              <Icon size={17} /><span>{prompt}</span>
            </button>
          );
        })}
      </div>
      {view === "empty" && <EmptyState />}
      {view === "loading" && <LoadingState />}
      {view === "preview" && (
        <div className="analysis-results">
          <div className="preview-banner">
            Synthetic preview · frontend interaction only · no API request sent
          </div>
          <ExecutiveSummary />
          <QualitativeInsight evidence={evidence} onEvidence={onEvidence} />
          <section className="analysis-section">
            <div className="section-kicker">03 · Financial comparison</div>
            <div className="metric-grid">
              {financialItems.map((item) => (
                <MetricCard key={item.company} item={item} />
              ))}
            </div>
            <div className="financial-grid">
              <ComparisonTable items={financialItems} />
              <MarginChart data={chartData} />
            </div>
          </section>
        </div>
      )}
    </section>
  );
}
