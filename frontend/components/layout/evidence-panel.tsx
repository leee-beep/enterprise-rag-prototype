import { Calculator, ChevronLeft, ChevronRight, FileSearch2, X } from "lucide-react";
import type { EvidenceViewModel } from "@/types/presentation";

export function EvidencePanel({ evidence, selectedId, open, onClose, onSelect }: {
  evidence: EvidenceViewModel[];
  selectedId: string | null;
  open: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
}) {
  const selectedIndex = evidence.findIndex((item) => item.id === selectedId);
  const current = selectedIndex >= 0 ? selectedIndex : 0;
  return (
    <aside className={`evidence-pane ${open ? "drawer-open" : ""}`} aria-label="Evidence and provenance panel">
      <div className="panel-heading">
        <div><p className="eyebrow">Structured provenance</p><h2>Evidence</h2></div>
        <div className="panel-actions"><span className="panel-count">{evidence.length}</span><button className="drawer-close" onClick={onClose} type="button" aria-label="Close evidence panel"><X size={19} /></button></div>
      </div>
      {evidence.length > 1 && <div className="evidence-navigation" aria-label="Evidence navigation"><span>{selectedIndex >= 0 ? current + 1 : 0} of {evidence.length}</span><div><button type="button" disabled={current <= 0} onClick={() => onSelect(evidence[current - 1].id)} aria-label="Previous evidence"><ChevronLeft size={15} /></button><button type="button" disabled={current >= evidence.length - 1} onClick={() => onSelect(evidence[current + 1].id)} aria-label="Next evidence"><ChevronRight size={15} /></button></div></div>}
      {evidence.length ? (
        <div className="evidence-list">
          {evidence.map((item) => (
            <button className={`evidence-card ${selectedId === item.id ? "selected" : ""}`} key={item.id} onClick={() => onSelect(item.id)} type="button" aria-current={selectedId === item.id ? "true" : undefined} aria-label={`Select evidence ${item.id}`}>
              <div className="evidence-card-top">
                <span className="evidence-kind">{item.kind === "financial" ? <Calculator size={13} /> : <FileSearch2 size={13} />}{item.kind === "financial" ? "Financial" : "Qualitative"}</span>
                <strong>{item.id}</strong>
              </div>
              <dl className="evidence-summary">
                <div><dt>Company</dt><dd>{item.companyName ?? item.companyId ?? "Not available"}</dd></div>
                <div><dt>Year</dt><dd>{item.fiscalYear ?? "Not available"}</dd></div>
              </dl>
              <div className="source-stack">
                {item.sources.map((source, index) => (
                  <div className="source-reference" key={`${item.id}-${index}`}>
                    <span>Source {index + 1}</span>
                    <strong title={source.sourceTitle}>{source.sourceTitle}</strong>
                    <dl>
                      <div><dt>Page</dt><dd>{source.pageNumber ?? "Not available"}</dd></div>
                      <div><dt>Document</dt><dd>{source.documentType ?? "Not available"}</dd></div>
                      {source.sourceMetric && <div><dt>Metric</dt><dd>{source.sourceMetric}</dd></div>}
                    </dl>
                  </div>
                ))}
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div className="evidence-empty"><FileSearch2 size={22} /><p>No evidence yet</p><small>Structured citations and financial provenance appear after analysis.</small></div>
      )}
    </aside>
  );
}
