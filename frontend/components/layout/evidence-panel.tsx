import { Calculator, FileSearch2, X } from "lucide-react";
import type { EvidenceViewModel } from "@/types/presentation";

export function EvidencePanel({ evidence, selectedId, open, onClose, onSelect }: {
  evidence: EvidenceViewModel[];
  selectedId: string | null;
  open: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
}) {
  return (
    <aside className={`evidence-pane ${open ? "drawer-open" : ""}`} aria-label="Evidence panel">
      <div className="panel-heading">
        <div><p className="eyebrow">Source trail</p><h2>Evidence</h2></div>
        <div className="panel-count">{evidence.length}</div>
        <button className="drawer-close" onClick={onClose} type="button" aria-label="Close evidence"><X size={19} /></button>
      </div>
      {evidence.length ? (
        <div className="evidence-list">
          {evidence.map((item) => (
            <button className={`evidence-card ${selectedId === item.id ? "selected" : ""}`} key={item.id} onClick={() => onSelect(item.id)} type="button">
              <div className="evidence-card-top">
                {item.kind === "financial" ? <Calculator size={14} /> : <FileSearch2 size={14} />}
                <span>{item.id}</span>
                <small>{item.companyName ?? item.companyId ?? "Unspecified company"}{item.fiscalYear ? ` · FY${item.fiscalYear}` : ""}</small>
              </div>
              <strong>{item.kind === "financial" ? "Financial provenance" : "Annual-report evidence"}</strong>
              {item.sources.map((source, index) => (
                <div className="source-reference" key={`${item.id}-${index}`}>
                  <span>{source.sourceTitle}</span>
                  <small>{[source.documentType, source.fiscalYear ? `FY${source.fiscalYear}` : null, source.pageNumber ? `p. ${source.pageNumber}` : null, source.sourceMetric].filter(Boolean).join(" · ")}</small>
                </div>
              ))}
            </button>
          ))}
        </div>
      ) : (
        <div className="evidence-empty"><FileSearch2 size={22} /><p>No evidence selected</p><small>Citations and financial provenance will appear after analysis.</small></div>
      )}
    </aside>
  );
}
