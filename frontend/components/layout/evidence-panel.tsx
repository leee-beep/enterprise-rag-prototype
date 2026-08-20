import { Calculator, ChevronLeft, ChevronRight, FileSearch2, X } from "lucide-react";
import type { EvidenceViewModel } from "@/types/presentation";
import { useI18n } from "@/lib/i18n-context";

export function EvidencePanel({ evidence, selectedId, open, onClose, onSelect }: {
  evidence: EvidenceViewModel[];
  selectedId: string | null;
  open: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
}) {
  const { t } = useI18n();
  const selectedIndex = evidence.findIndex((item) => item.id === selectedId);
  const current = selectedIndex >= 0 ? selectedIndex : 0;
  return (
    <aside className={`evidence-pane ${open ? "drawer-open" : ""}`} aria-label="Evidence and provenance panel">
      <div className="panel-heading">
        <div><p className="eyebrow">{t("structuredProvenance")}</p><h2>{t("evidence")}</h2></div>
        <div className="panel-actions"><span className="panel-count">{evidence.length}</span><button className="drawer-close" onClick={onClose} type="button" aria-label="Close evidence panel"><X size={19} /></button></div>
      </div>
      {evidence.length > 1 && <div className="evidence-navigation" aria-label="Evidence navigation"><span>{selectedIndex >= 0 ? current + 1 : 0} of {evidence.length}</span><div><button type="button" disabled={current <= 0} onClick={() => onSelect(evidence[current - 1].id)} aria-label="Previous evidence"><ChevronLeft size={15} /></button><button type="button" disabled={current >= evidence.length - 1} onClick={() => onSelect(evidence[current + 1].id)} aria-label="Next evidence"><ChevronRight size={15} /></button></div></div>}
      {evidence.length ? (
        <div className="evidence-list">
          {evidence.map((item) => (
            <button className={`evidence-card ${selectedId === item.id ? "selected" : ""}`} key={item.id} onClick={() => onSelect(item.id)} type="button" aria-current={selectedId === item.id ? "true" : undefined} aria-label={`Select evidence ${item.id}`}>
              <div className="evidence-card-top">
                <span className="evidence-kind">{item.kind === "financial" ? <Calculator size={13} /> : <FileSearch2 size={13} />}{item.kind === "financial" ? t("financial") : t("qualitative")}</span>
                <strong>{item.id}</strong>
              </div>
              <dl className="evidence-summary">
                <div><dt>{t("company")}</dt><dd>{item.companyName ?? item.companyId ?? t("notAvailable")}</dd></div>
                <div><dt>{t("year")}</dt><dd>{item.fiscalYear ?? t("notAvailable")}</dd></div>
              </dl>
              <div className="source-stack">
                {item.sources.map((source, index) => (
                  <div className="source-reference" key={`${item.id}-${index}`}>
                    <span>{t("source")} {index + 1}</span>
                    <strong title={source.sourceTitle}>{source.sourceTitle}</strong>
                    <dl>
                      <div><dt>{t("page")}</dt><dd>{source.pageNumber ?? t("notAvailable")}</dd></div>
                      <div><dt>{t("document")}</dt><dd>{source.documentType ?? t("notAvailable")}</dd></div>
                      {source.sourceMetric && <div><dt>{t("metric")}</dt><dd>{source.sourceMetric}</dd></div>}
                    </dl>
                  </div>
                ))}
              </div>
            </button>
          ))}
        </div>
      ) : (
        <div className="evidence-empty"><FileSearch2 size={22} /><p>{t("noEvidence")}</p><small>{t("evidenceEmpty")}</small></div>
      )}
    </aside>
  );
}
