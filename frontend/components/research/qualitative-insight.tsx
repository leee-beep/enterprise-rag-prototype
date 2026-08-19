import type { EvidenceViewModel } from "@/types/presentation";

export function QualitativeInsight({ evidence, onEvidence }: { evidence: EvidenceViewModel[]; onEvidence: (id: string) => void }) {
  return <section className="analysis-section"><div className="section-kicker">Qualitative signals</div><div className="insight-grid">{evidence.filter((item) => item.kind === "qualitative").map((item) => <article key={item.id}><div><span>{item.companyName ?? item.companyId ?? "Source"}</span><button type="button" onClick={() => onEvidence(item.id)}>{item.id}</button></div><h4>{item.sources[0]?.sourceTitle ?? "Structured citation"}</h4><p>{item.sources.map((source) => source.documentType).filter(Boolean).join(" · ")}</p></article>)}</div></section>;
}
