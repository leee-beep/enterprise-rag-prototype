import type { EvidenceViewModel } from "@/types/presentation";

export function QualitativeInsight({
  evidence,
  onEvidence,
}: {
  evidence: EvidenceViewModel[];
  onEvidence: (id: string) => void;
}) {
  return (
    <section className="analysis-section">
      <div className="section-kicker">02 · Qualitative signals</div>
      <div className="insight-grid">
        {evidence
          .filter((item) => item.kind === "qualitative")
          .map((item) => (
            <article key={item.id}>
              <div>
                <span>{item.company}</span>
                <button type="button" onClick={() => onEvidence(item.id)}>
                  {item.id}
                </button>
              </div>
              <h4>{item.title}</h4>
              <p>{item.excerpt}</p>
            </article>
          ))}
      </div>
    </section>
  );
}
