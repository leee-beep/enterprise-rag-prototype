import { FileText, ShieldCheck } from "lucide-react";
import type { EvidenceViewModel, StructuredComparisonViewModel } from "@/types/presentation";
import { useI18n } from "@/lib/i18n-context";

function Links({ ids, onEvidence }: { ids: string[]; onEvidence: (id: string) => void }) {
  return <div className="profile-evidence-links">{ids.map((id) => <button key={id} type="button" onClick={() => onEvidence(id)}><FileText size={12} />{id}</button>)}</div>;
}

export function ExecutiveComparison({ comparison, evidence, onEvidence }: {
  comparison: StructuredComparisonViewModel | null;
  evidence: EvidenceViewModel[];
  onEvidence: (id: string) => void;
}) {
  const { t } = useI18n();
  if (!comparison) {
    const qualitative = evidence.filter((item) => item.kind === "qualitative");
    if (!qualitative.length) return null;
    return <section className="analysis-section executive-comparison"><div className="section-heading"><span>01</span><div><p className="section-kicker">{t("verifiedEvidenceCoverage")}</p><h3>{t("verifiedEvidenceCoverage")}</h3></div></div><div className="profile-evidence-links">{qualitative.map((item) => <button key={item.id} type="button" onClick={() => onEvidence(item.id)}><FileText size={12} />{item.companyName ?? item.companyId ?? item.id} · {item.id}</button>)}</div></section>;
  }

  return (
    <section className="analysis-section executive-comparison" aria-labelledby="executive-comparison-title">
      <div className="section-heading">
        <span>01</span>
        <div>
          <p className="section-kicker">{t("executiveComparison")}</p>
          <h3 id="executive-comparison-title">{t("strategyProfiles")}</h3>
        </div>
      </div>
      {comparison.missingCompanies.length > 0 && <p className="section-description"><strong>{t("missingCompanies")}:</strong> {comparison.missingCompanies.join(", ")}</p>}
      <div className="comparison-grid">
        {comparison.companyProfiles.map((profile) => (
          <article className="company-profile" key={profile.companyId}>
            <div className="company-profile-heading">
              <div><span className="company-monogram">{profile.companyId.slice(0, 1).toUpperCase()}</span><h4>{profile.companyId}</h4></div>
              <span className="verified-label"><ShieldCheck size={13} /> {t("verified")}</span>
            </div>
            <p>{profile.summary}</p><Links ids={profile.evidenceIds} onEvidence={onEvidence} />
          </article>
        ))}
      </div>
      <div className="comparison-dimensions"><h4>{t("comparisonDimensions")}</h4>{comparison.dimensions.map((dimension) => <article key={dimension.label}><h5>{dimension.label}</h5>{dimension.observations.map((observation) => <div key={observation.companyId}><strong>{observation.companyId}</strong><p>{observation.text}</p><Links ids={observation.evidenceIds} onEvidence={onEvidence} /></div>)}</article>)}</div>
      {comparison.keyTakeaway && <section className="key-takeaway"><p className="section-kicker">{t("keyTakeaway")}</p><p>{comparison.keyTakeaway.text}</p><Links ids={comparison.keyTakeaway.evidenceIds} onEvidence={onEvidence} /></section>}
    </section>
  );
}
