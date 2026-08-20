import { Building2, FileText, Scale, ShieldCheck } from "lucide-react";
import { useI18n } from "@/lib/i18n-context";

export function EmptyState() {
  const { t } = useI18n();
  return (
    <section className="empty-state" aria-labelledby="workspace-start-title">
      <div className="empty-intro">
        <span className="empty-icon"><FileText size={20} /></span>
        <div>
          <p className="section-kicker">{t("researchScope")}</p>
          <h3 id="workspace-start-title">{t("emptyTitle")}</h3>
          <p>{t("emptyDescription")}</p>
        </div>
      </div>
      <div className="scope-grid">
        <article><Building2 size={17} /><strong>{t("companies")}</strong><span>ASUS · Gigabyte · MSI</span></article>
        <article><Scale size={17} /><strong>{t("analysis")}</strong><span>{t("strategyFinancialsChange")}</span></article>
        <article><ShieldCheck size={17} /><strong>{t("authority")}</strong><span>{t("annualReportsFacts")}</span></article>
      </div>
    </section>
  );
}
