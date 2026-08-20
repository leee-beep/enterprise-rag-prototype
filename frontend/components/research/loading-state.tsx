import { useI18n } from "@/lib/i18n-context";

export function LoadingState() {
  const { t } = useI18n();
  const phases = [t("phaseSearch"), t("phaseFinancial"), t("phaseGenerate"), t("phaseProvenance")];
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <div className="loading-orbit" aria-hidden="true"><span /></div>
      <div>
        <p className="section-kicker">{t("analysisProgress")}</p>
        <strong>{t("buildingBrief")}</strong>
        <div className="loading-phases">{phases.map((phase) => <span key={phase}>{phase}</span>)}</div>
        <p className="loading-note">{t("loadingNote")}</p>
      </div>
    </div>
  );
}
