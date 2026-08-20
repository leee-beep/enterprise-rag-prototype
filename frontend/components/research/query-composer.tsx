import { ArrowUp, Command, LockKeyhole } from "lucide-react";
import { useI18n } from "@/lib/i18n-context";

export function QueryComposer({ value, context, submitting, onChange, onAnalyze }: {
  value: string;
  context: string;
  submitting: boolean;
  onChange: (value: string) => void;
  onAnalyze: () => void;
}) {
  const { t } = useI18n();
  return (
    <div className="composer">
      <div className="composer-heading">
        <label htmlFor="research-query">{t("researchQuestion")}</label>
        <span><LockKeyhole size={12} /> {t("localAnalysis")}</span>
      </div>
      <textarea
        id="research-query"
        value={value}
        disabled={submitting}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (!submitting && (event.ctrlKey || event.metaKey) && event.key === "Enter") onAnalyze();
        }}
        placeholder={t("queryPlaceholder")}
      />
      <div className="composer-context" aria-label={t("context")}>{t("context")} · {context}</div>
      <div className="composer-actions">
        <span><Command size={12} /> Ctrl/Cmd + Enter</span>
        <button type="button" aria-busy={submitting} disabled={submitting || !value.trim()} onClick={onAnalyze}>
          {submitting ? t("analyzing") : t("analyze")}<ArrowUp size={15} />
        </button>
      </div>
    </div>
  );
}
