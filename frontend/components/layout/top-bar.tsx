import { Menu, PanelRightOpen } from "lucide-react";
import type { UiBackendStatus } from "@/types/api";
import { StatusBadge } from "@/components/status/status-badge";
import { useI18n } from "@/lib/i18n-context";

export function TopBar({ status, onRetry, onMenu, onEvidence }: {
  status: UiBackendStatus;
  onRetry: () => void;
  onMenu: () => void;
  onEvidence: () => void;
}) {
  const { locale, setLocale, t } = useI18n();
  return <header className="topbar"><button className="mobile-icon" type="button" onClick={onMenu} aria-label={t("openNavigation")}><Menu size={19} /></button><div className="topbar-title"><p className="eyebrow">{t("localEnvironment")}</p><h1>{t("appName")}</h1></div><div className="topbar-actions"><span className="privacy-label">{t("privateLocal")}</span><div className="locale-switcher" role="group" aria-label={t("localeSwitcher")}><button type="button" className={locale === "zh-TW" ? "active" : ""} aria-pressed={locale === "zh-TW"} onClick={() => setLocale("zh-TW")}>{t("localeChinese")}</button><span aria-hidden="true">|</span><button type="button" className={locale === "en" ? "active" : ""} aria-pressed={locale === "en"} onClick={() => setLocale("en")}>{t("localeEnglish")}</button></div><StatusBadge status={status} onRetry={onRetry} /><button className="mobile-icon" type="button" onClick={onEvidence} aria-label={t("openEvidence")}><PanelRightOpen size={19} /></button></div></header>;
}
