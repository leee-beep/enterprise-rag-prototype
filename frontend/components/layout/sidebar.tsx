import { Building2, Clock3, Layers3, Trash2, X } from "lucide-react";
import type { CompanyOption, TopicShortcut } from "@/types/presentation";
import { useI18n } from "@/lib/i18n-context";

export interface HistoryEntry { question: string; timestamp: string; status: string; }

export function Sidebar({ companies, topics, selected, history, open, onClose, onClearHistory, onCompany, onTopic }: {
  companies: CompanyOption[];
  topics: TopicShortcut[];
  selected: string[];
  history: HistoryEntry[];
  open: boolean;
  onClose: () => void;
  onClearHistory: () => void;
  onCompany: (id: string) => void;
  onTopic: (query: string) => void;
}) {
  const { t } = useI18n();
  return (
    <aside className={`sidebar ${open ? "drawer-open" : ""}`} aria-label="Research navigation" aria-hidden={!open ? undefined : false}>
      <div className="sidebar-head">
        <div className="brand-mark"><Layers3 size={17} /><span>CI</span></div>
        <button className="drawer-close" onClick={onClose} type="button" aria-label="Close navigation"><X size={19} /></button>
      </div>
      <nav>
        <p className="nav-label">{t("companies")}</p>
        {companies.map((company) => <button className={`company-button company-${company.id} ${selected.includes(company.id) ? "active" : ""}`} key={company.id} type="button" aria-pressed={selected.includes(company.id)} onClick={() => onCompany(company.id)}><Building2 size={15} /><span>{company.name}<small>{company.ticker}</small></span></button>)}
        <p className="nav-label nav-spaced">{t("themes")}</p>
        {topics.map((topic) => <button className="topic-button" key={topic.label} type="button" onClick={() => onTopic(topic.query)}>{topic.label}</button>)}
        <div className="nav-label-row"><p className="nav-label nav-spaced">{t("recentQueries")}</p>{history.length > 0 && <button type="button" className="history-clear" onClick={onClearHistory} aria-label={t("clear")}><Trash2 size={12} /> {t("clear")}</button>}</div>
        {history.length ? history.slice(0, 5).map((item) => <button className="history-item" key={item.timestamp} type="button" onClick={() => onTopic(item.question)}><Clock3 size={13} /><span>{item.question}<small><em>{item.status.replace("_", " ")}</em>{new Date(item.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small></span></button>) : <p className="history-empty">{t("localHistory")}</p>}
      </nav>
    </aside>
  );
}
