import { Search } from "lucide-react";
export function QueryComposer({ value, onChange, onAnalyze }: { value: string; onChange: (value: string) => void; onAnalyze: () => void }) {
  return <div className="composer"><label htmlFor="research-query">Research question</label><textarea id="research-query" value={value} onChange={(event) => onChange(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") onAnalyze(); }} placeholder="Compare ASUS, Gigabyte, and MSI on profitability and strategic positioning…" /><div className="composer-actions"><span>Ctrl + Enter to analyze</span><button type="button" disabled={!value.trim()} onClick={onAnalyze}><Search size={16} /> Analyze preview</button></div></div>;
}
