import { Search } from "lucide-react";

export function QueryComposer({ value, submitting, onChange, onAnalyze }: {
  value: string;
  submitting: boolean;
  onChange: (value: string) => void;
  onAnalyze: () => void;
}) {
  return (
    <div className="composer">
      <label htmlFor="research-query">Research question</label>
      <textarea
        id="research-query"
        value={value}
        disabled={submitting}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (!submitting && (event.ctrlKey || event.metaKey) && event.key === "Enter") onAnalyze();
        }}
        placeholder="Compare ASUS, Gigabyte, and MSI on profitability and strategic positioning."
      />
      <div className="composer-actions">
        <span>Ctrl + Enter to analyze</span>
        <button type="button" aria-busy={submitting} disabled={submitting || !value.trim()} onClick={onAnalyze}>
          <Search size={16} /> {submitting ? "Analyzing" : "Analyze"}
        </button>
      </div>
    </div>
  );
}
