import { ArrowUp, Command, LockKeyhole } from "lucide-react";

export function QueryComposer({ value, context, submitting, onChange, onAnalyze }: {
  value: string;
  context: string;
  submitting: boolean;
  onChange: (value: string) => void;
  onAnalyze: () => void;
}) {
  return (
    <div className="composer">
      <div className="composer-heading">
        <label htmlFor="research-query">Research question</label>
        <span><LockKeyhole size={12} /> Local analysis</span>
      </div>
      <textarea
        id="research-query"
        value={value}
        disabled={submitting}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (!submitting && (event.ctrlKey || event.metaKey) && event.key === "Enter") onAnalyze();
        }}
        placeholder="Ask a comparative strategy or financial question…"
      />
      <div className="composer-context" aria-label="Selected company context">Context · {context}</div>
      <div className="composer-actions">
        <span><Command size={12} /> Ctrl/Cmd + Enter</span>
        <button type="button" aria-busy={submitting} disabled={submitting || !value.trim()} onClick={onAnalyze}>
          {submitting ? "Analyzing" : "Analyze"}<ArrowUp size={15} />
        </button>
      </div>
    </div>
  );
}
