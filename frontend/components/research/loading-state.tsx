const phases = [
  "Searching trusted evidence",
  "Preparing financial analysis",
  "Generating a grounded response",
  "Organizing provenance",
];

export function LoadingState() {
  return (
    <div className="loading-state" role="status" aria-live="polite">
      <div className="loading-orbit" aria-hidden="true"><span /></div>
      <div>
        <p className="section-kicker">Analysis in progress</p>
        <strong>Building your evidence-backed brief</strong>
        <div className="loading-phases">{phases.map((phase) => <span key={phase}>{phase}</span>)}</div>
        <p className="loading-note">These indicators describe the experience, not measured backend phases.</p>
      </div>
    </div>
  );
}
