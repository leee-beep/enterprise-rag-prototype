import { Building2, FileText, Scale, ShieldCheck } from "lucide-react";

export function EmptyState() {
  return (
    <section className="empty-state" aria-labelledby="workspace-start-title">
      <div className="empty-intro">
        <span className="empty-icon"><FileText size={20} /></span>
        <div>
          <p className="section-kicker">Research scope</p>
          <h3 id="workspace-start-title">Grounded comparison, ready when you are</h3>
          <p>Ask about strategy, positioning, risk, or verified financial performance. Every live conclusion stays connected to structured evidence.</p>
        </div>
      </div>
      <div className="scope-grid">
        <article><Building2 size={17} /><strong>Companies</strong><span>ASUS · Gigabyte · MSI</span></article>
        <article><Scale size={17} /><strong>Analysis</strong><span>Strategy · Financials · Change</span></article>
        <article><ShieldCheck size={17} /><strong>Authority</strong><span>Annual reports · Validated facts</span></article>
      </div>
    </section>
  );
}
