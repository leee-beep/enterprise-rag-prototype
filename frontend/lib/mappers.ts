import type { Citation, CompetitorAnalyzeResponse, FinancialClaim } from "@/types/api";
import type { EvidenceViewModel, FinancialDisplayItem, WorkspaceResultViewModel } from "@/types/presentation";

export function mapCitation(citation: Citation): EvidenceViewModel {
  return {
    id: citation.evidence_id,
    kind: citation.evidence_type === "qualitative" ? "qualitative" : "financial",
    companyId: citation.company_id,
    companyName: citation.company_name,
    fiscalYear: citation.fiscal_year === null ? null : String(citation.fiscal_year),
    sources: citation.sources.map((source) => ({
      sourceTitle: source.source_title,
      pageNumber: source.page_number === null ? null : String(source.page_number),
      sourceMetric: source.source_metric,
      fiscalYear: source.fiscal_year === null ? null : String(source.fiscal_year),
      documentType: source.document_type,
    })),
  };
}

export function mapFinancialClaim(claim: FinancialClaim): FinancialDisplayItem {
  return {
    evidenceId: claim.evidence_id,
    companyId: claim.company_id,
    claimType: claim.claim_type,
    role: claim.role,
    displayValue: claim.value,
    rankLabel: claim.rank === null ? null : String(claim.rank),
  };
}

export function mapAnalyzeResponse(response: CompetitorAnalyzeResponse): WorkspaceResultViewModel {
  return {
    question: response.question,
    status: response.status,
    answerText: response.answer_text,
    reasons: [...response.reasons],
    evidence: response.citations.map(mapCitation),
    financialItems: response.financial_claims.map(mapFinancialClaim),
    generation: response.generation,
    comparison: response.comparison ? {
      requestedCompanies: [...response.comparison.requested_companies],
      coveredCompanies: [...response.comparison.covered_companies],
      missingCompanies: [...response.comparison.missing_companies],
      companyProfiles: response.comparison.company_profiles.map((item) => ({ companyId: item.company_id, summary: item.summary, evidenceIds: [...item.evidence_ids] })),
      dimensions: response.comparison.comparison_dimensions.map((item) => ({ label: item.label, observations: item.observations.map((observation) => ({ companyId: observation.company_id, text: observation.text, evidenceIds: [...observation.evidence_ids] })) })),
      keyTakeaway: response.comparison.key_takeaway ? { text: response.comparison.key_takeaway.text, evidenceIds: [...response.comparison.key_takeaway.evidence_ids] } : null,
    } : null,
    responseLanguage: response.response_language,
  };
}
