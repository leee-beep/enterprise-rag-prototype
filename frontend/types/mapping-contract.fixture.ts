import { mapAnalyzeResponse } from "@/lib/mappers";
import type { CompetitorAnalyzeResponse, DomainStatus } from "./api";

const statuses: DomainStatus[] = ["completed", "partial", "ambiguous", "unsupported", "insufficient"];

for (const status of statuses) {
  const response: CompetitorAnalyzeResponse = {
    question: "Synthetic mapping fixture",
    status,
    answer_text: status === "completed" || status === "partial" ? "Synthetic answer" : null,
    reasons: status === "completed" ? [] : ["synthetic_reason"],
    citations: [{
      evidence_id: "E1",
      evidence_type: "qualitative",
      company_id: null,
      company_name: null,
      fiscal_year: null,
      sources: [{ source_title: "Synthetic source", page_number: null, source_metric: null, fiscal_year: null, document_type: null }],
    }],
    financial_claims: [{ evidence_id: "F1", claim_type: "calculated_metric", role: "subject", value: "001.2300", company_id: null, rank: null }],
    generation: null,
    comparison: null,
    response_language: "en",
  };
  const mapped = mapAnalyzeResponse(response);
  if (mapped.status !== status || mapped.financialItems[0]?.displayValue !== "001.2300") {
    throw new Error("Mapping contract fixture failed.");
  }
}

for (const companies of [["asus", "msi"], ["asus", "gigabyte", "msi"]]) {
  const response: CompetitorAnalyzeResponse = {
    question: "Synthetic structured comparison",
    status: "completed",
    answer_text: "Detailed grounded narrative",
    reasons: [], citations: [], financial_claims: [], generation: null,
    response_language: "en",
    comparison: {
      requested_companies: companies, covered_companies: companies, missing_companies: [],
      company_profiles: companies.map((company, index) => ({ company_id: company, summary: `${company} profile`, evidence_ids: [`E${index + 1}`] })),
      comparison_dimensions: [{ label: "Positioning", observations: companies.map((company, index) => ({ company_id: company, text: `${company} observation`, evidence_ids: [`E${index + 1}`] })) }],
      key_takeaway: { text: "Grounded takeaway", evidence_ids: companies.map((_, index) => `E${index + 1}`) },
    },
  };
  const mapped = mapAnalyzeResponse(response);
  if (mapped.comparison?.companyProfiles.length !== companies.length || mapped.answerText !== response.answer_text) throw new Error("Structured comparison mapping failed.");
}
