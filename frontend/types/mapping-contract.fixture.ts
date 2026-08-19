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
  };
  const mapped = mapAnalyzeResponse(response);
  if (mapped.status !== status || mapped.financialItems[0]?.displayValue !== "001.2300") {
    throw new Error("Mapping contract fixture failed.");
  }
}
