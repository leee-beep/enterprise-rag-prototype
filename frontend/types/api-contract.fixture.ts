import type {
  CompetitorAnalyzeRequest,
  CompetitorAnalyzeResponse,
  DomainStatus,
  GenerationMetadata,
} from "./api";

const validRequest = {
  question: "Synthetic contract-check question",
} satisfies CompetitorAnalyzeRequest;

const supportedStatuses = [
  "completed",
  "partial",
  "ambiguous",
  "unsupported",
  "insufficient",
] satisfies DomainStatus[];

const nullableGeneration = {
  provider: null,
  model: null,
} satisfies GenerationMetadata;

const nullableResponse = {
  question: validRequest.question,
  status: supportedStatuses[4],
  answer_text: null,
  reasons: ["Synthetic insufficient-evidence reason"],
  citations: [
    {
      evidence_id: "SYNTHETIC-E1",
      evidence_type: "qualitative",
      company_id: null,
      company_name: null,
      fiscal_year: null,
      sources: [
        {
          source_title: "Synthetic source",
          page_number: null,
          source_metric: null,
          fiscal_year: null,
          document_type: null,
        },
      ],
    },
  ],
  financial_claims: [
    {
      evidence_id: "SYNTHETIC-F1",
      claim_type: "metric",
      role: "subject",
      value: "12.3",
      company_id: null,
      rank: null,
    },
  ],
  generation: nullableGeneration,
  comparison: null,
  response_language: null,
} satisfies CompetitorAnalyzeResponse;

void nullableResponse;
