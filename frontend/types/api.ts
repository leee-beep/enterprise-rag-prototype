export type DomainStatus =
  | "completed"
  | "partial"
  | "ambiguous"
  | "unsupported"
  | "insufficient";

export type UiBackendStatus = "ready" | "loading" | "unavailable" | "unknown";
export type UiOperationState =
  | "idle"
  | "submitting"
  | "unavailable"
  | "server_failure";

export interface HealthResponse {
  status: "ok";
}

export interface ReadinessResponse {
  status: "ready";
  companies: string[];
  financial_facts_loaded: boolean;
  embedding_provider: string;
  embedding_model: string;
  generation_provider: string;
  generation_model: string;
}

export interface CompetitorAnalyzeRequest {
  question: string;
}

export interface CitationSourceReference {
  source_title: string;
  page_number: number | null;
  source_metric: string | null;
  fiscal_year: number | null;
  document_type: string | null;
}

export interface Citation {
  evidence_id: string;
  evidence_type: string;
  company_id: string | null;
  company_name: string | null;
  fiscal_year: number | null;
  sources: CitationSourceReference[];
}

export interface FinancialClaim {
  evidence_id: string;
  claim_type: string;
  role: string;
  value: string;
  company_id: string | null;
  rank: number | null;
}

export interface GenerationMetadata {
  provider: string | null;
  model: string | null;
}

export interface CompanyStrategyProfileResponse { company_id: string; summary: string; evidence_ids: string[]; }
export interface CompanyObservationResponse { company_id: string; text: string; evidence_ids: string[]; }
export interface ComparisonDimensionResponse { label: string; observations: CompanyObservationResponse[]; }
export interface GroundedKeyTakeawayResponse { text: string; evidence_ids: string[]; }
export interface StructuredComparisonResponse {
  requested_companies: string[];
  covered_companies: string[];
  missing_companies: string[];
  company_profiles: CompanyStrategyProfileResponse[];
  comparison_dimensions: ComparisonDimensionResponse[];
  key_takeaway: GroundedKeyTakeawayResponse | null;
}

export interface CompetitorAnalyzeResponse {
  question: string;
  status: DomainStatus;
  answer_text: string | null;
  reasons: string[];
  citations: Citation[];
  financial_claims: FinancialClaim[];
  generation: GenerationMetadata | null;
  comparison: StructuredComparisonResponse | null;
  response_language: "zh-TW" | "en" | null;
}

export interface ApiErrorDetail {
  code: string;
  message: string;
}

export interface ApiErrorResponse {
  error: ApiErrorDetail;
}
