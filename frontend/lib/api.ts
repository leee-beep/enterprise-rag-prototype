import type {
  ApiErrorResponse,
  CompetitorAnalyzeRequest,
  CompetitorAnalyzeResponse,
  HealthResponse,
  ReadinessResponse,
} from "@/types/api";

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_COMPETITOR_API_BASE_URL?.replace(/\/$/, "") ??
  "http://127.0.0.1:8765";

export class ApiClientError extends Error {
  constructor(message: string, readonly code: string, readonly status: number | null) {
    super(message);
    this.name = "ApiClientError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiClientError("The local analysis API could not be reached.", "network_failure", null);
  }
  if (!response.ok) {
    let message = `Request failed with status ${response.status}`;
    let code = "http_error";
    try {
      const body = (await response.json()) as Partial<ApiErrorResponse>;
      if (typeof body.error?.message === "string" && body.error.message.trim()) {
        message = body.error.message;
      }
      if (typeof body.error?.code === "string" && body.error.code.trim()) code = body.error.code;
    } catch { /* Keep the safe status-only message. */ }
    throw new ApiClientError(message, code, response.status);
  }
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiClientError("The local API returned an invalid response.", "invalid_response", response.status);
  }
}

export const competitorApi = {
  health: () => request<HealthResponse>("/health"),
  readiness: () => request<ReadinessResponse>("/ready"),
  analyzeCompetitor: (question: string) =>
    request<CompetitorAnalyzeResponse>("/api/competitor/analyze", {
      method: "POST",
      body: JSON.stringify({ question } satisfies CompetitorAnalyzeRequest),
    }),
};
