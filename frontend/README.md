# Competitor Intelligence Workspace

Responsive Next.js frontend for the local Competitor Intelligence RAG application. It checks backend readiness, submits research questions to the FastAPI analysis endpoint, and presents domain status, grounded prose, exact-string financial claims, structured citations, and evidence provenance.

## Stack

- Next.js 16 and React 19
- TypeScript
- Tailwind CSS
- Lucide React

## Local API integration

The browser uses `NEXT_PUBLIC_COMPETITOR_API_BASE_URL`, which defaults to `http://127.0.0.1:8765`.

- `GET /ready` drives the Connecting, Ready, and Unavailable states.
- `POST /api/competitor/analyze` sends one field: the natural-language `question`.
- The backend CORS allowlist must contain the browser origin, normally `http://localhost:3000` or `http://127.0.0.1:3000`.

The frontend is presentation-only for financial data. API financial values remain strings and are not parsed, recalculated, rounded, sorted, or converted into live charts. Citation identity comes from structured API objects rather than answer-text parsing.

## Run locally

Start the FastAPI service from the repository root first. Then:

```powershell
Set-Location frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open <http://localhost:3000>.

The full live workflow requires the local backend, Ollama, compatible company indexes, and curated financial facts. Those private resources are intentionally not bundled with the frontend or stored in Git.

## Session privacy

Browser `sessionStorage` retains only recent question text, timestamp, and domain status for the current tab session. Answers, citations, financial claims, evidence metadata, provider responses, and private source information are not persisted.

## Validation

```powershell
npm run lint
npm run typecheck
npm run build
```

Any committed portfolio screenshots must use clearly identified synthetic demonstration data. Do not capture private source content, financial values, paths, URLs, or local configuration.
