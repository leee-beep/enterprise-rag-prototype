# Competitor Intelligence Workspace

Local-first showcase frontend for the Enterprise RAG Prototype. This milestone provides the visual system, responsive workspace, typed API boundary, and interaction states; it does not yet submit analysis requests to the backend.

## Run locally

```powershell
npm install
Copy-Item .env.example .env.local
npm run dev
```

Open <http://localhost:3000>. The API base URL defaults to `http://127.0.0.1:8765` and can be configured with `NEXT_PUBLIC_COMPETITOR_API_BASE_URL`.

## Validation

```powershell
npm run lint
npm run typecheck
npm run build
```

All displayed company narrative and financial values in the preview are explicitly synthetic. No private document text, financial facts, indexes, manifests, API keys, or local absolute paths belong in this frontend.
