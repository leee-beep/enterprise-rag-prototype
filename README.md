# Competitor Intelligence RAG Prototype

Built on a modular, privacy-first enterprise RAG foundation.

This repository contains a local-first RAG engineering prototype for retrieving and comparing evidence from official corporate disclosures from:

- Gigabyte (2376)
- ASUS (2357)
- MSI (2377)

The current competitor workflow performs page-aware PDF ingestion, builds a separate local FAISS index for each company, and returns balanced evidence with company, year, page, and chunk provenance. It stops at retrieval: competitor-specific answer synthesis and user-facing citations are not implemented yet.

The Git repository remains named `enterprise-rag-prototype` because the competitor application is built on a reusable generic RAG backend.

## Current status

### Implemented and validated

- Recursive TXT and Markdown loading
- Configurable JSON and JSONL structured extraction
- Deterministic Markdown/MDX import and cleaning
- Deterministic structure-aware chunking with stable chunk identities
- Provider-neutral Gemini and Ollama embedding/generation abstractions
- Bounded Ollama embedding batches
- Generic Gemini/Ollama generation and one-shot `RAGPipeline`
- FAISS `IndexFlatL2` construction, scored search, persistence, and reload
- Page-aware competitor PDF ingestion with extraction-quality checks
- AES-compatible empty-password PDF loading and password-required PDF rejection
- Competitor metadata, SHA-256 validation, and stable source/page/chunk identities
- Read-only competitor PDF preflight: 12 of 12 real PDFs passed
- Separate 2025 ASUS, Gigabyte, and MSI FAISS indexes
- Balanced one-, two-, and three-company retrieval with equal Top-K allocation
- Company-local relevance scores and deterministic company grouping
- Retrieval provenance: company, ticker, year, document, PDF page, source ID, and chunk ID
- `competitor-retrieve` diagnostic CLI
- 326 passing offline tests
- Private PDFs, manifests, extracted text, and generated indexes excluded from Git

### Not yet implemented

- Competitor-specific Qwen comparison synthesis
- User-facing citations or citation verification
- Evidence-quality filtering and deduplication
- `financial_facts.csv` or structured financial comparison
- 2024 competitor indexes
- Qualitative indexing of consolidated financial reports
- Query translation, reranking, or hybrid/BM25 retrieval
- Streamlit or another user interface
- OCR
- Production access controls and operational hardening

## Validated source scope

The private local source collection contains official 2024 and 2025 annual reports and consolidated financial reports for all three companies. All 12 PDFs passed real extraction preflight.

The current qualitative indexes contain **only**:

- ASUS 2025 Annual Report
- Gigabyte 2025 Annual Report
- MSI 2025 Annual Report

The other nine PDFs are not embedded. Consolidated financial reports are retained primarily as authoritative sources for a future structured financial-facts layer.

### Validation results

| Validation | Result |
|---|---:|
| Real PDF preflight | 12 / 12 passed |
| ASUS 2025 index | 461 chunks |
| Gigabyte 2025 index | 1,149 chunks |
| MSI 2025 index | 326 chunks |
| Total indexed chunks | 1,936 |
| Automated test suite | 326 passed |

Real local Ollama indexing and retrieval were manually validated with `nomic-embed-text`. Automated provider tests remain offline and use fake clients or transports; they do not call Ollama, Gemini, localhost, or external networks.

## Competitor architecture

```text
Official competitor PDFs
        |
        v
Private source manifest
        |
        v
Metadata + SHA-256 validation
        |
        v
Page-aware PDF loading
        |
        v
LoadedDocument per extractable page
        |
        v
Structure-aware chunking
        |
        v
Local Ollama embedding
(nomic-embed-text)
        |
        v
Separate FAISS indexes
 +-----------+-----------+-----------+
 | ASUS 2025 | GIGABYTE  | MSI 2025  |
 +-----------+-----------+-----------+
        |
        v
BalancedCompetitorRetriever
(equal Top-K per company)
        |
        v
Company-grouped retrieval evidence
(company / year / page / chunk / local score)

CURRENT SYSTEM STOPS HERE

        |
        v
Planned:
Evidence filtering
Citation-ready context
Qwen competitor synthesis
Financial comparison
User interface
```

## Balanced multi-company retrieval

Each selected company is searched independently against its own index. For:

```text
--companies gigabyte asus msi
--top-k-per-company 2
```

the result order is deterministic:

```text
Gigabyte: Rank 1, Rank 2
ASUS:     Rank 1, Rank 2
MSI:      Rank 1, Rank 2
```

This prevents one company's corpus from consuming the entire Top-K. Raw FAISS-derived scores remain local to each company index and are not globally sorted or treated as calibrated across indexes.

### Competitor retrieval CLI

The diagnostic command loads existing company indexes and returns retrieved evidence:

```powershell
python -m enterprise_rag competitor-retrieve `
  --index-root data\vector_store\competitors `
  --companies gigabyte asus msi `
  --top-k-per-company 2 `
  "Compare the companies' AI strategies."
```

It prints company, company-local rank and score, source title, year, PDF page, chunk ID, and a short preview. It does **not** generate a competitor comparison answer.

Example retrieval questions:

- Compare Gigabyte, ASUS and MSI's AI strategies.
- What does each company identify as a major growth driver?
- Compare the companies' enterprise or server strategies.
- What products or business areas does each company emphasize?
- What does MSI say about AI servers?
- What does Gigabyte say about AI infrastructure?
- What does ASUS say about AI PCs?

Broad English comparison queries against zh-TW reports remain experimental and do not always return equally strong evidence.

## Privacy and data handling

- Real competitor PDFs remain local.
- The private source manifest remains local.
- Extracted page text remains local.
- Generated competitor FAISS indexes remain local.
- `data/private/` and `data/vector_store/` are Git-ignored.
- The repository contains code and synthetic or fictional test fixtures, not the real competitor corpus.

The validated competitor workflow uses Ollama. If `OLLAMA_BASE_URL` points to a trusted local or organization-controlled service, document chunks, queries, and retrieved context remain within that environment. Ollama is not inherently private when configured to use an untrusted remote endpoint.

Gemini remains an optional provider for the reusable generic RAG foundation. Selecting Gemini sends the relevant content to an external service and must follow organizational data-handling policy.

## Local setup

Create a virtual environment, install the editable project and copy the configuration template:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`pyproject.toml` is the dependency source of truth; `requirements.txt` installs the project and test extras in editable mode.

### Provider configuration

Embedding and generation providers are selected independently in `.env`:

```dotenv
EMBEDDING_PROVIDER=ollama
GENERATION_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_EMBEDDING_BATCH_SIZE=32
OLLAMA_CHAT_MODEL=qwen3:8b
OLLAMA_TIMEOUT_SECONDS=180
```

Ollama and the configured models must be installed and started separately by the user. This project does not download models or start the service.

Gemini can be selected for the generic foundation. `GEMINI_API_KEY` is validated only before a real Gemini operation; Ollama operations do not require it. Never commit `.env`.

### Run tests

```powershell
.\.venv\Scripts\python.exe -m pytest `
  --basetemp=C:\tmp\enterprise-rag-tests `
  -p no:cacheprovider `
  -ra
```

## Competitor data setup

Expected private layout:

```text
data/private/competitors/
    source_manifest.json
    sources/
        asus/
            2025/
        gigabyte/
            2025/
        msi/
            2025/
```

The manifest supplies validated company, year, document type, language, official source URL, and safe relative source path metadata. Private source data is intentionally excluded from Git.

Run the read-only preflight before indexing:

```powershell
.\.venv\Scripts\python.exe scripts\preflight_competitor_pdfs.py `
  --source-root data\private\competitors\sources `
  --manifest data\private\competitors\source_manifest.json
```

Preflight validates metadata, SHA-256 identity, encryption compatibility, page extraction, and basic extraction quality. It does not write a corpus or build an index.

## Reusable generic RAG foundation

The generic backend remains usable independently of the competitor application:

```text
TXT / MD / JSON / JSONL
        |
        v
LoadedDocument
        |
        v
DocumentChunk
        |
        v
EmbeddedChunk
        |
        v
FAISS
        |
        v
Retriever
        |
        v
Generic RAGPipeline
        |
        v
Generation provider
```

### Provider and index compatibility

Supported provider combinations include Gemini/Gemini, Ollama/Ollama, Gemini/Ollama, and Ollama/Gemini. Only the selected provider is called; there is no silent fallback.

Document and query embeddings must use the same provider, model, and vector space. New indexes include `index_manifest.json`; loading rejects a known provider/model mismatch before querying. Legacy indexes without a manifest remain loadable but cannot receive this compatibility check. Competitor manifest fields are additive and do not invalidate legacy generic manifests.

### Build a generic index

```powershell
python -m enterprise_rag build-index `
  --input data\documents `
  --output data\vector_store `
  --verbose
```

Use `--overwrite` only after reviewing an existing destination. Index publication is atomic: a complete, reloadable FAISS index, metadata file, and build manifest replace the destination together.

### Ask with a generic persisted index

```powershell
python -m enterprise_rag ask `
  --index-path data\vector_store `
  --top-k 4 `
  "What is retrieval-augmented generation?"
```

This generic command performs retrieval and generation through the configured providers. It does not build an index automatically and is separate from competitor comparison retrieval.

### Generic `retrieve` scope

The `retrieve` subcommand is a presentation boundary for an already-injected in-memory `Retriever`. It does not load an index, embed documents, generate an answer, or invoke a pipeline. Persisted competitor retrieval uses `competitor-retrieve` instead.

## Markdown/MDX import and cleaning

`data/raw_documents/` contains close-to-source third-party imports. `data/documents/` contains cleaned documents ready for generic ingestion. Both are local-only except for their `.gitkeep` files.

Preview an import:

```powershell
python scripts/import_docs.py `
  --source ..\langchain-docs-source\src `
  --output data\raw_documents\langchain `
  --dry-run `
  --verbose
```

Preview conservative MD/MDX cleaning:

```powershell
python scripts/clean_documents.py `
  --input data\raw_documents\langchain `
  --output data\documents\langchain `
  --dry-run `
  --verbose
```

The cleaner preserves useful Markdown and fenced code while removing supported presentation-layer MDX syntax. Strict mode is the default; `--skip-invalid` explicitly permits invalid sources to be skipped and recorded. Importing or cleaning does not create embeddings or indexes.

## Retrieval score semantics

Generic `Retriever` results use:

```text
score = 1 / (1 + squared_l2_distance)
```

The score is a monotonic ranking signal: higher means closer within the same embedding provider, model, vector space, and FAISS index.

It is **not**:

- a probability
- a confidence score
- an accuracy percentage

For balanced competitor retrieval, scores from different company indexes must not be compared as globally calibrated values.

## Known limitations

- English-to-zh-TW retrieval is useful but inconsistent.
- Table-heavy annual-report pages can pollute retrieval.
- Adjacent overlapping chunks may both be returned.
- Balanced Top-K guarantees company coverage, not evidence quality.
- There is no reranker, query translation, or hybrid retrieval.
- Competitor synthesis and user-facing citations are not implemented.
- There is no structured financial-analysis engine or UI.
- OCR is not supported.
- This is an engineering prototype, not a production-ready application.

## Roadmap

1. Improve retrieval quality with table/noise filtering and evidence deduplication.
2. Produce citation-ready evidence.
3. Add guarded Qwen competitor comparison synthesis.
4. Build 2024 historical competitor indexes.
5. Add a curated `financial_facts.csv` layer.
6. Implement structured financial comparison.
7. Add a user-facing interface.
8. Add evaluation, access controls, and production hardening.

## Historical notebook

The original Colab notebook is preserved as historical proof of concept and reference material. It is not the canonical implementation; the modular code under `src/enterprise_rag/` is the source of truth.
