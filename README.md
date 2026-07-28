# Enterprise RAG Prototype

A local-first, modular Retrieval-Augmented Generation engineering prototype.

## Current Status

The repository implements document ingestion, chunking, provider-selectable
embeddings, FAISS vector search and persistence, provider-independent retrieval,
and a one-shot RAG question-answering pipeline. Application assembly still
requires an existing index and explicitly configured model providers.

### Completed

- Standard Python `src` layout and editable installation
- Environment-based configuration with delayed Gemini API-key validation
- Recursive TXT and Markdown ingestion
- Deterministic local Markdown/MDX document import with a versioned manifest
- Configurable JSON and JSONL structured extraction
- Deterministic character-based chunking with metadata preservation
- Provider-neutral `EmbeddingClient` and `GenerationClient` interfaces
- Gemini embedding and generation adapters
- Ollama embedding and non-streaming generation adapters
- Independent embedding and generation provider selection
- Shared empty-vector, finite-number, dimension, and generated-text validation
- In-memory FAISS `IndexFlatL2` build and scored search
- Versioned FAISS index and chunk-metadata persistence
- Provider-independent `Retriever` with Top-K results and normalized metadata
- Context formatting, prompt construction, and one-shot `RAGPipeline`
- `retrieve`, `generate`, `ask`, and injected `build-index` CLI boundaries
- Fully offline tests with fake SDK clients and HTTP transports

### Not Yet Implemented

- PDF ingestion
- Full MDX cleaning and conversion into ingestion-ready documents
- Standalone document-to-index application assembly
- Automated index rebuilding and lifecycle management
- Source citations
- Retrieval and answer-quality evaluation
- API or UI
- Live Gemini and Ollama integration tests

## Current Data Flow

```text
TXT / MD / JSON / JSONL
        ↓
LoadedDocument
        ↓
DocumentChunk
        ↓
Selected Embedding Provider (Gemini or Ollama)
        ↓
EmbeddedChunk
        ↓
In-memory FAISS
```

The one-shot RAG pipeline connects retrieval to generation. It does not add
citations, memory, streaming, reranking, or multi-turn chat.

## Provider Architecture

```text
embeddings.py              generation.py
EmbeddingClient            GenerationClient
       ↓                           ↓
factory.create_*_client(Settings)
       ↓                           ↓
Gemini adapter             Ollama adapter
```

Configure providers independently in `.env`:

```dotenv
EMBEDDING_PROVIDER=gemini
GENERATION_PROVIDER=ollama
```

Supported combinations:

- Gemini embedding + Gemini generation
- Ollama embedding + Ollama generation
- Ollama embedding + Gemini generation
- Gemini embedding + Ollama generation

Only the selected provider is called for an operation. The application does not automatically call both providers and never silently falls back to another provider.

### Index Compatibility Rule

Document embeddings used to build a FAISS index and query embeddings used to search it must use the same embedding provider, model, and vector space. Generation is independent and can use either provider. An index manifest is not implemented yet, so this compatibility must currently be managed by the user.

### Ollama Setup

Ollama must be installed, started, and supplied with the configured models by the user. This project does not download models or start the Ollama service. The adapters use the local HTTP API at `OLLAMA_BASE_URL` and default to `http://localhost:11434`.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Runtime and test dependencies are declared in `pyproject.toml`; `requirements.txt` performs an editable development install. A Gemini key is checked only immediately before a real Gemini operation. Ollama operations never require a Gemini key.

## Running Tests

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=<writable-temp-directory> -p no:cacheprovider
```

Provider tests inject fake SDK clients or HTTP transports. They do not call Gemini, localhost, Ollama, or any external network.

## Data and Security

- `data/documents/` is local-only except for `.gitkeep`.
- `data/raw_documents/` contains close-to-source third-party imports and is
  local-only except for `.gitkeep`.
- `data/private/`, generated indexes, logs, and caches are ignored.
- `data/samples/` and `tests/fixtures/` contain public fictional data.
- Real embedding requests send chunk text to the selected provider; do not process confidential or personal data without authorization.

## Document Import

`data/raw_documents/` stores selected, close-to-source third-party Markdown or
MDX files. It is separate from `data/documents/`, which is the input area for
documents that are ready for the ingestion and indexing workflow. Importing raw
documents does not clean MDX, create embeddings, or build a FAISS index.

Preview the deterministic LangChain document selection without writing files:

```powershell
python scripts/import_docs.py `
  --source ..\langchain-docs-source\src `
  --output data\raw_documents\langchain `
  --dry-run `
  --verbose
```

Run the import after reviewing the preview:

```powershell
python scripts/import_docs.py `
  --source ..\langchain-docs-source\src `
  --output data\raw_documents\langchain
```

Imported third-party documentation remains subject to its original license and
usage terms. The user is responsible for confirming that copying, processing,
and later sending any content to a model provider is permitted. Imported
contents and manifests under `data/raw_documents/` are ignored by Git by
default; the external LangChain repository is never copied into this project.

## Document Cleaning & Normalization

`data/raw_documents/` contains close-to-source imports. The cleaning command
conservatively converts selected `.md`/`.mdx` files into UTF-8 Markdown under
`data/documents/`, where the existing Markdown loader can read them. Preview a
complete in-memory cleaning pass without writing files:

```powershell
python scripts/clean_documents.py `
  --input data\raw_documents\langchain `
  --output data\documents\langchain `
  --dry-run `
  --verbose
```

After reviewing warnings, write cleaned Markdown and the cleaning manifest:

```powershell
python scripts/clean_documents.py `
  --input data\raw_documents\langchain `
  --output data\documents\langchain
```

The cleaner preserves code and useful text while removing presentation-layer
MDX syntax. It is a conservative first-pass converter, not a complete MDX
renderer. Cleaning does not create embeddings or a FAISS index. Raw and cleaned
third-party content are ignored by Git by default, and the user remains
responsible for the source documents' license and permitted use.

## Original Notebook

The Colab notebook is preserved as historical proof of concept and is not the source of truth for the modular implementation.
## Retrieval Score Semantics

Retriever results use `score = 1 / (1 + squared_l2_distance)`. This is a monotonic display/relevance score derived from the raw squared L2 distance returned by FAISS:

- Higher means closer within the current index.
- It is not cosine similarity, a probability, or an accuracy percentage.
- It is only meaningful for comparisons within the same embedding provider, model, vector space, and index.
- Scores must not be compared across models or indexes.

## Retrieve CLI Scope

The `retrieve` subcommand presentation layer accepts a question, calls an already-injected in-memory `Retriever`, and prints Top-K score, source, and chunk text. It does not build or load an index, embed documents, generate an answer, or invoke a pipeline.

Persisted indexes can be loaded by `ask --index-path`. The `retrieve` command
still requires an injected `Retriever`, and `build-index` requires an injected
builder because the CLI does not silently load documents, call an embedding
provider, or rebuild an index. Missing application dependencies produce a
concise configuration error without an internal traceback.
