# Enterprise RAG Prototype

A local-first, modular Retrieval-Augmented Generation engineering prototype.

## Current Status

The repository implements document ingestion, chunking, provider-selectable
embeddings, FAISS vector search and persistence, provider-independent retrieval,
and a one-shot RAG question-answering pipeline. The application can now build
a persisted index from local documents and requires explicitly configured
model providers for real embedding or generation calls.

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
- Runnable `build-index`, persisted-index `ask`, and provider-level CLI commands
- Fully offline tests with fake SDK clients and HTTP transports

### Not Yet Implemented

- PDF ingestion
- Automated index rebuilding and lifecycle management
- Source citations
- Retrieval and answer-quality evaluation
- API or UI
- Live Gemini and Ollama integration tests

## Current Data Flow

```text
External Markdown / MDX
        → Document Import
        → data/raw_documents/
        → Document Cleaning
        → data/documents/

TXT / MD / JSON / JSONL
        → LoadedDocument
        → DocumentChunk
        → Selected Embedding Provider (Gemini or Ollama)
        → EmbeddedChunk
        → In-memory FAISS
        → Persisted FAISS index and build manifest
        → Ask
        → Retrieval
        → Generation
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

Document embeddings used to build a FAISS index and query embeddings used to
search it must use the same embedding provider, model, and vector space.
Generation is independent and can use either provider. New indexes include a
versioned `index_manifest.json`; `ask` rejects a provider/model mismatch before
querying. Legacy indexes without this manifest remain loadable but cannot
receive that compatibility check.

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

## Build Index

The standalone indexing service reads cleaned documents, chunks them, calls the
configured embedding provider, builds an in-memory FAISS `IndexFlatL2`, and
safely publishes the persisted index. Defaults come from `DOCUMENTS_DIR` and
`VECTOR_STORE_DIR` in `.env`:

```powershell
python -m enterprise_rag build-index
```

Paths can be overridden explicitly:

```powershell
python -m enterprise_rag build-index `
  --input data\documents `
  --output data\vector_store `
  --verbose
```

If the output directory already exists, the command stops without modifying
it. Review the path and explicitly request replacement:

```powershell
python -m enterprise_rag build-index `
  --input data\documents `
  --output data\vector_store `
  --overwrite `
  --verbose
```

Index creation happens in a sibling temporary directory. Only a complete,
reloadable FAISS index, metadata file, and build manifest replace the final
directory. A failed build does not delete the previous index.

Gemini indexing requires `GEMINI_API_KEY` and sends document chunks to the
configured Gemini embedding model. Ollama indexing requires the configured
local Ollama service and embedding model to already be available. The project
does not download models, start Ollama, or silently switch providers.

Successful output includes:

```text
Documents loaded: ...
Chunks created: ...
Vectors embedded: ...
Embedding dimension: ...
Index saved to: ...
```

The generated `index_manifest.json` records its schema version, build time,
source identifier, embedding provider/model/dimension, chunk settings,
document count, and chunk count. Generated indexes under `data/vector_store/`
are local artifacts and are ignored by Git.

After building an index, ask a one-shot question with the same embedding
provider and model:

```powershell
python -m enterprise_rag ask `
  --index-path data\vector_store `
  "What is retrieval-augmented generation?"
```

This command performs retrieval and generation through the configured
providers. It does not rebuild the index automatically.

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
  --output data\documents\langchain `
  --strict
```

The cleaner preserves code and useful text while removing presentation-layer
MDX syntax. It is a conservative first-pass converter, not a complete MDX
renderer. Cleaning does not create embeddings or a FAISS index. Raw and cleaned
third-party content are ignored by Git by default, and the user remains
responsible for the source documents' license and permitted use.

Strict mode is the safe default. It preflights every source, reports all
invalid files, and publishes nothing unless the whole batch is valid. When a
trusted upstream corpus contains known malformed documents, explicitly use
non-strict mode:

```powershell
python scripts/clean_documents.py `
  --input data\raw_documents\langchain `
  --output data\documents\langchain `
  --skip-invalid `
  --verbose
```

Non-strict mode records every invalid or empty source and its reason in the
versioned `cleaning_manifest.json`, then publishes all valid documents.
Cleaning is prepared in a sibling temporary directory; the complete document
tree and manifest replace the final output together. A failed preflight or
publish does not damage an existing complete dataset and does not leave a
manifest-free partial output.

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
still requires an injected `Retriever`; it remains a presentation boundary for
tests and application composition. `build-index` is now the explicit command
that loads documents and calls the selected embedding provider. Missing
configuration or indexing failures produce a concise error without an internal
traceback.
