# Enterprise RAG Prototype

A local-first, modular Retrieval-Augmented Generation engineering prototype.

## Current Status

The repository implements document ingestion, chunking, provider-selectable embeddings, in-memory vector search, and provider-independent retrieval. It does **not** yet implement an end-to-end natural-language question-answering pipeline.

### Completed

- Standard Python `src` layout and editable installation
- Environment-based configuration with delayed Gemini API-key validation
- Recursive TXT and Markdown ingestion
- Configurable JSON and JSONL structured extraction
- Deterministic character-based chunking with metadata preservation
- Provider-neutral `EmbeddingClient` and `GenerationClient` interfaces
- Gemini embedding and generation adapters
- Ollama embedding and non-streaming generation adapters
- Independent embedding and generation provider selection
- Shared empty-vector, finite-number, dimension, and generated-text validation
- In-memory FAISS `IndexFlatL2` build and scored search
- Provider-independent `Retriever` with Top-K results and normalized metadata
- Injectable `retrieve` CLI command formatting for score, source, and chunk
- Fully offline tests with fake SDK clients and HTTP transports

### Not Yet Implemented

- PDF ingestion
- Retriever orchestration and query-to-index search flow
- Complete RAG pipeline
- FAISS persistence, manifest, and index lifecycle management
- CLI application
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

Generation clients can accept a prompt and return text, but they are not connected to retrieval. The project therefore cannot yet answer a natural-language question end to end.

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
- `data/private/`, generated indexes, logs, and caches are ignored.
- `data/samples/` and `tests/fixtures/` contain public fictional data.
- Real embedding requests send chunk text to the selected provider; do not process confidential or personal data without authorization.

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

The index is currently in-memory only. A new independent process cannot automatically access an index created by an earlier process because persistence and standalone index loading are not implemented. Running the module without an injected Retriever returns a concise configuration error without an internal traceback. A standalone CLI workflow belongs to the later Pipeline/Persistence milestone.