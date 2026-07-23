# Enterprise RAG Prototype

A local-first, modular Retrieval-Augmented Generation engineering prototype.

## Current Status

The repository currently implements document ingestion through in-memory vector search. It does **not** yet provide natural-language question answering, retrieval orchestration, or answer generation.

### Completed

- Standard Python `src` layout and editable installation
- Environment-based configuration with delayed Gemini API-key validation
- Recursive TXT and Markdown ingestion
- Configurable JSON and JSONL structured extraction
- UTF-8 and UTF-8 BOM handling
- Deterministic character-based chunking with overlap and metadata preservation
- Provider-neutral embedding abstraction
- Google Gen AI embedding adapter with batch processing
- Empty-vector and vector-dimension validation
- In-memory FAISS `IndexFlatL2` build and search
- Offline tests using fake embedding clients and vectors
- 58 offline tests passing

### Not Yet Implemented

- PDF ingestion
- Ollama embedding or generation providers
- Retriever orchestration and query embedding flow
- Answer generation
- Complete RAG pipeline
- FAISS persistence and index lifecycle management
- CLI application
- Source citations
- Retrieval and answer-quality evaluation
- API or UI

## Current Data Flow

```text
TXT / MD / JSON / JSONL
        ↓
LoadedDocument
        ↓
DocumentChunk
        ↓
EmbeddedChunk
        ↓
In-memory FAISS
```

The implemented flow stops before query retrieval and generation. The project cannot yet answer a natural-language question end to end.

## Repository Layout

```text
enterprise-rag-prototype/
├── data/
│   ├── documents/          # Local source documents; ignored except .gitkeep
│   └── samples/            # Public, non-sensitive ingestion samples
├── notebooks/              # Preserved Colab proof of concept
├── src/enterprise_rag/     # Local Python package
├── tests/                  # Offline automated tests and fixtures
├── .env.example            # Safe environment-variable template
├── pyproject.toml          # Package metadata and dependency source
└── requirements.txt        # Editable development install via pyproject.toml
```

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`requirements.txt` installs the editable project with its test dependencies. Runtime and test dependency declarations are maintained in `pyproject.toml`.

A Gemini API key is not required for configuration, document loading, JSON extraction, chunking, FAISS tests, or mocked embedding tests. Before a real Google embedding request, copy `.env.example` to `.env` and replace the placeholder:

```dotenv
GEMINI_API_KEY=your_real_key_here
```

Never commit `.env`, private documents, generated indexes, logs, or caches.

## Running Tests

Use a writable temporary directory outside the project when local `.test-tmp` permissions are unreliable:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp=<writable-temp-directory> -p no:cacheprovider
```

All tests are designed to run offline and do not call the Gemini API.

## Data and Security

- `data/documents/` is local-only except for `.gitkeep`.
- `data/private/`, generated FAISS files, index directories, and Qdrant storage are ignored.
- `data/samples/` and `tests/fixtures/` contain public fictional test data and remain trackable.
- Real embedding requests send chunk text to the configured provider; do not use confidential or personal data without authorization.

## Original Notebook

The Colab notebook is preserved as historical proof of concept. It is not the source of truth for the modular local implementation.