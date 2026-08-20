# Enterprise RAG Prototype — Historical Development Note

> This file preserves a concise record of the project's early engineering foundation. It is not a complete milestone chronology. See [README.md](README.md) for the current competitor-intelligence architecture, setup, capabilities, and limitations.

## Early modular RAG foundation

- TXT, Markdown, JSON, and JSONL ingestion
- Deterministic document and chunk identities
- Structure-aware chunking with metadata preservation
- `EmbeddedChunk` and FAISS `IndexFlatL2`
- Persisted index metadata and provider/model compatibility checks
- Offline test baseline

## Local provider foundation

- Independent embedding and generation provider settings
- Gemini and Ollama adapters behind shared client protocols
- Factory-owned provider selection
- Ollama `/api/embed` batching and non-streaming `/api/generate`
- Provider response, vector dimension, finite-value, timeout, and connection validation
- Fake SDK clients and transports for offline tests

## Retrieval foundation

- Provider-independent query embedding and retrieval
- FAISS scored search with stable chunk metadata
- Persisted index load and validation
- Generic retrieval, generation, and one-shot RAG pipeline boundaries

The project later evolved into the local Competitor Intelligence RAG application documented in the root README. The original Colab notebook remains historical reference only; modular code under `src/enterprise_rag/` is the implementation source of truth.
