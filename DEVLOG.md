# Enterprise RAG Prototype 開發紀錄

## Milestone 1：Modular RAG Baseline

- TXT、Markdown、JSON、JSONL ingestion
- Character-based chunking 與 metadata 保留
- EmbeddedChunk 與 in-memory FAISS IndexFlatL2
- Offline test baseline

## Milestone 2：Local LLM Foundation

- Settings
  - Embedding 與 Generation provider 可獨立選擇
  - Gemini / Ollama models、base URL、timeout
  - Provider 正規化與驗證
  - Gemini API Key 延遲驗證

- Provider Architecture
  - EmbeddingClient
  - GenerationClient
  - GeminiEmbeddingClient
  - GeminiGenerationClient
  - OllamaEmbeddingClient
  - OllamaGenerationClient
  - create_embedding_client
  - create_generation_client

- Ollama HTTP
  - Python standard library `urllib.request`
  - `/api/embed` batch embedding
  - `/api/generate` non-streaming generation
  - HTTP、timeout、connection、JSON 與 response schema 錯誤分類

- Validation
  - 空向量、向量數量、維度、NaN、Infinity
  - 空白 generation response
  - 所有 provider tests 使用 mock SDK 或 fake transport，不呼叫網路

## 目前資料流

```text
TXT / MD / JSON / JSONL
↓
LoadedDocument
↓
DocumentChunk
↓
Gemini 或 Ollama Embedding
↓
EmbeddedChunk
↓
In-memory FAISS
```

Generation provider 已可獨立建立並接受 prompt，但尚未連接 Retriever context。

## 重要限制

- 建立 index 與查詢 index 必須使用相同 embedding provider、model 與向量空間。
- 尚未實作 index manifest，因此目前需由使用者自行維持相容性。
- 尚未執行真實 Gemini 或 Ollama integration test。
- Retriever、完整 Pipeline、citations、CLI、PDF、API/UI 尚未完成。