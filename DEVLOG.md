# Enterprise RAG Prototype 開發紀錄

## 目前完成

- Project Structure
  - `src` layout
  - `pyproject.toml` editable installation

- Config
  - Settings
  - `.env` 載入
  - Gemini API Key 延遲驗證
  - JSON / JSONL extraction settings

- Models
  - LoadedDocument
  - DocumentChunk
  - EmbeddedChunk
  - DocumentLoadWarning

- Document Ingestion
  - 遞迴載入 `.md`、`.txt`、`.json`、`.jsonl`
  - UTF-8 與 UTF-8 BOM
  - 穩定 document ID
  - 空白文件與結構化 warning
  - JSON structured extraction：auto、record、recursive

- Chunking
  - 確定性字元切割
  - 段落與換行邊界
  - 穩定 chunk ID 與 overlap
  - metadata 保留

- Embedding
  - EmbeddingClient abstraction
  - Batch Embedding
  - Google Gen AI Adapter
  - FakeEmbeddingClient
  - 向量數量、空值與維度驗證

- Vector Store
  - In-memory FAISS IndexFlatL2
  - EmbeddedChunk 對應關係
  - 最小 top-k search
  - 向量與查詢維度驗證

- Testing
  - 58 個離線測試全部通過
  - Embedding 測試使用 fake client
  - Vector Store 測試使用假向量
  - 不呼叫真實 Gemini API

## 目前完整流程

```text
TXT / MD / JSON / JSONL
↓
LoadedDocument
↓
DocumentChunk
↓
EmbeddedChunk
↓
In-memory FaissVectorStore (IndexFlatL2)
```

目前流程中斷在 query retrieval 與 generation 之前，尚不能完成自然語言問答。

## 尚未完成

- PDF ingestion
- Ollama providers
- Retriever
- Generation
- Complete RAG Pipeline
- Vector Store persistence and index lifecycle
- CLI application
- Citations
- Evaluation
- API / UI