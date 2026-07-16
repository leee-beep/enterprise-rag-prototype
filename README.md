# enterprise-rag-prototype
A modular Retrieval-Augmented Generation (RAG) prototype built with LangChain, Gemini and FAISS.
## Project Objective

This project aims to build a modular RAG pipeline that can:

- Load multiple public PDF documents
- Split documents into reusable text chunks
- Generate embeddings with Gemini
- Store and search vectors with FAISS
- Retrieve relevant context
- Generate grounded answers with citations

The prototype is designed so that the knowledge base can be replaced without rewriting the entire pipeline.

## Current Progress

- [x] Basic text chunking
- [x] LangChain Document conversion
- [x] Gemini embeddings
- [x] FAISS vector store
- [x] Semantic retrieval
- [x] Gemini answer generation
- [ ] Multi-document PDF loading
- [ ] Source and page metadata
- [ ] Citation output
- [ ] Persistent FAISS index
- [ ] Retrieval test set

## Tech Stack

- Python
- Google Colab
- LangChain
- Gemini API
- FAISS
- GitHub

## Planned Architecture

```text
PDF documents
      ↓
Document loader
      ↓
Text chunking
      ↓
Gemini embeddings
      ↓
FAISS vector store
      ↓
Retriever
      ↓
Gemini
      ↓
Grounded answer with citations


