# Mini RAG Engine (`mini-rag-engine`)

A zero-framework, local Retrieval-Augmented Generation (RAG) pipeline built from scratch in Python. This project demonstrates core vector search mechanics, sliding-window text chunking, dense embedding vectorization, and local LLM context synthesis without high-level abstractions like LangChain or LlamaIndex.

---

## Technical Architecture

```text
┌─────────────────┐      ┌─────────────────┐      ┌──────────────────┐
│  Raw Documents  │ ───► │ Sliding Window  │ ───► │  Sentence-       │
│  (.txt format)  │      │ Text Parser     │      │  Transformers    │
└─────────────────┘      └─────────────────┘      └──────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐      ┌─────────────────┐      ┌──────────────────┐
│  Ollama LLM     │ ◄─── │ NumPy Cosine    │ ◄─── │  384-Dim Dense   │
│  (Llama 3.2)    │      │ Similarity      │      │  Vector Matrix   │
└─────────────────┘      └─────────────────┘      └──────────────────┘# badge test
