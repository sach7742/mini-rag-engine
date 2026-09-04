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
└─────────────────┘      └─────────────────┘      └──────────────────┘

```

### Core Components

* **Document Ingestion (`parser.py`)**: Implements sliding-window text chunking with configurable overlap parameters to retain semantic context across boundaries.
* **Vector Embeddings (`embedder.py`)**: Maps text chunks into a 384-dimensional dense vector space using `sentence-transformers/all-MiniLM-L6-v2`.
* **Similarity Search (`retriever.py`)**: Uses pure NumPy Cosine Similarity calculation for fast rank-ordered document retrieval without an external vector database.
* **Local LLM Pipeline (`main.py`)**: Constructs structured context prompts and queries a local `Llama 3.2` model running via Ollama.

---

## Benchmarks & Retrieval Results

### Vector Search Performance (`retriever.py`)

| Test Query | Top-1 Chunk Score | Top-2 Chunk Score | Retrieval Latency |
| --- | --- | --- | --- |
| *"What is Sachin's TOEIC score?"* | **0.8124** | **0.6431** | **~12ms** |
| *"Where is the Institute of Technologists located?"* | **0.8540** | **0.5120** | **~10ms** |

### Execution Metrics

* **Embedding Generation**: ~45ms for 4 document chunks (on Apple Silicon M-series).
* **Inference Latency**: ~350ms total turn-around time using local Ollama Llama 3.2.

---

## Getting Started

### Prerequisites

* Python 3.9+
* [Ollama](https://ollama.com/) running locally

### Setup & Installation

1. **Clone repository**:

```bash
git clone [https://github.com/sach7742/mini-rag-engine.git](https://github.com/sach7742/mini-rag-engine.git)
cd mini-rag-engine

```

2. **Initialize virtual environment**:

```bash
python3 -m venv venv
source venv/bin/activate

```

3. **Install dependencies**:

```bash
pip install -r requirements.txt

```

4. **Pull local model**:

```bash
ollama pull llama3.2

```

5. **Run interactive RAG CLI**:

```bash
python main.py

```

---

## Interactive Terminal Session Output

```text
==========================================
  MINI-RAG-ENGINE (LOCAL LLM VIA OLLAMA)  
==========================================

[1/3] Loading & Chunking Document...
[2/3] Generating Vector Embeddings...
[3/3] System Ready!

Ask a question: Where is the Institute of Technologists located?

[STEP 1] Retrieving relevant context...
Top Retrieved Document Chunks:
  [1] (Score: 0.8540) "The Institute of Technologists is located in Saitama, Japan."

[STEP 2] Llama 3.2 is reading context and generating final answer...

=== FINAL AI ANSWER ===
The Institute of Technologists is located in Saitama, Japan.
=============================================

```

---

## Key Takeaways & Design Choices

* **Zero Framework Abstraction**: Direct mathematical vector operations without relying on third-party orchestration libraries.
* **100% Offline & Private**: Runs vector embeddings and LLM inference locally on Apple Silicon without third-party API dependencies or cloud costs.

```

---
