# RAG PDF Assistant

> **Ask questions about any PDF - answered by AI, grounded in your document.**  
> Fully local. No API keys. No cloud. Just your machine.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?logo=chainlink)
![FAISS](https://img.shields.io/badge/FAISS-CPU-orange)
![Gradio](https://img.shields.io/badge/Gradio-4.x-ff6b6b?logo=gradio)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that gives an LLM access to external knowledge by:

1. **Retrieving** the most relevant passages from a document
2. **Augmenting** the LLM's prompt with those passages
3. **Generating** an answer grounded in real content - not hallucinated

This prevents hallucinations and makes the model's answers verifiable and traceable back to the source.

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────┐
│  PyPDF Loader   │  ← Extract raw text from pages
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Text Splitter  │  ← Chunk into 500-char windows (50-char overlap)
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│  HuggingFace Embeddings      │  ← all-MiniLM-L6-v2 (384-dim vectors)
│  sentence-transformers       │
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────┐
│   FAISS Index   │  ← Fast similarity search over all chunks
└────────┬────────┘
         │  User Query
         ▼
┌──────────────────┐
│  Top-K Retrieval │  ← Return 4 most semantically similar chunks
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  Qwen2.5-0.5B-Instruct   │  ← Generate answer from context + query
└──────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/rag-pdf-assistant.git
cd rag-pdf-assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Gradio app
cd app
python app.py
```

Open your browser at `http://localhost:7860`, upload a PDF, and start asking!

You can also run a single query from the command line without launching the UI:

```bash
cd app
python rag_pipeline.py ../data/sample_pdfs/your.pdf "What is this document about?"
```

---

## Project Structure

```
rag-pdf-assistant/
├── app/
│   ├── app.py            # Gradio demo UI
│   └── rag_pipeline.py   # Core RAG logic (load, chunk, embed, retrieve, answer)
├── notebooks/
│   └── rag_walkthrough.ipynb  # Step-by-step explanation with visualizations
├── data/
│   └── sample_pdfs/      # Drop test PDFs here
├── requirements.txt
└── README.md
```

---

## Models Used

| Component | Model | Why |
|---|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Fast, accurate, 384-dim, CPU-friendly |
| Vector Store | `FAISS (CPU)` | Millisecond similarity search, no server needed |
| LLM | `Qwen/Qwen2.5-0.5B-Instruct` | Small (~500M params), instruction-tuned, runs on CPU |

All models are downloaded automatically via HuggingFace on first run (~1 GB total).

---

## Notebook Walkthrough

The [`notebooks/rag_walkthrough.ipynb`](notebooks/rag_walkthrough.ipynb) covers:

- PDF loading and raw text inspection
- Chunk size analysis with distribution plots
- Embedding vectors and dimensionality
- FAISS index construction
- Retrieval with similarity scores
- Answer generation with Qwen2.5
- **Bonus**: PCA visualization of chunk embeddings vs. query

---

## Key Design Decisions

- **Chunk overlap (50 chars)**: Prevents context from being cut off at chunk boundaries
- **Normalized embeddings**: Cosine similarity equals dot product on unit vectors — faster
- **`stuff`-style context assembly**: Simple and effective for short documents; for very long docs, `map_reduce` is better
- **Top-K = 4**: Balances context window size vs. answer coverage
- **Deterministic decoding (`do_sample=False`)**: Same question → same answer, easier to debug

---

## Extending This Project

- Swap `Qwen2.5-0.5B-Instruct` for a 7B model via `llama.cpp` for much better answers
- Add `BM25` hybrid retrieval alongside dense embeddings for better recall
- Persist the FAISS index to disk (`save_vectorstore` / `load_vectorstore` are already there) to avoid re-embedding on every session
- Add a re-ranking step with a cross-encoder model

---

## 📄 License

MIT — free to use, modify, and share.
