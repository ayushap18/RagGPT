# RagGPT — Self-Correcting Agentic RAG

**Date:** 2026-07-16
**Status:** Approved

## Overview

A Streamlit web app where users upload PDFs and ask questions. Instead of
naive retrieve-and-answer, a LangGraph agent rewrites the query, runs hybrid
search, grades its own retrievals, retries when they are weak, and verifies
every citation before showing the answer. The sidebar shows each pipeline
step live.

## Pipeline (LangGraph state graph)

1. **Ingest** (on upload): PyMuPDF extracts text per page → chunks of ~800
   characters with overlap → embedded with a local `sentence-transformers`
   model → stored in ChromaDB. A BM25 keyword index is built alongside.
   No embedding API needed; Groq is used only for reasoning steps.
2. **Rewrite:** Groq LLM turns a vague question into a precise search query.
3. **Retrieve:** BM25 + vector search merged with reciprocal rank fusion,
   top ~20, reranked to top 5 with a local cross-encoder.
4. **Grade:** LLM scores each chunk relevant/irrelevant. Fewer than 2
   relevant → loop back to Rewrite (max 2 retries), then answer honestly:
   "I couldn't find this in your documents."
5. **Generate:** Answer with inline citations `[1]`, `[2]` mapped to page
   numbers.
6. **Verify:** LLM checks each cited claim is supported by its chunk; one
   regeneration if verification fails.

## Tech stack

- Python, Streamlit UI
- LangGraph for the agent pipeline
- Groq API: `llama-3.3-70b-versatile` for answers, `llama-3.1-8b-instant`
  for grader/rewriter calls
- ChromaDB (local, persistent) for vectors
- `sentence-transformers` for embeddings and cross-encoder reranking
- `rank_bm25` for keyword search
- PyMuPDF for PDF text extraction

## File layout

```
RagGPT/
  app.py       # Streamlit UI: upload, chat, live pipeline trace in sidebar
  graph.py     # LangGraph pipeline + prompts
  ingest.py    # PDF → chunks → vector + BM25 indexes
  eval.py      # runnable check: sample PDF + golden Q&As
  README.md    # architecture diagram + setup
```

## Error handling

- Groq free-tier rate limits → retry with exponential backoff.
- Scanned/image-only PDFs (no extractable text) → clear warning instead of
  a silent empty index.

## Testing

`eval.py` ingests a bundled sample PDF and asserts the pipeline answers a
few golden questions with correct citations. No test framework; one
runnable script.

## Deployment

Streamlit Community Cloud (free tier), `GROQ_API_KEY` provided via
Streamlit secrets. Gives a live public URL for the portfolio.

## Deliberately out of scope (v1)

User accounts, chat history persistence, multi-model support, OCR for
scanned PDFs.
