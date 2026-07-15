# RagGPT

Self-correcting agentic RAG: upload PDFs, ask questions, and watch the agent
rewrite queries, grade its own retrievals, retry when they're weak, and
verify every citation before answering.

```
Question
   │
   ▼
 Query rewriter (LLM)
   │
   ▼
 Hybrid search: BM25 + vectors ─► rerank
   │
   ▼
 Relevance grader (LLM) ── weak? ─► rewrite & retry
   │ good
   ▼
 Answer with citations ─► citation verifier
   │
   ▼
 Final answer (or "I couldn't find this in your documents")
```

## Stack

Python · Streamlit · LangGraph · Groq · ChromaDB · sentence-transformers · BM25

## Status

In development — see `docs/superpowers/specs/` for the design.
