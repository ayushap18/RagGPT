"""PDF ingestion and hybrid (vector + BM25) retrieval for RagGPT."""

import functools
import os
import sys

# Streamlit Cloud's system sqlite is too old for chromadb; swap in pysqlite3 when present.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
import fitz
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
COLLECTION = "raggpt"


@functools.lru_cache(maxsize=1)
def _embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@functools.lru_cache(maxsize=1)
def _reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _collection(persist_dir):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(COLLECTION)


def _chunk_page(text, page_num):
    """Split one page's text into ~CHUNK_SIZE chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append({"text": chunk, "page": page_num})
        if start + CHUNK_SIZE >= len(text):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ingest_pdf(pdf_path: str, persist_dir: str = "chroma_db") -> dict:
    doc_name = os.path.basename(pdf_path)
    pdf = fitz.open(pdf_path)
    chunks = []
    for i, page in enumerate(pdf):
        chunks.extend(_chunk_page(page.get_text(), i + 1))
    n_pages = len(pdf)
    pdf.close()
    if not chunks:
        raise ValueError("no extractable text")

    texts = [c["text"] for c in chunks]
    embeddings = _embedder().encode(texts).tolist()
    col = _collection(persist_dir)
    # re-ingesting the same filename replaces its old chunks
    col.delete(where={"doc": doc_name})
    col.add(
        ids=[f"{doc_name}-{i}" for i in range(len(chunks))],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"doc": doc_name, "page": c["page"]} for c in chunks],
    )
    return {"doc": doc_name, "pages": n_pages, "chunks": len(chunks)}


class HybridRetriever:
    def __init__(self, persist_dir: str = "chroma_db"):
        self.col = _collection(persist_dir)
        self._bm25 = None
        self._bm25_count = -1  # collection size the cached index was built at

    def _bm25_index(self):
        count = self.col.count()
        if count != self._bm25_count:
            data = self.col.get(include=["documents", "metadatas"])
            self._corpus = [
                {"id": i, "text": t, "doc": m["doc"], "page": m["page"]}
                for i, t, m in zip(data["ids"], data["documents"], data["metadatas"])
            ]
            self._bm25 = BM25Okapi([c["text"].lower().split() for c in self._corpus])
            self._bm25_count = count
        return self._bm25

    def search(self, query: str, k: int = 5) -> list[dict]:
        bm25 = self._bm25_index()
        if not self._corpus:
            return []
        by_id = {c["id"]: c for c in self._corpus}

        # vector top 20
        emb = _embedder().encode([query]).tolist()
        res = self.col.query(query_embeddings=emb, n_results=min(20, len(self._corpus)))
        vec_ids = res["ids"][0]

        # BM25 top 20
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        bm25_ids = [self._corpus[i]["id"] for i in ranked[:20]]

        # reciprocal rank fusion
        rrf = {}
        for ids in (vec_ids, bm25_ids):
            for rank, cid in enumerate(ids):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (60 + rank)
        candidates = [by_id[cid] for cid in sorted(rrf, key=rrf.get, reverse=True)]

        # cross-encoder rerank
        ce_scores = _reranker().predict([(query, c["text"]) for c in candidates])
        reranked = sorted(zip(ce_scores, candidates), key=lambda p: p[0], reverse=True)
        return [
            {"text": c["text"], "doc": c["doc"], "page": c["page"], "score": float(s)}
            for s, c in reranked[:k]
        ]


def get_retriever(persist_dir: str = "chroma_db") -> HybridRetriever:
    return HybridRetriever(persist_dir)
