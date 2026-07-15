"""Self-correcting RAG pipeline: rewrite -> retrieve -> grade -> generate -> verify."""

import json
import random
import re
import time
from typing import Any, Callable, Optional, TypedDict

import groq
from langgraph.graph import END, START, StateGraph

ANSWER_MODEL = "llama-3.3-70b-versatile"
FAST_MODEL = "llama-3.1-8b-instant"
MAX_REWRITE_RETRIES = 2
NOT_FOUND_ANSWER = "I couldn't find this in your documents."

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = groq.Groq()
    return _client


def _chat(model: str, prompt: str) -> str:
    """One-shot completion with exponential backoff on rate limits."""
    attempts = 6
    for attempt in range(attempts):
        try:
            resp = _get_client().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            return (resp.choices[0].message.content or "").strip()
        except groq.RateLimitError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(2**attempt + random.random(), 30))


class State(TypedDict, total=False):
    question: str
    query: str
    chunks: list
    relevant: list
    answer: str
    citations: list
    found: bool
    steps: list
    retries: int  # rewrites done so far
    verify_count: int
    feedback: str  # verifier feedback for regeneration
    retriever: Any
    on_step: Optional[Callable]


def _step(state: State, node: str, detail: str) -> None:
    # steps is mutated in place: the list object is shared across the whole run.
    step = {"node": node, "detail": detail}
    state["steps"].append(step)
    if state.get("on_step"):
        state["on_step"](step)


def _rewrite(state: State) -> dict:
    prompt = (
        "Rewrite this question as a short, precise search query for document "
        f"retrieval. Return only the query, nothing else.\n\nQuestion: {state['question']}"
    )
    if state.get("query"):
        prompt += (
            f"\n\nA previous query found nothing useful: {state['query']}\n"
            "Try different wording."
        )
    query = _chat(FAST_MODEL, prompt)
    _step(state, "rewrite", query)
    return {"query": query, "retries": state["retries"] + 1}


def _retrieve(state: State) -> dict:
    chunks = state["retriever"].search(state["query"], k=5)
    _step(state, "retrieve", f"retrieved {len(chunks)} chunks")
    return {"chunks": chunks}


def _parse_labels(raw: str, n: int) -> list:
    """Extract n relevant/irrelevant labels from possibly-malformed LLM output."""
    labels = []
    try:
        arr = json.loads(re.search(r"\[.*\]", raw, re.S).group())
        for x in arr:
            x = str(x).lower()
            labels.append("relevant" if "relevant" in x and "irrelevant" not in x else "irrelevant")
    except Exception:
        labels = re.findall(r"\b(irrelevant|relevant)\b", raw.lower())
    return (labels + ["irrelevant"] * n)[:n]


def _grade(state: State) -> dict:
    chunks = state["chunks"]
    listing = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    prompt = (
        f"Question: {state['question']}\n\nChunks:\n{listing}\n\n"
        "For each chunk, decide whether it helps answer the question. Reply with "
        f'ONLY a JSON array of {len(chunks)} strings, each "relevant" or "irrelevant".'
    )
    labels = _parse_labels(_chat(FAST_MODEL, prompt), len(chunks))
    relevant = [c for c, lab in zip(chunks, labels) if lab == "relevant"]
    _step(state, "grade", f"{len(relevant)}/{len(chunks)} chunks relevant")
    return {"relevant": relevant}


def _after_grade(state: State) -> str:
    if len(state["relevant"]) >= 2:
        return "generate"
    if state["retries"] <= MAX_REWRITE_RETRIES:
        return "rewrite"
    return "giveup"


def _giveup(state: State) -> dict:
    _step(state, "giveup", NOT_FOUND_ANSWER)
    return {"answer": NOT_FOUND_ANSWER, "citations": [], "found": False}


def _generate(state: State) -> dict:
    chunks = state["relevant"]
    listing = "\n\n".join(
        f"[{i + 1}] (from {c['doc']}, page {c['page']}) {c['text']}"
        for i, c in enumerate(chunks)
    )
    prompt = (
        "Answer the question using ONLY the numbered chunks below. Cite every "
        "claim inline with its chunk number, like [1] or [2]. If the chunks do "
        "not contain the answer, say so.\n\n"
        f"Chunks:\n{listing}\n\nQuestion: {state['question']}"
    )
    if state.get("feedback"):
        prompt += (
            f"\n\nYour previous answer failed verification:\n{state['feedback']}\n"
            "Rewrite the answer making only claims directly supported by the chunks."
        )
    answer = _chat(ANSWER_MODEL, prompt)
    cited = sorted(
        {int(m) for m in re.findall(r"\[(\d+)\]", answer) if 1 <= int(m) <= len(chunks)}
    )
    citations = [
        {"n": n, "doc": chunks[n - 1]["doc"], "page": chunks[n - 1]["page"], "text": chunks[n - 1]["text"]}
        for n in cited
    ]
    _step(state, "generate", answer)
    return {"answer": answer, "citations": citations, "found": True}


def _verify(state: State) -> dict:
    chunks = state["relevant"]
    listing = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(chunks))
    prompt = (
        "Check the answer below against the source chunks. Every claim followed "
        "by a citation [n] must be supported by chunk [n]. Reply exactly "
        '"PASS" if all cited claims are supported, otherwise "FAIL: <what is unsupported>".\n\n'
        f"Chunks:\n{listing}\n\nAnswer:\n{state['answer']}"
    )
    raw = _chat(ANSWER_MODEL, prompt)
    passed = "FAIL" not in raw.upper()
    _step(state, "verify", raw if raw else "PASS")
    return {
        "verify_count": state["verify_count"] + 1,
        "feedback": "" if passed else raw,
    }


def _after_verify(state: State) -> str:
    # regenerate at most once
    if state["feedback"] and state["verify_count"] < 2:
        return "generate"
    return END


def _build_graph():
    g = StateGraph(State)
    g.add_node("rewrite", _rewrite)
    g.add_node("retrieve", _retrieve)
    g.add_node("grade", _grade)
    g.add_node("generate", _generate)
    g.add_node("verify", _verify)
    g.add_node("giveup", _giveup)
    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", _after_grade, {"generate": "generate", "rewrite": "rewrite", "giveup": "giveup"})
    g.add_edge("generate", "verify")
    g.add_conditional_edges("verify", _after_verify, {"generate": "generate", END: END})
    g.add_edge("giveup", END)
    return g.compile()


_GRAPH = _build_graph()


def answer_question(question: str, retriever, on_step=None) -> dict:
    final = _GRAPH.invoke(
        {
            "question": question,
            "retriever": retriever,
            "on_step": on_step,
            "steps": [],
            "retries": 0,
            "verify_count": 0,
            "query": "",
            "feedback": "",
        }
    )
    return {
        "answer": final["answer"],
        "citations": final["citations"],
        "steps": final["steps"],
        "found": final["found"],
    }
