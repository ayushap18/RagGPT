"""RagGPT Streamlit UI: upload PDFs, chat, live pipeline trace."""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RagGPT", page_icon="📄", layout="wide")

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to a .env file or your environment.")
    st.stop()

from graph import answer_question  # noqa: E402
from ingest import get_retriever, ingest_pdf  # noqa: E402

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = []  # list of {"doc", "pages", "chunks"}


def render_citations(citations):
    if not citations:
        return
    with st.expander(f"Citations ({len(citations)})"):
        for c in citations:
            st.markdown(f"**[{c['n']}] {c['doc']} — page {c['page']}**")
            st.caption(c["text"])


# ---- Sidebar: upload/index + pipeline trace ----
with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )
    if st.button("Index documents"):
        if not uploads:
            st.warning("Choose at least one PDF first.")
        for up in uploads or []:
            tmp_dir = tempfile.mkdtemp(prefix="raggpt_")
            pdf_path = os.path.join(tmp_dir, up.name)
            with open(pdf_path, "wb") as f:
                f.write(up.getbuffer())
            try:
                with st.spinner(f"Indexing {up.name}…"):
                    info = ingest_pdf(pdf_path)
                st.session_state.indexed.append(info)
            except ValueError:
                st.error(
                    f"{up.name}: no extractable text — this looks like a "
                    "scanned/image-only PDF, which is not supported."
                )

    for info in st.session_state.indexed:
        st.caption(f"{info['doc']}: {info['pages']} pages, {info['chunks']} chunks")

    st.header("Pipeline trace")
    trace_area = st.container()

# ---- Main: chat ----
st.title("RagGPT")

if not st.session_state.indexed:
    st.info("Upload PDFs in the sidebar and click **Index documents** to get started.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_citations(msg.get("citations"))

question = st.chat_input("Ask a question about your documents")
if question:
    if not st.session_state.indexed:
        st.warning("No documents indexed yet — upload and index a PDF first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with trace_area.status("Running pipeline…", expanded=True) as status:

            def on_step(step):
                status.write(f"**{step['node']}** — {step['detail']}")

            result = answer_question(question, get_retriever(), on_step=on_step)
            status.update(label="Pipeline finished", state="complete")

        with st.chat_message("assistant"):
            st.markdown(result["answer"])
            render_citations(result["citations"])

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"],
            }
        )
