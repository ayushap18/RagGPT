"""End-to-end eval: generate a sample PDF, ingest it, answer golden questions live.

Run: python eval.py  (requires a valid GROQ_API_KEY). Exits nonzero on failure.
"""

import tempfile
import os

import fitz

from graph import answer_question
from ingest import get_retriever, ingest_pdf

# Each page states its key fact at the start AND the end, so with per-page
# 800-char chunking + overlap both chunks of the page carry the fact and the
# grader can find >= 2 relevant chunks.
PAGES = [
    (
        "The Zephyr Institute was founded in 2011 in Lisbon, Portugal, by the "
        "marine biologist Dr. Rosa Almeida. The institute grew out of a small "
        "university research group focused on coastal ecology. Its mission is "
        "to monitor and protect Atlantic coastal ecosystems using low-cost, "
        "open-source instrumentation. Early work concentrated on water quality "
        "sampling in the Tagus estuary, where volunteers collected weekly "
        "measurements of salinity, turbidity, and dissolved oxygen. Over the "
        "following decade the institute expanded its scope to include acoustic "
        "monitoring of dolphin populations and long-term studies of intertidal "
        "biodiversity. The organisation maintains partnerships with several "
        "European universities and publishes all of its datasets openly. "
        "To repeat the founding details for the record: the Zephyr Institute "
        "was founded in 2011 in Lisbon, Portugal, by Dr. Rosa Almeida."
    ),
    (
        "The Zephyr Institute has an annual operating budget of 3.2 million "
        "euros, funded primarily by the Atlantic Science Foundation. Roughly "
        "half of the budget pays for research staff salaries, while the rest "
        "covers equipment, vessel time, and public outreach programmes. The "
        "institute deliberately keeps administrative overhead below ten "
        "percent, and every annual report is audited independently and "
        "published on its website. Additional smaller grants come from "
        "municipal governments along the Portuguese coast and from a citizen "
        "membership scheme with about four thousand supporters. Financial "
        "planning is done on a rolling three-year horizon so that long-term "
        "monitoring programmes are never interrupted by single-year funding "
        "gaps. In summary, the annual operating budget of the Zephyr "
        "Institute is 3.2 million euros, funded by the Atlantic Science "
        "Foundation."
    ),
    (
        "The Zephyr Institute operates a fleet of 47 underwater drones that "
        "map seagrass meadows along the Algarve coast. Each drone is a small "
        "autonomous vehicle carrying a downward-facing camera, a side-scan "
        "sonar unit, and sensors for temperature and chlorophyll. The drones "
        "run pre-programmed survey transects and surface every few hours to "
        "upload imagery over a satellite link. Machine-learning models then "
        "stitch the imagery into seasonal maps of seagrass extent, which feed "
        "directly into national marine conservation planning. Battery swaps "
        "and hull maintenance are handled from a converted fishing vessel "
        "that serves as the fleet's mobile base. Survey campaigns run from "
        "spring through early autumn, when water clarity along the southern "
        "coast is at its best, and the resulting maps are compared year on "
        "year to detect meadow growth or decline. To restate the key figure: "
        "the institute's fleet consists of 47 underwater drones mapping "
        "seagrass meadows along the Algarve coast."
    ),
]

# (question, keywords that must appear in the answer, page that must be cited)
GOLDEN = [
    ("When and where was the Zephyr Institute founded?", ["2011", "lisbon"], 1),
    ("What is the annual budget of the Zephyr Institute?", ["3.2"], 2),
    ("How many underwater drones does the Zephyr Institute operate?", ["47"], 3),
]

OFF_TOPIC = "Who won the 1998 FIFA World Cup?"


def build_pdf(path: str) -> None:
    doc = fitz.open()
    for text in PAGES:
        page = doc.new_page()
        rc = page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=11)
        assert rc >= 0, "page text overflowed the textbox"
    doc.save(path)
    doc.close()


def main() -> None:
    work = tempfile.mkdtemp(prefix="raggpt_eval_")
    pdf_path = os.path.join(work, "zephyr.pdf")
    persist_dir = os.path.join(work, "chroma")
    build_pdf(pdf_path)

    info = ingest_pdf(pdf_path, persist_dir=persist_dir)
    print(f"ingested: {info}")
    assert info["doc"] == "zephyr.pdf"
    assert info["pages"] == 3
    assert info["chunks"] >= 6, f"expected >=2 chunks per page, got {info['chunks']}"

    retriever = get_retriever(persist_dir)

    for question, keywords, page in GOLDEN:
        print(f"\nQ: {question}")
        result = answer_question(question, retriever)
        print(f"A: {result['answer']}")
        assert result["found"], f"pipeline gave up on: {question}"
        answer = result["answer"].lower()
        for kw in keywords:
            assert kw in answer, f"answer missing {kw!r} for: {question}"
        cited_pages = [c["page"] for c in result["citations"]]
        assert page in cited_pages, (
            f"expected citation of page {page}, got pages {cited_pages}"
        )

    print(f"\nQ (off-topic): {OFF_TOPIC}")
    result = answer_question(OFF_TOPIC, retriever)
    print(f"A: {result['answer']}")
    assert not result["found"], "off-topic question should return found=False"

    print("\nPASS")


if __name__ == "__main__":
    main()
