"""Query dashboard: ask questions, see answer + citations + confidence, and
compare hybrid vs dense-only retrieval side by side."""
import os
import requests
import streamlit as st

API = os.environ.get("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Hybrid RAG", layout="wide")
st.title("Hybrid Search RAG")

q = st.text_input("Ask a question about the indexed docs")
compare = st.checkbox("Compare hybrid vs dense-only", value=True)
verify = st.checkbox("Verify citations", value=True)


def render(col, title, payload):
    col.subheader(title)
    r = requests.post(f"{API}/v1/ask", json=payload, timeout=120).json()
    col.markdown(f"**Answered:** {r['answered']}")
    col.write(r["answer"])
    c = r["confidence"]
    col.metric("Composite confidence", c["composite"])
    col.caption(f"retrieval={c['retrieval_confidence']} | "
                f"citations={c['citation_coverage']} | "
                f"completeness={c['answer_completeness']}")
    if r["citations"]:
        col.markdown("**Citations**")
        for cit in r["citations"]:
            flag = "OK" if cit.get("supported") else "UNVERIFIED"
            col.markdown(f"- [{cit['marker']}] {cit['source_document']} — *{flag}*")
    with col.expander("Retrieved chunks"):
        for i, rc in enumerate(r["retrieved"], 1):
            col.markdown(f"**[{i}]** {rc['chunk']['source_document']} "
                         f"(rerank={rc.get('rerank_score')})")
            col.write(rc["chunk"]["text"][:500])


if st.button("Ask") and q:
    if compare:
        c1, c2 = st.columns(2)
        render(c1, "Hybrid", {"question": q, "hybrid": True, "verify_citations": verify})
        render(c2, "Dense-only", {"question": q, "hybrid": False, "verify_citations": verify})
    else:
        render(st, "Answer", {"question": q, "hybrid": True, "verify_citations": verify})
