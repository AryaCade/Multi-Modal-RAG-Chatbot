import streamlit as st
from multi_modal_ingest import multi_modal_ingest, normalize_element
from vector_store import build_faiss, load_faiss
from qa_engine import answer_question
import tempfile
import os

st.set_page_config(page_title="📄 Multi-Modal RAG Chatbot", layout="wide")

# ----------------------------
# SESSION STATE STRUCTURE
# ----------------------------
if "faiss_store" not in st.session_state:
    st.session_state.faiss_store = None

if "ready" not in st.session_state:
    st.session_state.ready = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("📂 Document Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF", 
    type=["pdf"]
)

if uploaded_file is not None:

    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_pdf_path = tmp.name

    st.sidebar.info(f"📄 File uploaded: {uploaded_file.name}")

    # Process PDF
    with st.spinner("Extracting and indexing PDF..."):
        file_bytes = uploaded_file.getvalue()
        elements = multi_modal_ingest(file_bytes)

        chunks = []
        for el in elements:
            ch = normalize_element(el, file_bytes)
            if ch:
                chunks.append(ch)

        # Build FAISS index
        build_faiss(chunks, "vector_store_user")
        st.session_state.faiss_store = load_faiss("vector_store_user")
        st.session_state.ready = True
        st.session_state.pdf_name = uploaded_file.name
        
    st.sidebar.success("✅ PDF indexed and ready to query!")


# ----------------------------
# MAIN UI
# ----------------------------

st.title("🤖 Multi-Modal RAG Chatbot")
st.caption("Upload a PDF and ask questions about it")

if st.session_state.ready:
    st.success(f"Current Document: {st.session_state.pdf_name}")

else:
    st.warning("Upload a PDF to get started")


# User question
query = st.chat_input("Ask a question")


# ----------------------------
# Handle Query
# ----------------------------

if query:

    if not st.session_state.ready:
        st.error("Upload a PDF first!")
    else:
        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Searching document..."):
            answer, citations = answer_question(query, store=st.session_state.faiss_store)

        with st.chat_message("assistant"):
            st.write(answer)

            #if citations:
            #    st.write("### 🔖 Citations")
            #    for c in citations:
            #        st.write("- " + c)
