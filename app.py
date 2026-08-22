"""
Streamlit frontend for the PDF RAG project.

This app combines the two original scripts (PRoject_database.py + project.py)
into a single UI:
  1) Upload PDF(s) from the sidebar
  2) Click "Process PDF(s)" -> load -> split -> embed -> store in Chroma
  3) Ask questions in the chat box below -> the retriever pulls relevant
     context -> ChatMistralAI generates the answer

To run:
    streamlit run app.py

Your .env file must already have MISTRAL_API_KEY (and any other keys
required by langchain's init_embeddings), just like in the original
scripts.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the question using ONLY the context provided below.
If the answer is not present in the context, then answer with the help of your own knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
)

st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄", layout="wide")

# ---------------- Custom theme: pink/black main area, red/black sidebar ----------------
st.markdown(
    """
    <style>
    /* Main content area: pink + black gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #1a0010 0%, #3d0021 45%, #1a0010 100%);
        color: #ffe6f2;
    }

    /* Sidebar: red + black gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a0000 0%, #3d0000 50%, #1a0000 100%);
        color: #ffe0e0;
    }
    [data-testid="stSidebar"] * {
        color: #ffe0e0 !important;
    }

    /* Headings and captions in the main area */
    h1, h2, h3, [data-testid="stCaptionContainer"] {
        color: #ff4fa3 !important;
    }

    /* Buttons: pink/red accent */
    .stButton > button {
        background-color: #ff2d78;
        color: white;
        border: 1px solid #ff2d78;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #b3003c;
        border: 1px solid #b3003c;
        color: white;
    }

    /* Chat input and expanders */
    [data-testid="stChatInput"] {
        background-color: #2a0015;
    }
    .streamlit-expanderHeader {
        color: #ff4fa3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- Cached resources ----------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return init_embeddings("mistralai:mistral-embed")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-latest")


# ---------------- Core RAG helpers ----------------
def build_vector_store(uploaded_files, chunk_size, chunk_overlap):
    """Logic from PRoject_database.py: load -> split -> embed -> store.

    NOTE: This vector store only lives in memory (RAM) and is never
    saved to disk. As soon as the session/app restarts (or the "Clear"
    button is clicked), this data disappears automatically.
    """
    all_docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            # Track which uploaded file each page came from
            for d in docs:
                d.metadata["source_file"] = uploaded_file.name
            all_docs.extend(docs)
        finally:
            os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(all_docs)

    # persist_directory intentionally omitted -> this Chroma collection
    # stays only in this process's memory, nothing is saved to disk.
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
    )
    return vector_store, len(chunks)


def answer_question(vector_store, question, k):
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    response = get_llm().invoke(PROMPT.format(context=context, question=question))
    return response.content, docs


# ---------------- Session state ----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📄 PDF Setup")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    with st.expander("Chunking settings"):
        chunk_size = st.number_input("Chunk size", value=1500, min_value=200, step=100)
        chunk_overlap = st.number_input("Chunk overlap", value=500, min_value=0, step=50)

    if st.button("🚀 Process PDF(s)", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Reading PDF, creating chunks, generating embeddings..."):
                vector_store, num_chunks = build_vector_store(
                    uploaded_files, chunk_size, chunk_overlap
                )
                st.session_state.vector_store = vector_store
            st.success(f"Vector DB ready ✅ ({num_chunks} chunks) — only for this session, nothing is saved anywhere.")

    st.divider()

    if st.button("🗑️ Delete my data now", use_container_width=True):
        st.session_state.vector_store = None
        st.session_state.messages = []
        st.success("Vector DB and chat history deleted.")
        st.rerun()

    st.divider()
    k = st.slider("Number of chunks to retrieve (k)", min_value=1, max_value=10, value=3)

    if st.session_state.vector_store is not None:
        st.success("Status: Vector DB ready")
    else:
        st.info("Status: No vector DB loaded")

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- Main chat area ----------------
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF from the sidebar, process it, then ask questions below.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Type your question here...")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload and process a PDF from the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = answer_question(st.session_state.vector_store, question, k)
                st.markdown(answer)
                with st.expander("View source chunks"):
                    for i, d in enumerate(docs, start=1):
                        page = d.metadata.get("page", "?")
                        src = d.metadata.get("source_file", "")
                        st.markdown(f"**Chunk {i}** — `{src}` (page {page})")
                        st.text(d.page_content[:400] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
