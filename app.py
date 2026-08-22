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

# ---------------- Theme: "Ink & Highlighter" ----------------
# Deep ink-navy surfaces with a warm amber "highlighter" accent, like
# annotating a PDF at night. Covers Streamlit's header/footer chrome too,
# not just the content area, so there's no leftover default-black strip.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink-0: #0B0E14;
        --ink-1: #10141D;
        --ink-2: #171C28;
        --panel: #1B2130;
        --border: #2A3142;
        --amber: #F2A73B;
        --amber-dim: #8A5F1F;
        --text: #E9E6DD;
        --text-muted: #8D93A3;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Kill Streamlit's default red/orange top decoration line */
    [data-testid="stDecoration"] {
        background-image: linear-gradient(90deg, var(--amber), var(--ink-2));
    }

    /* Top toolbar (Deploy / menu bar) */
    [data-testid="stHeader"] {
        background-color: var(--ink-0);
        border-bottom: 1px solid var(--border);
    }
    [data-testid="stHeader"] * {
        color: var(--text) !important;
    }

    /* Main content area */
    [data-testid="stAppViewContainer"] {
        background: var(--ink-1);
        color: var(--text);
    }
    [data-testid="stMain"] {
        background: var(--ink-1);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--ink-0);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] h2 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: 0.02em;
    }

    /* Bottom chat-input bar (was showing as a plain black strip) */
    [data-testid="stBottom"] > div {
        background: var(--ink-0);
        border-top: 1px solid var(--border);
    }
    [data-testid="stChatInput"] {
        background-color: var(--ink-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
    }

    /* Titles */
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: var(--text) !important;
        border-bottom: 3px solid var(--amber);
        display: inline-block;
        padding-bottom: 6px;
    }
    h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text) !important;
    }
    [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--panel);
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: var(--amber);
        color: var(--amber);
    }
    /* Primary "Process PDF(s)" button gets the highlighter treatment */
    .stButton > button[kind="primary"] {
        background-color: var(--amber);
        border: 1px solid var(--amber);
        color: #241505;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #ffbf5c;
        border-color: #ffbf5c;
        color: #241505;
    }

    /* File uploader, expanders, sliders */
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--panel) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 10px;
    }
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        background-color: var(--panel) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
    }
    [data-testid="stSlider"] [role="slider"] {
        background-color: var(--amber) !important;
    }

    /* Status boxes (success / info / warning) */
    [data-testid="stAlertContentSuccess"] { color: #C8E6A0 !important; }
    [data-testid="stAlertContentInfo"] { color: #A9C4E8 !important; }
    [data-testid="stAlertContentWarning"] { color: #F2D08A !important; }
    div[data-baseweb="notification"] {
        background-color: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* Chat bubbles: assistant = highlighted margin note, user = plain ink card */
    [data-testid="stChatMessage"] {
        background-color: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 4px 6px;
        margin-bottom: 10px;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border-left: 3px solid var(--amber);
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
