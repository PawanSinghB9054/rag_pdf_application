"""
Streamlit frontend for the PDF RAG project.

Ye app tumhare do scripts (PRoject_database.py + project.py) ka logic
ek hi jagah combine karta hai:
  1) Sidebar se PDF(s) upload karo
  2) "Process PDF(s)" dabao -> load -> split -> embed -> Chroma me store
  3) Neeche chat box me sawal poocho -> retriever context nikalega ->
     ChatMistralAI answer dega

Run karne ke liye:
    streamlit run app.py

.env file me MISTRAL_API_KEY (aur langchain init_embeddings ke liye
zaroori keys) already set honi chahiye, jaisa tumhare original scripts
me tha.
"""

import os
import shutil
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

PERSIST_DIR = "chroma_db"

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


# ---------------- Cached resources ----------------
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return init_embeddings("mistralai:mistral-embed")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-latest")


# ---------------- Core RAG helpers ----------------
def build_vector_store(uploaded_files, chunk_size, chunk_overlap, clear_old):
    """PRoject_database.py ka logic: load -> split -> embed -> store."""
    if clear_old and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

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

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )
    return vector_store, len(chunks)


def load_existing_vector_store():
    """project.py ka logic: disk se already-bani DB load karna."""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


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
        "PDF upload karo", type=["pdf"], accept_multiple_files=True
    )

    with st.expander("Chunking settings"):
        chunk_size = st.number_input("Chunk size", value=1500, min_value=200, step=100)
        chunk_overlap = st.number_input("Chunk overlap", value=500, min_value=0, step=50)
        clear_old = st.checkbox("Process se pehle purani DB clear karo", value=True)

    if st.button("🚀 Process PDF(s)", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Pehle kam se kam ek PDF upload karo.")
        else:
            with st.spinner("PDF padh raha hu, chunks bana raha hu, embeddings generate ho rahi hain..."):
                vector_store, num_chunks = build_vector_store(
                    uploaded_files, chunk_size, chunk_overlap, clear_old
                )
                st.session_state.vector_store = vector_store
            st.success(f"Vector DB taiyar ✅ ({num_chunks} chunks)")

    st.divider()

    if st.button("📂 Load existing Vector DB", use_container_width=True):
        if os.path.exists(PERSIST_DIR):
            st.session_state.vector_store = load_existing_vector_store()
            st.success("Purani chroma_db load ho gayi ✅")
        else:
            st.error("chroma_db folder abhi tak nahi bana. Pehle ek PDF process karo.")

    st.divider()
    k = st.slider("Retrieve karne wale chunks (k)", min_value=1, max_value=10, value=3)

    if st.session_state.vector_store is not None:
        st.success("Status: Vector DB ready")
    else:
        st.info("Status: Koi vector DB load nahi hai")

    if st.button("🗑️ Chat history clear karo", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------- Main chat area ----------------
st.title("📄 Apne PDF se chat karo")
st.caption("Sidebar se PDF upload karke process karo, phir neeche sawal poocho.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Apna sawal yahan likho...")

if question:
    if st.session_state.vector_store is None:
        st.warning("Pehle sidebar se PDF upload karke process karo.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Soch raha hu..."):
                answer, docs = answer_question(st.session_state.vector_store, question, k)
                st.markdown(answer)
                with st.expander("Source chunks dekho"):
                    for i, d in enumerate(docs, start=1):
                        page = d.metadata.get("page", "?")
                        src = d.metadata.get("source_file", "")
                        st.markdown(f"**Chunk {i}** — `{src}` (page {page})")
                        st.text(d.page_content[:400] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})