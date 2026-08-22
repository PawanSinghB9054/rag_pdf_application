from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings import init_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate


# =========================
# 1. PDF Load
# =========================

docs = PyPDFLoader("AI & ML DIGITAL NOTES.pdf")
data = docs.load()


# =========================
# 2. Split Text
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=21,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(data)

print("Total chunks:", len(chunks))


# =========================
# 3. Embedding Model
# =========================

embeddings = init_embeddings("mistralai:mistral-embed")


# =========================
# 4. Chroma Vector DB
# =========================

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Vector store ready!")
print("Total vectors:", vector_store._collection.count())


# =========================
# 5. Create Retriever
# =========================

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


# =========================
# 6. User Query
# =========================

query = "What is RNN?"


# =========================
# 7. Retrieve Relevant Chunks
# =========================

results = retriever.invoke(query)


# =========================
# 8. Combine Retrieved Chunks
# =========================

context = "\n\n".join(
    doc.page_content for doc in results
)


# =========================
# 9. LLM
# =========================

llm = init_chat_model(
    "mistral-small-latest",
    model_provider="mistralai"
)


# =========================
# 10. Prompt
# =========================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.
Answer the user's question using only the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}"""
    ),
    (
        "human",
        "{question}"
    )
])


# =========================
# 11. Send Context + Query to LLM
# =========================

messages = prompt.format_messages(
    context=context,
    question=query
)

response = llm.invoke(messages)


# =========================
# 12. Final Answer
# =========================

print("\nAnswer:\n")
print(response.content)