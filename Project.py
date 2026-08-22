from dotenv import load_dotenv
load_dotenv()

from langchain.embeddings import init_embeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate


# Embeddings
embedding = init_embeddings("mistralai:mistral-embed")


# Load existing Vector DB
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding
)


# Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# LLM
llm = ChatMistralAI(
    model="mistral-small-latest"
)


# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the question using ONLY the context provided below.

If the answer is not present in the context,Then give ans with the help of your intelligence"

Context:
{context}

Question:
{question}

Answer:
""")


# User question
question = input("Ask your question: ")


# Retrieve relevant documents
docs = retriever.invoke(question)


# Convert documents into context
context = "\n\n".join(
    doc.page_content for doc in docs
)


# Send context + question to LLM
response = llm.invoke(
    prompt.format(
        context=context,
        question=question
    )
)


# Final answer
print("\nAnswer:")
print(response.content)