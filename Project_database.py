#Load pdf 
# Split chunks 
# Embedding 
# Vector db

from dotenv import load_dotenv 

load_dotenv() 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings
from langchain_community.vectorstores import Chroma

# PDF Load #
docs = PyPDFLoader("handom.pdf")
data = docs.load()


# Split data 
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500 ,
    chunk_overlap = 500 
)

chunks = splitter.split_documents(data)
print(" size of chunks : " , len(chunks))

# Embeddings 
embeddings = init_embeddings("mistralai:mistral-embed")

# Vector db
vector_store = Chroma.from_documents(
    documents = chunks , 
    embedding = embeddings ,
    persist_directory= "chroma_db"
)

print("Vector is ready")