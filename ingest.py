
import os
from dotenv import load_dotenv
from supabase import create_client

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

load_dotenv()

# 1. Connect to Supabase
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Supabase credentials are missing in .env")

supabase = create_client(url, key)

# 2. Load PDF
loader = PyPDFLoader("data/company_policy.pdf")
documents = loader.load()

# 3. Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")

# 4. Generate embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5. Store chunks and embeddings in Supabase
vectorstore = SupabaseVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents"
)

print("Documents successfully stored in Supabase!")
