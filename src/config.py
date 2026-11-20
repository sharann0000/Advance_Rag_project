
import os
from dotenv import load_dotenv
from supabase import create_client

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore

load_dotenv()

# Embedding model (must match ingestion)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Supabase connection
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("Supabase credentials are missing in .env")

supabase = create_client(url, key)

# Load existing Supabase vector store
vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents"
)

# Keep the same retriever interface
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Existing LLM configuration
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
