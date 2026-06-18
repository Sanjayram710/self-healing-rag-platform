from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "chroma_db"

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )