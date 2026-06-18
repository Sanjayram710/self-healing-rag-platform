from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os

DB_PATH = "chroma_db"

def get_vectorstore():
    embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

    return Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )