import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chunk size between 1000-1500 characters with overlap 200-300 for context continuity
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 250

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    return splitter.split_documents(documents)

load_dotenv()

logger = logging.getLogger('performance')

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)

def generate_answer(question, context):

    # Log context details for debugging
    if context:
        chunk_count = context.count("\n") + 1
        logger.info(f"[ANSWER] Received {chunk_count} chunks in context for question: '{question}'")
        # Log preview of first 200 chars
        preview = context[:200].replace('\n', ' ')
        logger.info(f"[ANSWER] Context preview: {preview}")

    prompt = f"""
    You are a helpful AI assistant.

    Use ALL the context below to answer the question comprehensively.

    If information is missing, say you don't know.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return response.content