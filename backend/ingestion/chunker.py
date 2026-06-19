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