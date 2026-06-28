import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

# Fallback/override for older SQLite versions on Linux (Chroma requires sqlite3 >= 3.35.0)
try:
    import sqlite3

    if sqlite3.sqlite_version_info < (3, 35, 0):
        try:
            import pysqlite3

            sys.modules["sqlite3"] = pysqlite3
            logger.info("[Chroma] Successfully patched sqlite3 with pysqlite3 for compatibility.")
        except ImportError:
            logger.warning(
                "[Chroma] SQLite version %s is older than required (3.35.0) and pysqlite3 is not available. ChromaDB initialization may fail.",
                sqlite3.sqlite_version,
            )
except ImportError:
    pass

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "chroma_db"

_embedding_model = None
_vectorstore = None


def _load_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    start = time.perf_counter()
    logger.info("[Chroma] Loading embedding model...")
    _embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": os.getenv("HF_TOKEN")},
    )
    logger.info("[Chroma] Embedding model loaded in %.2fs", time.perf_counter() - start)
    return _embedding_model


def _load_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    start = time.perf_counter()
    logger.info("[Chroma] Loading vector store from %s...", DB_PATH)
    _vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=_load_embedding_model(),
    )
    logger.info("[Chroma] Vector store ready in %.2fs", time.perf_counter() - start)
    return _vectorstore


def initialize_vectorstore():
    return _load_vectorstore()


def get_vectorstore():
    return _load_vectorstore()
