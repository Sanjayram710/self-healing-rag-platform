import asyncio
import json
import logging
import os
import uuid

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
import shutil
from datetime import datetime

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.analytics import (
    append_query_log,
    filter_queries_by_timestamp,
    load_stats,
    log_analytics_summary,
    resolve_analytics_period,
    summarize_analytics,
)
from backend.auth import AuthenticatedUser, verify_firebase_token
from backend.chat_history_store import ChatPersistenceError, create_chat, get_user_chats, get_chat, delete_chat, append_message, update_chat, search_chats
from backend.collections_store import load_collections, create_collection, delete_collection, find_collection
from backend.document_store import (
    delete_from_storage,
    delete_document_meta,
    get_document_meta,
    list_all_document_meta,
    save_document_meta,
    sync_from_storage,
    upload_to_storage,
)
from backend.persistence import get_firestore_client
from backend.settings_store import load_settings, save_settings
from backend.classification import build_direct_response, classify_query
import time
from backend.performance_logger import timed_stage
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.document_loader import load_document
from backend.ingestion.embeddings import store_documents
from backend.vectorstore.chroma import initialize_vectorstore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))

logger = logging.getLogger(__name__)
FALLBACK_MESSAGE = "I couldn't find enough relevant information in the uploaded documents to answer this question confidently."


def _index_document_file(file_path, filename, file_meta):
    documents = load_document(file_path)
    chunks = chunk_documents(documents)

    for chunk in chunks:
        chunk.metadata["user_id"] = file_meta.get("user_id")
        chunk.metadata["filename"] = filename
        if file_meta.get("collection_id"):
            chunk.metadata["collection_id"] = file_meta.get("collection_id")

    store_documents(chunks)
    return documents, chunks

app = FastAPI(title="Self-Healing RAG Platform")

cors_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
if cors_origins_str:
    origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True if origins != ["*"] else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_request_timing(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        logger.info(
            "[Request] %s %s completed in %.2fs",
            request.method,
            request.url.path,
            time.perf_counter() - start,
        )


@app.on_event("startup")
def hydrate_persistent_storage():
    """
    Startup hook - runs once when the FastAPI process starts.

    1. Downloads any uploaded files from Firebase Storage that are not already
       present on the local ephemeral disk.
    2. If the Chroma vector store is empty (fresh Railway container), rebuilds
       it by re-indexing every restored document.
    3. Syncs chat history from Firestore.
    """
    startup_start = time.perf_counter()
    logger.info("[Startup] Beginning application startup...")

    firebase_start = time.perf_counter()
    try:
        get_firestore_client()
        logger.info("[Startup] Firebase client prewarmed in %.2fs", time.perf_counter() - firebase_start)
    except Exception:
        logger.error("[Startup] Firebase client prewarm failed.", exc_info=True)

    vector_start = time.perf_counter()
    try:
        initialize_vectorstore()
        logger.info("[Startup] Vector store prewarmed in %.2fs", time.perf_counter() - vector_start)
    except Exception:
        logger.error("[Startup] Vector store prewarm failed.", exc_info=True)

    restore_start = time.perf_counter()
    logger.info("[Startup] Syncing documents from Firebase Storage...")
    try:
        restored_docs = sync_from_storage()  # list[(filename, meta)]
        logger.info("[Startup] %d documents available locally after sync.", len(restored_docs))
    except Exception:
        logger.error("[Startup] sync_from_storage() failed.", exc_info=True)
        restored_docs = []
    logger.info("[Startup] Storage sync completed in %.2fs", time.perf_counter() - restore_start)

    try:
        vectorstore = initialize_vectorstore()
        existing_docs = vectorstore.get().get("documents", [])
        if not existing_docs:
            logger.info("[Startup] Vector store is empty - rebuilding from %d documents.", len(restored_docs))
            indexed = 0
            for filename, file_meta in restored_docs:
                file_path = os.path.join(UPLOAD_DIR, filename)
                if not os.path.isfile(file_path):
                    logger.warning("[Startup] Skipping %s - file not on disk.", filename)
                    continue
                try:
                    _index_document_file(file_path, filename, file_meta)
                    indexed += 1
                    logger.info("[Startup] Indexed %s into vector store.", filename)
                except Exception:
                    logger.error("[Startup] Failed to index %s.", filename, exc_info=True)
            logger.info("[Startup] Vector store rebuild complete - %d/%d documents indexed.", indexed, len(restored_docs))
        else:
            logger.info("[Startup] Vector store already populated (%d chunks).", len(existing_docs))
    except Exception:
        logger.error("[Startup] Vector store check/rebuild failed.", exc_info=True)

    chat_sync_start = time.perf_counter()
    try:
        from backend.chat_history_store import sync_chats_from_firestore
        sync_chats_from_firestore()
    except Exception:
        logger.warning("[Startup] Chat history sync failed.", exc_info=True)
    logger.info("[Startup] Chat sync phase completed in %.2fs", time.perf_counter() - chat_sync_start)
    logger.info("[Startup] Application startup completed in %.2fs", time.perf_counter() - startup_start)
class Query(BaseModel):
    question: str
    collection_id: str | None = None
    chat_id: str | None = None


@app.get("/api/health")
@app.get("/health")
def health():
    return {"status": "ok", "message": "Self-Healing RAG Running"}


@app.post("/api/documents/upload")
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_id: str | None = None,
    current_user: AuthenticatedUser = Depends(verify_firebase_token),
):
    logger.info(
        "[Upload] Incoming upload request: filename=%s, collection_id=%s, user_id=%s",
        file.filename, collection_id, current_user.uid
    )
    
    allowed_extensions = [".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".xlsx", ".jpg", ".jpeg", ".png"]
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in allowed_extensions:
        logger.warning("[Upload] Unsupported file type: %s", extension)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {extension}. Allowed formats: {', '.join(allowed_extensions)}"
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    logger.info("[Upload] Local destination path: %s", file_path)

    # 1. Write to local disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_size = os.path.getsize(file_path)
        logger.info("[Upload] Local copy saved. Size: %d bytes (%s KB)", file_size, round(file_size / 1024, 2))
    except Exception as e:
        logger.error("[Upload] Failed to write file locally: %s", file.filename, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file on the server disk: {str(e)}"
        )

    # 2. Upload to Firebase Storage (required — this is the permanent store)
    logger.info("[Upload] Persisting %s to Firebase Storage...", file.filename)
    if not upload_to_storage(file.filename):
        logger.error("[Upload] Firebase Storage upload failed for: %s", file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=503,
            detail="Firebase Storage is unavailable. Document was not persisted.",
        )
    logger.info("[Upload] Firebase Storage upload successful: %s", file.filename)

    # 3. Parse and index into Chroma
    try:
        logger.info("[Upload] Parsing file content: %s", file.filename)
        documents = load_document(file_path)
        logger.info("[Upload] Parsing completed. Loaded %d pages.", len(documents))

        logger.info("[Upload] Chunking documents: %s", file.filename)
        chunks = chunk_documents(documents)
        logger.info("[Upload] Chunking completed. Generated %d chunks.", len(chunks))

        for chunk in chunks:
            chunk.metadata["user_id"] = current_user.uid
            chunk.metadata["filename"] = file.filename
            if collection_id:
                chunk.metadata["collection_id"] = collection_id

        logger.info("[Upload] Indexing %d chunks into ChromaDB...", len(chunks))
        store_documents(chunks)
        logger.info("[Upload] ChromaDB indexing completed.")
    except ValueError as val_err:
        logger.warning("[Upload] Validation/OCR error during parsing: %s", str(val_err))
        # Roll back Storage upload and local file
        delete_from_storage(file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=400,
            detail=str(val_err)
        )
    except Exception as exc:
        logger.error("[Upload] Unexpected error during indexing: %s", file.filename, exc_info=True)
        # Roll back Storage upload and local file
        delete_from_storage(file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and index document content: {str(exc)}"
        )

    # 4. Save per-document metadata to Firestore
    ocr_used = any(doc.metadata.get("ocr_used", False) for doc in documents)
    file_meta = {
        "filename": file.filename,
        "pages": len(documents),
        "chunks": len(chunks),
        "collection_id": collection_id,
        "ocr_used": ocr_used,
        "user_id": current_user.uid,
        "size_bytes": file_size,
        "uploaded_at": datetime.now().isoformat(),
        "status": "indexed",
    }
    
    logger.info("[Upload] Saving document metadata to Firestore: %s", file_meta)
    if not save_document_meta(file.filename, file_meta):
        logger.error("[Upload] Firestore metadata update failed for: %s", file.filename)
        # Metadata failed — roll back Storage upload and local file
        delete_from_storage(file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=503,
            detail="Firestore is unavailable. Document metadata was not persisted.",
        )

    logger.info(
        "[Upload] %s uploaded by %s successfully — %d pages, %d chunks.",
        file.filename, current_user.uid, len(documents), len(chunks),
    )
    return {
        "message": "Document uploaded and indexed successfully",
        "filename": file.filename,
        "file_type": extension,
        "pages": len(documents),
        "chunks": len(chunks),
        "size_bytes": file_size,
        "size_kb": round(file_size / 1024, 2),
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "indexed",
    }


@app.get("/api/documents")
@app.get("/documents")
def get_documents(current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    """
    List documents for the current user.
    Uses Firestore as the source of truth; falls back to scanning the local
    uploads directory if Firestore is unavailable.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Primary source: Firestore per-document metadata
    all_meta = list_all_document_meta()  # dict[filename -> meta]

    # Fallback: scan local disk if Firestore returned nothing
    if not all_meta:
        logger.warning("[Documents] Firestore metadata unavailable; falling back to local disk scan.")
        for fname in os.listdir(UPLOAD_DIR):
            fp = os.path.join(UPLOAD_DIR, fname)
            if os.path.isfile(fp):
                all_meta[fname] = {
                    "filename": fname,
                    "user_id": None,
                    "size_bytes": os.path.getsize(fp),
                    "pages": 0,
                    "chunks": 0,
                    "status": "indexed",
                    "uploaded_at": datetime.fromtimestamp(os.path.getctime(fp)).isoformat(),
                }

    documents = []
    for fname, file_meta in all_meta.items():
        # Filter to this user's documents only.
        # uid=None is a fallback-mode stub — show it to whoever is logged in.
        owner = file_meta.get("user_id")
        if owner is not None and owner != current_user.uid:
            continue

        # Get size from disk if available, otherwise use stored metadata
        file_path = os.path.join(UPLOAD_DIR, fname)
        size_bytes = (
            os.path.getsize(file_path)
            if os.path.isfile(file_path)
            else file_meta.get("size_bytes", 0)
        )

        documents.append({
            "id": fname,
            "name": fname,
            "filename": fname,
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
            "pages": file_meta.get("pages", 0),
            "chunks": file_meta.get("chunks", 0),
            "status": file_meta.get("status", "indexed"),
            "uploaded_at": (
                file_meta.get("uploaded_at")
                or datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
                if os.path.isfile(file_path)
                else file_meta.get("uploaded_at", "")
            ),
        })

    return {
        "count": len(documents),
        "documents": documents,
    }


@app.delete("/api/documents/{filename}")
@app.delete("/documents/{filename}")
def delete_document(
    filename: str,
    current_user: AuthenticatedUser = Depends(verify_firebase_token),
):
    # Authorise via Firestore metadata (works even if local file is gone)
    file_meta = get_document_meta(filename)
    if not file_meta or file_meta.get("user_id") != current_user.uid:
        raise HTTPException(status_code=404, detail="File not found or not authorized.")

    # Delete local copy (best-effort — may not be present after a redeploy)
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            logger.warning("[Delete] Could not remove local file %s.", filename, exc_info=True)

    # Delete from Firebase Storage
    if not delete_from_storage(filename):
        logger.warning("[Delete] Could not delete %s from Firebase Storage.", filename)

    # Delete metadata from Firestore
    if not delete_document_meta(filename):
        logger.warning("[Delete] Could not delete Firestore metadata for %s.", filename)

    logger.info("[Delete] %s deleted by %s.", filename, current_user.uid)
    return {"message": f"{filename} deleted successfully"}


@app.post("/api/documents/{filename}/reindex")
def reindex_document(
    filename: str,
    current_user: AuthenticatedUser = Depends(verify_firebase_token),
):
    file_meta = get_document_meta(filename)
    if not file_meta or file_meta.get("user_id") != current_user.uid:
        raise HTTPException(status_code=404, detail="File not found or not authorized.")

    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not on local disk; it will be restored on next startup.")

    documents, chunks = _index_document_file(file_path, filename, file_meta)

    updated_meta = {
        **file_meta,
        "pages": len(documents),
        "chunks": len(chunks),
        "status": "indexed",
        "reindexed_at": datetime.now().isoformat(),
    }
    if not save_document_meta(filename, updated_meta):
        logger.warning("[Reindex] Firestore metadata update failed for %s.", filename)

    return {
        "message": f"{filename} re-indexed successfully",
        "filename": filename,
        "pages": len(documents),
        "chunks": len(chunks),
        "status": "indexed",
    }


@app.get("/api/analytics")
@app.get("/analytics")
def analytics(
    current_user: AuthenticatedUser = Depends(verify_firebase_token),
    selected_range: str = QueryParam(default="7d", alias="range"),
    start_date: str | None = QueryParam(default=None, alias="start_date"),
    end_date: str | None = QueryParam(default=None, alias="end_date"),
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    try:
        start_dt, end_dt = resolve_analytics_period(selected_range, start_date, end_date)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    total_documents = len([
        fname for fname in os.listdir(UPLOAD_DIR)
        if os.path.isfile(os.path.join(UPLOAD_DIR, fname))
    ])
    total_size_bytes = sum(
        os.path.getsize(os.path.join(UPLOAD_DIR, fname))
        for fname in os.listdir(UPLOAD_DIR)
        if os.path.isfile(os.path.join(UPLOAD_DIR, fname))
    )
    vector_store_mb = round(total_size_bytes / (1024 * 1024), 2)

    stats = load_stats()
    all_queries = [
        q for q in stats.get("queries", [])
        if q.get("user", {}).get("uid") == current_user.uid
    ]
    logger.info("Analytics range selected: %s", selected_range)
    logger.info("Analytics records before filtering: %s", len(all_queries))
    filtered_queries = filter_queries_by_timestamp(all_queries, start_dt, end_dt)
    logger.info("Analytics records after filtering: %s", len(filtered_queries))
    stats["queries"] = filtered_queries
    data = summarize_analytics(
        stats=stats,
        total_documents=total_documents,
        vector_store_mb=vector_store_mb,
        start_dt=start_dt,
        end_dt=end_dt,
        selected_range=selected_range,
    )
    log_analytics_summary(selected_range, len(all_queries), len(filtered_queries), data)
    return data

class ChatCreate(BaseModel):
    title: str
    collection_id: str | None = None




@app.post("/api/chat")
def ask_question(payload: Query, current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    query_type = classify_query(payload.question)
    if query_type != "Document Question":
        answer = build_direct_response(query_type)
        result = {
            "answer": answer,
            "confidence": 100,
            "faithfulness": 100,
            "relevance": 100,
            "precision": 100,
            "recall": 100,
            "grounded": True,
            "reason": "",
            "attempts": 1,
            "sources": [],
            "search_source": "Direct Response",
            "query_type": query_type,
        }
    else:
        from backend.graph import run_self_healing_rag

        result = run_self_healing_rag(
            payload.question,
            query_type=query_type,
            collection_id=payload.collection_id,
            user_id=current_user.uid,
        )
        result["query_type"] = query_type

    append_query_log({
        "question": payload.question,
        "user": {
            "uid": current_user.uid,
            "name": current_user.name,
            "email": current_user.email,
        },
        "collection_id": payload.collection_id,
        "chat_id": payload.chat_id,
        "timestamp": datetime.now().isoformat(),
        "confidence": result.get("confidence", 0),
        "faithfulness": result.get("faithfulness", 0),
        "relevance": result.get("relevance", 0),
        "precision": result.get("precision", 0),
        "recall": result.get("recall", 0),
        "grounded": result.get("grounded", False),
        "reason": result.get("reason", ""),
        "attempts": result.get("attempts", 1),
        "sources": result.get("sources", []),
    })

    if payload.chat_id:
        analytics_metadata = {
            "query_type": result.get("query_type", query_type),
            "confidence": result.get("confidence", 0),
            "faithfulness": result.get("faithfulness", 0),
            "relevance": result.get("relevance", 0),
            "precision": result.get("precision", 0),
            "recall": result.get("recall", 0),
            "grounded": result.get("grounded", False),
            "status": "verified" if result.get("grounded") else "insufficient_context",
            "attempts": result.get("attempts", 1),
            "reason": result.get("reason", ""),
            "search_source": result.get("search_source", "Documents"),
        }
        try:
            append_message(
                payload.chat_id,
                current_user.uid,
                {
                    "id": str(uuid.uuid4()),
                    "type": "question",
                    "text": payload.question,
                    "timestamp": datetime.utcnow().isoformat(),
                    "collection_id": payload.collection_id,
                },
                collection_id=payload.collection_id,
                analytics=analytics_metadata,
            )
            append_message(
                payload.chat_id,
                current_user.uid,
                {
                    "id": str(uuid.uuid4()),
                    "type": "answer",
                    "text": result.get("answer", ""),
                    "timestamp": datetime.utcnow().isoformat(),
                    "confidence": result.get("confidence", 0),
                    "faithfulness": result.get("faithfulness", 0),
                    "relevance": result.get("relevance", 0),
                    "precision": result.get("precision", 0),
                    "recall": result.get("recall", 0),
                    "grounded": result.get("grounded", False),
                    "status": "verified" if result.get("grounded") else "insufficient_context",
                    "attempts": result.get("attempts", 1),
                    "reason": result.get("reason", ""),
                    "sources": result.get("sources", []),
                    "searchSource": result.get("search_source", "Documents"),
                    "queryType": result.get("query_type", query_type),
                },
                collection_id=payload.collection_id,
                analytics=analytics_metadata,
            )
        except ChatPersistenceError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    return result

@app.get("/api/chats")
@app.get("/chats")
def list_chats_endpoint(current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    return {"chats": get_user_chats(current_user.uid)}

@app.post("/api/chats")
@app.post("/chats")
def create_chat_endpoint(payload: ChatCreate, current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    try:
        chat = create_chat(
            current_user.uid,
            payload.title,
            collection_id=payload.collection_id,
        )
    except ChatPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return chat

# Search chats by title
@app.get("/api/chats/search")
@app.get("/chats/search")
def search_chats_endpoint(query: str = "", current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    return {"chats": search_chats(current_user.uid, query)}

@app.get("/api/chats/{chat_id}")
@app.get("/chats/{chat_id}")
def get_chat_endpoint(chat_id: str, current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    chat = get_chat(chat_id, current_user.uid)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@app.delete("/api/chats/{chat_id}")
@app.delete("/chats/{chat_id}")
def delete_chat_endpoint(chat_id: str, current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    try:
        deleted = delete_chat(chat_id, current_user.uid)
    except ChatPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not deleted:
        raise HTTPException(status_code=404, detail="Chat not found or not authorized")
    return {"message": "Chat deleted"}

# Update chat (rename or pin)
class ChatUpdate(BaseModel):
    title: str | None = None
    pinned: bool | None = None

@app.put("/api/chats/{chat_id}")
@app.put("/chats/{chat_id}")
def update_chat_endpoint(chat_id: str, payload: ChatUpdate, current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    try:
        updated = update_chat(chat_id, current_user.uid, title=payload.title, pinned=payload.pinned)
    except ChatPersistenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not updated:
        raise HTTPException(status_code=404, detail="Chat not found or not authorized")
    return {"updated": True}

@app.get("/api/collections")
@app.get("/collections")
def get_collections(current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    collections = load_collections()
    # Filter collections for this user or system
    user_collections = [c for c in collections if c.get("user_id") in (current_user.uid, "system")]
    return {"collections": user_collections}


class CollectionCreate(BaseModel):
    name: str

@app.post("/api/collections")
@app.post("/collections")
def create_new_collection(payload: CollectionCreate, current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    collection = create_collection(payload.name, current_user.uid)
    return collection


@app.delete("/api/collections/{collection_id}")
@app.delete("/collections/{collection_id}")
def remove_collection(collection_id: str, current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    collection = find_collection(collection_id)
    if not collection:
        return {"error": "Collection not found"}
    if collection.get("user_id") not in (current_user.uid, "system"):
        return {"error": "Unauthorized"}
    deleted = delete_collection(collection_id)
    return {"message": "Collection deleted"}


@app.get("/api/settings")
@app.get("/settings")
def get_user_settings(current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    return load_settings(current_user.uid)


class SettingsUpdate(BaseModel):
    settings: dict

@app.post("/api/settings")
@app.post("/settings")
def update_user_settings(payload: SettingsUpdate, current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    updated = save_settings(current_user.uid, payload.settings)
    return updated


@app.get("/api/logs/stream")
async def stream_logs():
    async def event_generator():
        steps = [
            "Question Received",
            "Vector Retrieval",
            "Answer Gen",
            "Verification Agent",
            "Critic Agent",
        ]
        while True:
            for step in steps:
                payload = json.dumps({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "step": step,
                    "status": "active" if step == "Verification Agent" else "ok",
                })
                yield f"data: {payload}\n\n"
                await asyncio.sleep(2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Serve static frontend files if built
frontend_dist_path = os.path.join(BASE_DIR, "dist")
if os.path.exists(frontend_dist_path):
    from fastapi.staticfiles import StaticFiles
    logger.info("[Startup] Mounting frontend static files from: %s", frontend_dist_path)
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")
else:
    logger.info("[Startup] Frontend build directory not found at: %s. Running in API-only mode.", frontend_dist_path)
    @app.get("/")
    def root():
        return {"status": "ok", "message": "Self-Healing RAG Running (API-only mode)"}
