import asyncio
import json
import logging
import os
import uuid

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(name)s: %(message)s')
import shutil
from datetime import datetime

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Query as QueryParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from backend.analytics import (
    append_query_log,
    build_analytics_report,
    filter_queries_by_timestamp,
    load_stats,
    log_analytics_summary,
    resolve_analytics_period,
    summarize_analytics,
)
from backend.auth import AuthenticatedUser, verify_firebase_token
from backend.chat_history_store import ChatPersistenceError, create_chat, get_user_chats, get_chat, delete_chat, append_message, update_chat, search_chats
from backend.collections_store import load_collections, create_collection, delete_collection, find_collection
from backend.document_store import delete_from_storage, sync_from_storage, upload_to_storage
from backend.persistence import get_firestore_client
from backend.settings_store import load_settings, save_settings
from backend.classification import build_direct_response, classify_query
import time
from backend.performance_logger import timed_stage
from backend.ingestion.chunker import chunk_documents
from backend.ingestion.document_loader import load_document
from backend.ingestion.embeddings import store_documents

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
META_FILE = os.path.join(BASE_DIR, "data", "documents_meta.json")

logger = logging.getLogger(__name__)
FALLBACK_MESSAGE = "I couldn't find enough relevant information in the uploaded documents to answer this question confidently."


def _load_meta():
    """Load document metadata from Firestore; fallback to local JSON file if Firestore unavailable."""
    client = get_firestore_client()
    if client:
        doc_ref = client.collection('metadata').document('documents_meta')
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
    # Fallback to local file
    os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
    if os.path.exists(META_FILE):
        try:
            with open(META_FILE, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception:
            pass
    return {}


def _save_meta(meta):
    os.makedirs(os.path.dirname(META_FILE), exist_ok=True)
    with open(META_FILE, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    client = get_firestore_client()
    if client:
        doc_ref = client.collection('metadata').document('documents_meta')
        doc_ref.set(meta)
        return True
    return False


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def hydrate_persistent_storage():
    # Restore any files that were uploaded to Firebase Storage but not present locally.
    restored_files = sync_from_storage()
    if restored_files:
        logger.info("Restored %s uploaded files from Firebase Storage.", restored_files)

    # Ensure vector store (Chroma) is populated. If empty, rebuild from all persisted documents.
    try:
        from backend.vectorstore.chroma import get_vectorstore
        vectorstore = get_vectorstore()
        # Chroma's .get() returns a dict with a 'documents' key.
        docs = vectorstore.get().get("documents", [])
        if not docs:
            logger.info("Vector store empty; rebuilding from uploaded documents.")
            meta = _load_meta()
            for fname in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, fname)
                if os.path.isfile(file_path):
                    file_meta = meta.get(fname, {})
                    if not file_meta:
                        logger.warning("Skipping %s during vector rebuild because metadata is missing.", fname)
                        continue
                    _index_document_file(file_path, fname, file_meta)
    except Exception as e:
        logger.warning("Failed to verify or rebuild vector store: %s", e, exc_info=True)

    # Sync chat history from Firestore to local cache on startup.
    try:
        from backend.chat_history_store import sync_chats_from_firestore
        sync_chats_from_firestore()
    except Exception:
        logger.warning("Chat history sync on startup failed.", exc_info=True)


class Query(BaseModel):
    question: str
    collection_id: str | None = None
    chat_id: str | None = None


@app.get("/")
def root():
    return {"message": "Self-Healing RAG Running"}


@app.post("/api/documents/upload")
@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_id: str | None = None,
    current_user: AuthenticatedUser = Depends(verify_firebase_token),
):
    allowed_extensions = [".pdf", ".docx", ".pptx", ".txt", ".md", ".csv", ".xlsx", ".jpg", ".jpeg", ".png"]
    extension = os.path.splitext(file.filename)[1].lower()

    if extension not in allowed_extensions:
        return {"error": f"Unsupported file type: {extension}"}

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if not upload_to_storage(file.filename):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=503, detail="Firebase Storage is unavailable. Document was not persisted.")

    documents = load_document(file_path)
    chunks = chunk_documents(documents)

    for chunk in chunks:
        chunk.metadata["user_id"] = current_user.uid
        chunk.metadata["filename"] = file.filename
        if collection_id:
            chunk.metadata["collection_id"] = collection_id
    store_documents(chunks)

    file_size = os.path.getsize(file_path)
    meta = _load_meta()
    
    ocr_used = any(doc.metadata.get("ocr_used", False) for doc in documents)
    
    meta[file.filename] = {
        "pages": len(documents),
        "chunks": len(chunks),
        "collection_id": collection_id,
        "ocr_used": ocr_used,
        "user_id": current_user.uid,
        "size_bytes": file_size,
        "uploaded_at": datetime.now().isoformat(),
        "status": "indexed",
    }
    if not _save_meta(meta):
        delete_from_storage(file.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=503, detail="Firestore is unavailable. Document metadata was not persisted.")

    return {
        "message": "Document uploaded and indexed successfully",
        "filename": file.filename,
        "file_type": extension,
        "pages": len(documents),
        "chunks": len(chunks),
        "size_bytes": file_size,
        "size_kb": round(file_size / 1024, 2),
        "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "indexed"
    }


@app.get("/api/documents")
@app.get("/documents")
def get_documents(current_user: AuthenticatedUser = Depends(verify_firebase_token)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    documents = []
    meta = _load_meta()

    for fname in os.listdir(UPLOAD_DIR):
        file_path = os.path.join(UPLOAD_DIR, fname)
        if not os.path.isfile(file_path):
            continue
            
        file_meta = meta.get(fname, {})
        if file_meta.get("user_id") != current_user.uid:
            continue
            
        file_size = os.path.getsize(file_path)
        documents.append({
            "id": fname,
            "name": fname,
            "filename": fname,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 2),
            "pages": file_meta.get("pages", 0),
            "chunks": file_meta.get("chunks", 0),
            "status": file_meta.get("status", "indexed"),
            "uploaded_at": file_meta.get("uploaded_at") or datetime.fromtimestamp(os.path.getctime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
        })

    return {
        "count": len(documents),
        "documents": documents
    }


@app.delete("/api/documents/{filename}")
@app.delete("/documents/{filename}")
def delete_document(
    filename: str,
    current_user: AuthenticatedUser = Depends(verify_firebase_token),
):
    file_path = os.path.join(UPLOAD_DIR, filename)

    meta = _load_meta()
    if meta.get(filename, {}).get("user_id") != current_user.uid:
        return {"error": "File not found or not authorized"}

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    os.remove(file_path)
    if not delete_from_storage(filename):
        logger.warning("Could not delete %s from Firebase Storage.", filename)
    meta = _load_meta()
    if filename in meta:
        del meta[filename]
        if not _save_meta(meta):
            raise HTTPException(status_code=503, detail="Firestore is unavailable. Document metadata was not updated.")

    return {"message": f"{filename} deleted successfully"}


@app.post("/api/documents/{filename}/reindex")
def reindex_document(
    filename: str,
    current_user: AuthenticatedUser = Depends(verify_firebase_token),
):
    file_path = os.path.join(UPLOAD_DIR, filename)

    meta = _load_meta()
    file_meta = meta.get(filename, {})
    if file_meta.get("user_id") != current_user.uid:
        return {"error": "File not found or not authorized"}

    if not os.path.exists(file_path):
        return {"error": "File not found"}

    documents, chunks = _index_document_file(file_path, filename, file_meta)

    meta = _load_meta()
    meta[filename] = {
        **file_meta,
        "pages": len(documents),
        "chunks": len(chunks),
        "status": "indexed",
        "reindexed_at": datetime.now().isoformat(),
    }
    if not _save_meta(meta):
        raise HTTPException(status_code=503, detail="Firestore is unavailable. Document metadata was not updated.")

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

@app.get("/api/analytics/export")
@app.get("/analytics/export")
def export_analytics(
    format: str = "json",
    selected_range: str = QueryParam(default="7d", alias="range"),
    start_date: str | None = QueryParam(default=None, alias="start_date"),
    end_date: str | None = QueryParam(default=None, alias="end_date"),
    current_user: AuthenticatedUser = Depends(verify_firebase_token),
):
    """Export analytics data as JSON or PDF for download.
    Default format is JSON. Use `format=pdf` for PDF output.
    Includes: total queries, confidence metrics, faithfulness metrics,
    hallucination rate, retry statistics, date range, and charts summary.
    """
    fmt = format.lower().strip()
    if fmt not in ("json", "pdf"):
        logger.warning("[Export] Unsupported format requested: %s by user %s", fmt, current_user.uid)
        raise HTTPException(status_code=400, detail=f"Unsupported format '{fmt}'. Supported formats: json, pdf")

    logger.info("[Export] Request received — format=%s range=%s user=%s", fmt, selected_range, current_user.uid)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    try:
        total_documents = len([
            fname for fname in os.listdir(UPLOAD_DIR)
            if os.path.isfile(os.path.join(UPLOAD_DIR, fname))
        ])
        total_size_bytes = sum(
            os.path.getsize(os.path.join(UPLOAD_DIR, fname))
            for fname in os.listdir(UPLOAD_DIR)
            if os.path.isfile(os.path.join(UPLOAD_DIR, fname))
        )
    except Exception as exc:
        logger.error("[Export] Failed to read upload directory: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read document storage.")

    vector_store_mb = round(total_size_bytes / (1024 * 1024), 2)

    try:
        stats = load_stats()
    except Exception as exc:
        logger.error("[Export] Failed to load analytics stats: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to load analytics data.")

    user_queries = [
        q for q in stats.get("queries", [])
        if q.get("user", {}).get("uid") == current_user.uid
    ]
    logger.info("[Export] Total user queries loaded: %s", len(user_queries))

    try:
        start_dt, end_dt = resolve_analytics_period(selected_range, start_date, end_date)
    except ValueError as exc:
        logger.warning("[Export] Invalid date range: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    filtered_queries = filter_queries_by_timestamp(user_queries, start_dt, end_dt)
    logger.info("[Export] Queries after date filter: %s", len(filtered_queries))

    stats["queries"] = filtered_queries
    try:
        raw_data = summarize_analytics(
            stats=stats,
            total_documents=total_documents,
            vector_store_mb=vector_store_mb,
            start_dt=start_dt,
            end_dt=end_dt,
            selected_range=selected_range,
        )
    except Exception as exc:
        logger.error("[Export] Failed to summarize analytics: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to compute analytics summary.")

    log_analytics_summary(selected_range, len(user_queries), len(filtered_queries), raw_data)

    # Build structured report using build_analytics_report
    report = build_analytics_report(raw_data)
    # Enrich report with additional fields required by spec
    report["documents_indexed"] = total_documents
    report["vector_store_mb"] = vector_store_mb
    report["status"] = raw_data.get("status", "healthy")
    report["generated_at"] = datetime.utcnow().isoformat() + "Z"
    report["charts_summary"] = {
        "confidence_trend": [
            {"date": item.get("date"), "average_confidence": item.get("average_confidence", 0)}
            for item in raw_data.get("history", [])
        ],
        "faithfulness_trend": [
            {"date": item.get("date"), "average_faithfulness": item.get("average_faithfulness", 0)}
            for item in raw_data.get("history", [])
        ],
        "retry_trend": [
            {"date": item.get("date"), "retries": item.get("retries", 0)}
            for item in raw_data.get("history", [])
        ],
        "overview": [
            {"name": "Total Queries", "value": raw_data.get("total_queries", 0)},
            {"name": "Retried Queries", "value": raw_data.get("retry_count", 0)},
            {"name": "Reliable Answers", "value": raw_data.get("reliable_count", 0)},
            {"name": "Hallucinations", "value": raw_data.get("hallucination_count", 0)},
        ],
    }

    # ── JSON export ────────────────────────────────────────────────────────────
    if fmt == "json":
        try:
            import json as _json
            content_str = _json.dumps(report, default=str, indent=2)
        except Exception as exc:
            logger.error("[Export] JSON serialisation failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to serialise report as JSON.")

        logger.info("[Export] JSON export successful for user %s", current_user.uid)
        return StreamingResponse(
            iter([content_str.encode("utf-8")]),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=analytics_report.json"},
        )

    # ── PDF export ─────────────────────────────────────────────────────────────
    if fmt == "pdf":
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.pdfgen import canvas as rl_canvas
        except ImportError as exc:
            logger.error("[Export] ReportLab not installed: %s", exc)
            raise HTTPException(status_code=500, detail="PDF generation library (reportlab) is not installed.")

        try:
            import io as _io
            buffer = _io.BytesIO()
            c = rl_canvas.Canvas(buffer, pagesize=letter)
            page_width, page_height = letter

            def _new_page_if_needed(y_pos, min_y=80):
                if y_pos < min_y:
                    c.showPage()
                    return page_height - 60
                return y_pos

            def _section_header(y_pos, title):
                y_pos = _new_page_if_needed(y_pos, 120)
                c.setFillColor(colors.HexColor("#F5DFD2"))
                c.rect(40, y_pos - 6, page_width - 80, 22, fill=1, stroke=0)
                c.setFillColor(colors.HexColor("#6A2A05"))
                c.setFont("Helvetica-Bold", 11)
                c.drawString(50, y_pos + 4, title)
                c.setFillColor(colors.black)
                c.setFont("Helvetica", 10)
                return y_pos - 26

            def _kv_row(y_pos, label, value, row_bg=False):
                y_pos = _new_page_if_needed(y_pos)
                if row_bg:
                    c.setFillColor(colors.HexColor("#FDF5F2"))
                    c.rect(40, y_pos - 4, page_width - 80, 16, fill=1, stroke=0)
                    c.setFillColor(colors.black)
                c.setFont("Helvetica", 10)
                c.drawString(55, y_pos, str(label))
                c.drawRightString(page_width - 50, y_pos, str(value))
                return y_pos - 18

            # ── Cover header ───────────────────────────────────────────────────
            c.setFillColor(colors.HexColor("#C84B2F"))
            c.rect(0, page_height - 90, page_width, 90, fill=1, stroke=0)
            c.setFillColor(colors.white)
            c.setFont("Helvetica-Bold", 22)
            c.drawString(50, page_height - 48, "Self-Healing RAG Platform")
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, page_height - 68, "Analytics Report")
            c.setFont("Helvetica", 10)
            c.drawString(50, page_height - 84, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

            y = page_height - 110

            # ── Date Range ────────────────────────────────────────────────────
            y = _section_header(y, "Date Range")
            date_range = report.get("date_range", {})
            row_bg = False
            for lbl, val in [
                ("Range", date_range.get("range", selected_range)),
                ("Start Date", (date_range.get("start_date") or "N/A")[:10]),
                ("End Date", (date_range.get("end_date") or "N/A")[:10]),
            ]:
                y = _kv_row(y, lbl, val, row_bg)
                row_bg = not row_bg
            y -= 10

            # ── Summary ───────────────────────────────────────────────────────
            y = _section_header(y, "Summary")
            row_bg = False
            for lbl, val in [
                ("Total Queries", report.get("total_queries", 0)),
                ("Documents Indexed", total_documents),
                ("Vector Store Size (MB)", vector_store_mb),
                ("System Status", report.get("status", "healthy").title()),
            ]:
                y = _kv_row(y, lbl, val, row_bg)
                row_bg = not row_bg
            y -= 10

            # ── Confidence Metrics ────────────────────────────────────────────
            y = _section_header(y, "Confidence Metrics")
            conf = report.get("confidence_metrics", {})
            row_bg = False
            y = _kv_row(y, "Average Confidence", f"{conf.get('average_confidence', 0):.1f}%", row_bg)
            y -= 10

            # ── Faithfulness Metrics ──────────────────────────────────────────
            y = _section_header(y, "Faithfulness Metrics")
            faith = report.get("faithfulness_metrics", {})
            row_bg = False
            for lbl, key in [
                ("Average Faithfulness", "average_faithfulness"),
                ("Average Relevance", "average_relevance"),
                ("Average Precision", "average_precision"),
                ("Average Recall", "average_recall"),
            ]:
                y = _kv_row(y, lbl, f"{faith.get(key, 0):.1f}%", row_bg)
                row_bg = not row_bg
            y -= 10

            # ── Hallucination Rate ────────────────────────────────────────────
            y = _section_header(y, "Hallucination Rate")
            hall = report.get("hallucination_rate", {})
            row_bg = False
            for lbl, val in [
                ("Hallucination Rate", f"{hall.get('rate', 0):.1f}%"),
                ("Hallucinated Queries", hall.get("count", 0)),
                ("Reliable Answers", hall.get("reliable_count", 0)),
            ]:
                y = _kv_row(y, lbl, val, row_bg)
                row_bg = not row_bg
            y -= 10

            # ── Retry Statistics ──────────────────────────────────────────────
            y = _section_header(y, "Retry Statistics")
            retry = report.get("retry_statistics", {})
            row_bg = False
            for lbl, val in [
                ("Retry Rate", f"{retry.get('retry_rate', 0):.1f}%"),
                ("Total Retried Queries", retry.get("retry_count", 0)),
            ]:
                y = _kv_row(y, lbl, val, row_bg)
                row_bg = not row_bg
            y -= 10

            # ── Charts Summary ────────────────────────────────────────────────
            charts = report.get("charts_summary", {})
            overview = charts.get("overview", [])
            if overview:
                y = _section_header(y, "Charts Summary — Overview")
                row_bg = False
                for item in overview:
                    y = _kv_row(y, item.get("name", ""), item.get("value", 0), row_bg)
                    row_bg = not row_bg
                y -= 10

            # ── Most Queried Documents ────────────────────────────────────────
            top_docs = report.get("most_queried_documents", [])
            if top_docs:
                y = _section_header(y, "Most Queried Documents")
                row_bg = False
                for item in top_docs[:10]:
                    y = _kv_row(y, item.get("document", ""), f"{item.get('queries', 0)} queries", row_bg)
                    row_bg = not row_bg
                y -= 10

            # ── Footer ────────────────────────────────────────────────────────
            c.setFillColor(colors.HexColor("#6A4034"))
            c.setFont("Helvetica", 8)
            c.drawCentredString(page_width / 2, 30, "Self-Healing RAG Platform  |  Confidential Analytics Report")

            c.save()
            buffer.seek(0)
            pdf_bytes = buffer.getvalue()
        except Exception as exc:
            logger.error("[Export] PDF generation failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"PDF generation failed: {exc}")

        logger.info("[Export] PDF export successful for user %s (%d bytes)", current_user.uid, len(pdf_bytes))
        return StreamingResponse(
            iter([pdf_bytes]),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=analytics_report.pdf"},
        )

    # Unreachable — guard already checked above
    raise HTTPException(status_code=400, detail="Invalid format. Supported: json, pdf")


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
