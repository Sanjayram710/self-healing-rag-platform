import json
import logging
import os
import threading
import time

import firebase_admin
from firebase_admin import credentials, firestore, storage

logger = logging.getLogger(__name__)

_firebase_lock = threading.Lock()
_firestore_client = None
_storage_bucket = None


def _service_account_payload():
    payload = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if payload:
        payload_str = payload.strip()
        if payload_str.startswith("{"):
            try:
                return json.loads(payload_str)
            except Exception as exc:
                logger.error("Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON as inline JSON: %s", exc)
        else:
            # Treat as file path
            if os.path.exists(payload_str):
                try:
                    with open(payload_str, "r", encoding="utf-8") as file:
                        return json.load(file)
                except Exception as exc:
                    logger.error("Failed to read FIREBASE_SERVICE_ACCOUNT_JSON from file path %s: %s", payload_str, exc)
            else:
                logger.warning("FIREBASE_SERVICE_ACCOUNT_JSON file path does not exist: %s", payload_str)

    path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as exc:
            logger.error("Failed to read FIREBASE_SERVICE_ACCOUNT file %s: %s", path, exc)

    return None


def get_firebase_app():
    with _firebase_lock:
        try:
            return firebase_admin.get_app()
        except ValueError:
            payload = _service_account_payload()
            bucket_name = os.getenv("FIREBASE_BUCKET") or os.getenv("VITE_FIREBASE_STORAGE_BUCKET")
            
            # Make sure bucket name is clean (e.g. remove gs:// prefix if present)
            if bucket_name:
                bucket_name = bucket_name.strip()
                if bucket_name.startswith("gs://"):
                    bucket_name = bucket_name[5:]
                if bucket_name.endswith("/"):
                    bucket_name = bucket_name[:-1]

            options = {"storageBucket": bucket_name} if bucket_name else None
            
            if payload:
                logger.info("[Firebase] Initializing app with service account credentials and bucket: %s", bucket_name)
                cred = credentials.Certificate(payload)
                return firebase_admin.initialize_app(cred, options)

            logger.info("[Firebase] Initializing app with default credentials and bucket: %s", bucket_name)
            return firebase_admin.initialize_app(options=options)


def get_firestore_client():
    global _firestore_client
    try:
        if _firestore_client is not None:
            return _firestore_client
        start = time.perf_counter()
        app = get_firebase_app()
        _firestore_client = firestore.client(app=app)
        logger.info("[Firebase] Firestore client ready in %.2fs", time.perf_counter() - start)
        return _firestore_client
    except Exception:
        logger.warning("Firestore client unavailable; using local persistence only.", exc_info=True)
        return None


def get_storage_bucket():
    global _storage_bucket
    try:
        if _storage_bucket is not None:
            return _storage_bucket
        start = time.perf_counter()
        app = get_firebase_app()
        _storage_bucket = storage.bucket(app=app)
        logger.info("[Firebase] Storage bucket ready in %.2fs", time.perf_counter() - start)
        return _storage_bucket
    except Exception:
        logger.warning("Firebase Storage unavailable; using local uploads only.", exc_info=True)
        return None
