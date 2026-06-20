import os
import logging
from firebase_admin import initialize_app, credentials, storage
from pathlib import Path

logger = logging.getLogger(__name__)

# Initialise Firebase app (idempotent)
if not len(firebase_admin._apps):
    cred_path = os.getenv("FIREBASE_SERVICE_ACCOUNT")
    if not cred_path:
        raise RuntimeError("FIREBASE_SERVICE_ACCOUNT env var not set")
    cred = credentials.Certificate(cred_path)
    initialize_app(cred, {"storageBucket": os.getenv("FIREBASE_BUCKET")})

# Base upload directory (local cache) and bucket reference
UPLOAD_DIR = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads"))
BUCKET = storage.bucket()

def save_file_locally(file_path: str, filename: str) -> None:
    """Ensure the local upload directory exists and move the file there."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, filename)
    # If the file is already at the destination, skip copy
    if os.path.abspath(file_path) != os.path.abspath(dest):
        os.replace(file_path, dest)
    logger.info(f"Saved file locally: {dest}")

def upload_to_firebase(filename: str) -> None:
    """Upload a file from the local upload directory to Firebase Storage under `uploads/`."""
    local_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file {local_path} not found for Firebase upload")
    blob = BUCKET.blob(f"uploads/{filename}")
    blob.upload_from_filename(local_path)
    logger.info(f"Uploaded {filename} to Firebase Storage bucket {BUCKET.name}")

def download_all_from_firebase() -> None:
    """Download all objects from the `uploads/` folder in Firebase Storage into the local upload directory.
    Existing files are left untouched to avoid unnecessary overwrites.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    blobs = BUCKET.list_blobs(prefix="uploads/")
    for blob in blobs:
        # Strip the prefix to get the filename
        filename = os.path.basename(blob.name)
        if not filename:
            continue
        local_path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(local_path):
            logger.debug(f"File {filename} already exists locally; skipping download")
            continue
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded {filename} from Firebase Storage to {local_path}")
