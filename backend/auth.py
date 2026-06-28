import json
import logging
import os
import threading
import time
from dataclasses import dataclass

import firebase_admin
from fastapi import Depends, Header, HTTPException, status
from firebase_admin import auth, credentials

logger = logging.getLogger("auth")
logger.setLevel(logging.DEBUG)


@dataclass
class AuthenticatedUser:
    uid: str
    name: str
    email: str
    profile_picture: str


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
        )

    return token


_app_lock = threading.Lock()


def _get_firebase_app():
    """Initialise the Firebase Admin SDK exactly once and return the app."""
    with _app_lock:
        try:
            app = firebase_admin.get_app()
            logger.debug("[Firebase] SDK already initialised - returning existing app.")
            return app
        except ValueError:
            pass

        start = time.perf_counter()
        logger.info("[Firebase] Initialising Firebase Admin SDK...")

        cred = None
        init_path = None
        project_id = "self-healing-rag-a57d9"

        env_val = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
        if env_val:
            if env_val.startswith("{"):
                try:
                    cred = credentials.Certificate(json.loads(env_val))
                    init_path = "env var (inline JSON)"
                    logger.info("[Firebase] Credentials loaded from env var (JSON string).")
                except Exception as exc:
                    logger.error("[Firebase] Failed to parse env var as JSON: %s", exc)
            else:
                if os.path.isfile(env_val):
                    try:
                        cred = credentials.Certificate(env_val)
                        init_path = f"env var (file: {env_val})"
                        logger.info("[Firebase] Credentials loaded from env file path: %s", env_val)
                    except Exception as exc:
                        logger.error("[Firebase] Failed to load from env file path %s: %s", env_val, exc)
                else:
                    logger.warning("[Firebase] Env var file path does not exist: %s", env_val)

        if cred is None:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            fallback = os.path.join(backend_dir, "firebase-admin.json")
            if os.path.isfile(fallback):
                try:
                    cred = credentials.Certificate(fallback)
                    init_path = f"fallback ({fallback})"
                    logger.info("[Firebase] Credentials loaded from fallback: %s", fallback)
                except Exception as exc:
                    logger.error("[Firebase] Fallback file failed: %s", exc)

        if cred is None:
            backend_dir = os.path.dirname(os.path.abspath(__file__))
            try:
                for fname in os.listdir(backend_dir):
                    if fname.endswith(".json") and "firebase-adminsdk" in fname:
                        path = os.path.join(backend_dir, fname)
                        try:
                            cred = credentials.Certificate(path)
                            init_path = f"auto-detected ({fname})"
                            logger.info("[Firebase] Credentials loaded from auto-detected: %s", path)
                            break
                        except Exception as exc:
                            logger.error("[Firebase] Auto-detect %s failed: %s", path, exc)
            except Exception as exc:
                logger.error("[Firebase] Error scanning backend dir: %s", exc)

        options = {"projectId": project_id}

        try:
            if cred:
                app = firebase_admin.initialize_app(cred, options=options)
                logger.info("[Firebase] SDK initialised via %s (project=%s).", init_path, project_id)
            else:
                logger.warning("[Firebase] No credentials found - trying default credentials.")
                app = firebase_admin.initialize_app(options=options)
                logger.info("[Firebase] SDK initialised with default credentials (project=%s).", project_id)
            logger.info("[Firebase] Admin SDK init completed in %.2fs", time.perf_counter() - start)
            return app
        except Exception as exc:
            logger.critical("[Firebase] Initialisation FAILED: %s", exc)
            raise


def verify_firebase_token(
    authorization: str | None = Header(default=None),
) -> AuthenticatedUser:
    request_start = time.perf_counter()

    if authorization:
        logger.debug("[Auth] Authorization header present (len=%d).", len(authorization))
    else:
        logger.warning("[Auth] No Authorization header.")

    token = _extract_bearer_token(authorization)

    try:
        auth_start = time.perf_counter()
        _get_firebase_app()
        logger.info("[Auth] Firebase app ready in %.2fs", time.perf_counter() - auth_start)
    except Exception as exc:
        logger.error("[Auth] Firebase app init failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable",
        )

    try:
        verify_start = time.perf_counter()
        decoded = auth.verify_id_token(token)
        logger.info(
            "[Auth] Token verified in %.2fs - uid=%s email=%s",
            time.perf_counter() - verify_start,
            decoded.get("uid"),
            decoded.get("email"),
        )
    except Exception as exc:
        logger.error("[Auth] Token verification failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Firebase token",
        )

    logger.info("[Auth] Total auth dependency time %.2fs", time.perf_counter() - request_start)
    return AuthenticatedUser(
        uid=str(decoded.get("uid", "")),
        name=str(decoded.get("name", "")),
        email=str(decoded.get("email", "")),
        profile_picture=str(decoded.get("picture", "")),
    )


CurrentUser = Depends(verify_firebase_token)
