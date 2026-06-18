import os
import json
import threading
from dataclasses import dataclass

import firebase_admin
from fastapi import Depends, Header, HTTPException, status
from firebase_admin import auth, credentials


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
    with _app_lock:
        try:
            return firebase_admin.get_app()
        except ValueError:

            service_account_json = os.getenv(
                "FIREBASE_SERVICE_ACCOUNT_JSON"
            )

            if service_account_json:
                service_account = json.loads(
                    service_account_json
                )

                cred = credentials.Certificate(
                    service_account
                )

                return firebase_admin.initialize_app(
                    cred
                )

            return firebase_admin.initialize_app()


def verify_firebase_token(
    authorization: str | None = Header(default=None)
) -> AuthenticatedUser:

    token = _extract_bearer_token(authorization)

    _get_firebase_app()

    try:
        decoded = auth.verify_id_token(token)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired Firebase token",
        ) from exc

    return AuthenticatedUser(
        uid=str(decoded.get("uid", "")),
        name=str(decoded.get("name", "")),
        email=str(decoded.get("email", "")),
        profile_picture=str(decoded.get("picture", "")),
    )


CurrentUser = Depends(verify_firebase_token)