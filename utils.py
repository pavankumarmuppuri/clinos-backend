# utils.py
from fastapi import HTTPException
from fastapi.responses import JSONResponse

# ─── Standardized error response for routes ──────────────────────────────
def route_error(status_code: int, error: str, message: str) -> HTTPException:
    """
    Returns a structured HTTPException for consistent API errors.
    """
    return HTTPException(
        status_code=status_code,
        detail={
            "status": status_code,
            "error": error,
            "message": message,
        },
    )


# ─── Standardized success response for routes ───────────────────────────
def success(data=None, message="Success", http_status=200):
    """
    Returns a standardized JSON response for successful operations.
    """
    return JSONResponse(
        status_code=http_status,
        content={
            "status": http_status,
            "error": None,
            "message": message,
            "data": data,
        },
    )


# ─── Optional helper to validate Firebase UID ────────────────────────────
def validate_firebase_uid(uid: str):
    """
    Simple helper to check if a Firebase UID is valid.
    (You can expand this with actual Firebase Admin checks)
    """
    if not uid or len(uid) < 10:
        raise route_error(400, "invalid_uid", "Invalid Firebase UID provided.")
    return True