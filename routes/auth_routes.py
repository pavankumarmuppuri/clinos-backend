from fastapi import APIRouter, status
from pydantic import BaseModel, EmailStr
from datetime import datetime
from firebase_admin import firestore

# ✅ Import Firebase config (initializes ONLY once)
import firebase_config

from auth import hash_password, verify_password, create_access_token
from utils import route_error, success

# ✅ Firestore client
firebase_db = firestore.client()

router = APIRouter(prefix="/auth", tags=["Authentication"])

# ── Schemas ────────────────────────────────────────────────────────────────
class DoctorSignup(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: str = "doctor"


class DoctorLogin(BaseModel):
    email: EmailStr
    password: str


# ── Signup ─────────────────────────────────────────────────────────────────
@router.post("/signup", status_code=201)
def signup(payload: DoctorSignup):
    """Register a new doctor or patient account and return a JWT token."""

    users_ref = firebase_db.collection("users")

    # Check if email exists
    existing = users_ref.where("email", "==", payload.email).get()
    if existing:
        raise route_error(
            status.HTTP_409_CONFLICT,
            "email_already_exists",
            f"An account with email '{payload.email}' already exists."
        )

    role = payload.role if payload.role in ("doctor", "patient") else "doctor"

    user_data = {
        "name": payload.name,
        "email": payload.email,
        "password_hash": hash_password(payload.password),
        "role": role,
        "created_at": datetime.utcnow().isoformat(),
    }

    # Save user
    doc_ref = users_ref.document()
    doc_ref.set(user_data)

    token = create_access_token(data={"sub": doc_ref.id})

    return success(
        data={
            "access_token": token,
            "token_type": "bearer",
            "doctor": {
                "id": doc_ref.id,
                "name": payload.name,
                "email": payload.email,
                "role": role,
                "created_at": user_data["created_at"],
            },
        },
        message="Account created successfully.",
        http_status=201,
    )


# ── Login ──────────────────────────────────────────────────────────────────
@router.post("/login", status_code=200)
def login(payload: DoctorLogin):
    """Authenticate a doctor or patient and return a JWT token."""

    users_ref = firebase_db.collection("users")

    docs = users_ref.where("email", "==", payload.email).get()
    if not docs:
        raise route_error(
            status.HTTP_401_UNAUTHORIZED,
            "invalid_credentials",
            "The email or password you entered is incorrect."
        )

    doc = docs[0]
    user = doc.to_dict()

    if not verify_password(payload.password, user.get("password_hash", "")):
        raise route_error(
            status.HTTP_401_UNAUTHORIZED,
            "invalid_credentials",
            "The email or password you entered is incorrect."
        )

    token = create_access_token(data={"sub": doc.id})

    return success(
        data={
            "access_token": token,
            "token_type": "bearer",
            "doctor": {
                "id": doc.id,
                "name": user.get("name"),
                "email": user.get("email"),
                "role": user.get("role", "doctor"),
                "created_at": user.get("created_at"),
            },
        },
        message="Login successful.",
        http_status=200,
    )