from fastapi import APIRouter, Depends, status
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

import firebase_config
from firebase_admin import firestore
from auth import get_current_doctor
from utils import route_error, success

firebase_db = firestore.client()
router      = APIRouter(prefix="/trials", tags=["Trials"])


# ── Schemas ───────────────────────────────────────────────────────────────────
class EligibilityCriteria(BaseModel):
    inclusion: List[str] = []
    exclusion: List[str] = []

class TrialCreate(BaseModel):
    nct_id: str
    short_title: str
    full_title: Optional[str] = None
    phase: Optional[str] = None
    status: Optional[str] = None
    target_enrollment: Optional[int] = None
    sponsor: Optional[str] = None
    description: Optional[str] = None
    conditions: List[str] = []
    interventions: List[str] = []
    eligibility: EligibilityCriteria = EligibilityCriteria()


def _trial_dict(data: dict) -> dict:
    return {
        "id":                data.get("id"),
        "nct_id":            data.get("nct_id"),
        "short_title":       data.get("short_title"),
        "full_title":        data.get("full_title"),
        "phase":             data.get("phase"),
        "status":            data.get("status"),
        "target_enrollment": data.get("target_enrollment"),
        "sponsor":           data.get("sponsor"),
        "description":       data.get("description"),
        "created_by":        data.get("created_by"),
        "created_at":        data.get("created_at"),
        "conditions":        data.get("conditions", []),
        "interventions":     data.get("interventions", []),
        "eligibility": {
            "inclusion": data.get("eligibility_inclusion", []),
            "exclusion": data.get("eligibility_exclusion", []),
        },
    }


# ── CREATE ────────────────────────────────────────────────────────────────────
@router.post("", status_code=201)
def create_trial(payload: TrialCreate, current_doctor: dict = Depends(get_current_doctor)):
    if current_doctor.get("role") == "patient":
        raise route_error(status.HTTP_403_FORBIDDEN, "forbidden",
                          "Patients cannot create trials.")

    trials_ref = firebase_db.collection("trials")
    existing   = trials_ref.where("nct_id", "==", payload.nct_id).get()
    if existing:
        raise route_error(status.HTTP_409_CONFLICT, "nct_id_already_exists",
                          f"A trial with NCT ID '{payload.nct_id}' already exists.")

    trial_data = {
        "nct_id":                payload.nct_id,
        "short_title":           payload.short_title,
        "full_title":            payload.full_title,
        "phase":                 payload.phase,
        "status":                payload.status,
        "target_enrollment":     payload.target_enrollment,
        "sponsor":               payload.sponsor,
        "description":           payload.description,
        "conditions":            payload.conditions,
        "interventions":         payload.interventions,
        "eligibility_inclusion": payload.eligibility.inclusion,
        "eligibility_exclusion": payload.eligibility.exclusion,
        "created_by":            current_doctor["id"],
        "created_at":            datetime.utcnow().isoformat(),
    }

    doc_ref = trials_ref.document()
    trial_data["id"] = doc_ref.id
    doc_ref.set(trial_data)

    return success(data=_trial_dict(trial_data),
                   message=f"Trial '{payload.short_title}' created successfully.", http_status=201)


# ── LIST ──────────────────────────────────────────────────────────────────────
@router.get("", status_code=200)
def list_trials(current_doctor: dict = Depends(get_current_doctor)):
    docs   = firebase_db.collection("trials").where("created_by", "==", current_doctor["id"]).stream()
    trials = [_trial_dict(d.to_dict()) for d in docs]
    return success(data=trials, message=f"{len(trials)} trial(s) found.")


# ── GET SINGLE ────────────────────────────────────────────────────────────────
@router.get("/{trial_id}", status_code=200)
def get_trial(trial_id: str, current_doctor: dict = Depends(get_current_doctor)):
    doc = firebase_db.collection("trials").document(trial_id).get()
    if not doc.exists or doc.to_dict().get("created_by") != current_doctor["id"]:
        raise route_error(status.HTTP_404_NOT_FOUND, "trial_not_found",
                          f"No trial found with ID {trial_id}.")
    return success(data=_trial_dict(doc.to_dict()), message="Trial retrieved successfully.")


# ── DELETE ────────────────────────────────────────────────────────────────────
@router.delete("/{trial_id}", status_code=200)
def delete_trial(trial_id: str, current_doctor: dict = Depends(get_current_doctor)):
    if current_doctor.get("role") == "patient":
        raise route_error(status.HTTP_403_FORBIDDEN, "forbidden",
                          "Patients cannot delete trials.")

    doc_ref = firebase_db.collection("trials").document(trial_id)
    doc     = doc_ref.get()
    if not doc.exists or doc.to_dict().get("created_by") != current_doctor["id"]:
        raise route_error(status.HTTP_404_NOT_FOUND, "trial_not_found",
                          f"No trial found with ID {trial_id}.")

    name = doc.to_dict().get("short_title", trial_id)
    doc_ref.delete()
    return success(data={"id": trial_id}, message=f"Trial '{name}' deleted successfully.")