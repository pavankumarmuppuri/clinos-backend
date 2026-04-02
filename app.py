from dotenv import load_dotenv
load_dotenv()

# ── Firebase Init (must happen before any route imports) ─────────────────────
import firebase_config
from firebase_admin import firestore
firebase_db = firestore.client()

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
import os

app = FastAPI(
    title="ClinOS TrialMatch API",
    description="AI-powered oncology clinical trial matching platform.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Exception Handlers ────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    slugs = {400:"bad_request",401:"unauthorized",403:"forbidden",
             404:"not_found",409:"conflict",422:"validation_error",500:"internal_server_error"}
    return JSONResponse(status_code=exc.status_code, content={
        "status": exc.status_code,
        "error":  slugs.get(exc.status_code, "error"),
        "message": exc.detail,
    })


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    fields = [{"field":" -> ".join(str(l) for l in e["loc"] if l != "body") or "request",
               "message":e["msg"],"type":e["type"]} for e in exc.errors()]
    return JSONResponse(status_code=422, content={
        "status":422,"error":"validation_error",
        "message":"One or more fields failed validation.","fields":fields,
    })


# ── Routers ───────────────────────────────────────────────────────────────────
from routes.auth_routes    import router as auth_router
from routes.patient_routes import router as patient_router
from routes.trial_routes   import router as trial_router
from routes.match_routes   import router as match_router

app.include_router(auth_router)
app.include_router(patient_router)
app.include_router(trial_router)
app.include_router(match_router)


# ── Static Files ──────────────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.get("/", tags=["Frontend"])
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"),
                        headers={"Cache-Control": "no-store, no-cache, must-revalidate"})

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Health & Debug ────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {"status": 200, "message": "ClinOS TrialMatch API is running"}

@app.get("/debug-db", tags=["Debug"])
def debug_db():
    try:
        cols = [c.id for c in firebase_db.collections()]
        return {"using": "firebase", "collections": cols}
    except Exception as e:
        return {"using": "firebase", "error": str(e)}