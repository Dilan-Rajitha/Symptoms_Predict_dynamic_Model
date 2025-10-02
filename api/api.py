# api/api.py  (or src/api.py if you run with uvicorn src.api:app)
import os, tempfile, requests, joblib, numpy as np, traceback, json
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

app = FastAPI(title="Symptoms Predict API")

# CORS for RN / web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Model path & dynamic download
# ---------------------------
LOCAL_MODEL = Path("models/model.joblib")                       # if present, will use this (good for local dev)
TMP_MODEL   = Path(os.getenv("VERCEL_TMP_DIR") or tempfile.gettempdir()) / "model.joblib"  # Railway/Vercel writable dir
MODEL_PATH  = LOCAL_MODEL if LOCAL_MODEL.exists() else TMP_MODEL

# ⚠️ CHANGE THIS to a PUBLIC direct link (GitHub Releases / S3 / Dropbox direct)
MODEL_URL   = "https://github.com/Dilan-Rajitha/Symptoms_Predict/raw/main/models/model.joblib"
# GitHub 'raw' URL with LFS sometimes fails; Releases asset is more reliable.

PIPELINE = None
MLB = None
META = {}

def ensure_model():
    """Download model if missing, then load into memory (works on Windows & serverless)."""
    global PIPELINE, MLB, META
    if PIPELINE is not None and MLB is not None:
        return
    try:
        # make sure dir exists
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

        # download once per cold-start if file not present
        if not MODEL_PATH.exists():
            print(f"[MODEL] Downloading from: {MODEL_URL}")
            r = requests.get(MODEL_URL, stream=True, timeout=90)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"[MODEL] Saved to: {MODEL_PATH}")

        saved = joblib.load(MODEL_PATH)
        PIPELINE = saved["pipeline"]
        MLB = saved["mlb"]
        META = saved.get("meta", {"source_url": MODEL_URL})
        print(f"[MODEL] Loaded OK from {MODEL_PATH}")
    except Exception as e:
        print("[MODEL LOAD ERROR]", e)
        print(traceback.format_exc())
        raise

# ---------------------------
# Request schema
# ---------------------------
class Request(BaseModel):
    lang: Optional[str] = "en"
    text: str
    age: Optional[int] = None
    sex: Optional[str] = None
    vitals: Optional[Dict[str, Any]] = None

# ---------------------------
# Triage logic (your original simple_triage)
# ---------------------------
def simple_triage(top):
    if not top:
        return {"level": "SELF_CARE", "why": ["No signal detected"]}

    t0 = top[0]  # highest probability condition

    # Life-threatening shortlist
    if t0["id"] in {"ami", "meningitis", "heatstroke", "dka", "stroke", "seizure"} and t0["prob"] > 0.35:
        return {"level": "EMERGENCY", "why": ["Potential life-threatening pattern"]}

    # Same-day urgent list
    if t0["id"] in {"appendicitis", "angina", "dengue_fever", "kidney_stones", "cholera", "typhoid"} and t0["prob"] > 0.35:
        return {"level": "URGENT_TODAY", "why": [f"{t0['name']} suspicion"]}

    # Low-confidence fallback
    if t0["prob"] < 0.25:
        return {"level": "SELF_CARE", "why": ["Low-risk pattern; monitor"]}

    # Default moderate risk
    return {"level": "GP_24_48H", "why": ["Moderate risk pattern"]}

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/")
def health():
    # load lazily to avoid startup failure if URL is down: omit ensure_model() here if you want a super-fast health
    return {"ok": True, "message": "POST /ai/symptom-check", "model_path": str(MODEL_PATH), "meta": META}

@app.post("/ai/symptom-check")
def check(req: Request):
    try:
        ensure_model()
        proba = PIPELINE.predict_proba([req.text])[0]
        idx = np.argsort(proba)[::-1][:3]

        top = []
        for i in idx:
            cid = str(MLB.classes_[i])
            p = float(proba[i])
            top.append({
                "id": cid,
                "name": cid.replace("_", " ").title(),
                "prob": p,
                "prob_pct": round(p * 100, 2)
            })

        tri = simple_triage(top)
        return {
            "top_conditions": top,
            "triage": tri,
            "disclaimer": "Educational aid; not a medical diagnosis."
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
