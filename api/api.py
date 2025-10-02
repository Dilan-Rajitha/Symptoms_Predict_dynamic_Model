# api/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os, tempfile, requests, joblib, numpy as np, traceback

app = FastAPI(title="Symptoms Predict API (Vercel/Local)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Paths (portable) ----
# 1) Prefer local repo model if present (good for local dev)
LOCAL_MODEL = Path("models/model.joblib")
# 2) Otherwise use OS temp dir (Windows: e.g. C:\Users\<you>\AppData\Local\Temp, Vercel: /tmp)
TMP_DIR = Path(os.getenv("VERCEL_TMP_DIR") or tempfile.gettempdir())
TMP_DIR.mkdir(parents=True, exist_ok=True)
TMP_MODEL = TMP_DIR / "model.joblib"

# Decide where to read/write the model
MODEL_PATH = LOCAL_MODEL if LOCAL_MODEL.exists() else TMP_MODEL

# Public URL to download the model (GitHub Release / S3 / Dropbox direct link)
MODEL_URL = "https://github.com/Dilan-Rajitha/Symptoms_Predict/raw/main/models/model.joblib"

PIPELINE = None
MLB = None
META = {}

def ensure_model():
    """Download model if missing, then load it."""
    global PIPELINE, MLB, META
    if PIPELINE is not None and MLB is not None:
        return
    try:
        if not MODEL_PATH.exists():
            # Download to the chosen folder (LOCAL_MODEL.parent or temp dir)
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            print(f"[MODEL] Downloading to: {MODEL_PATH}")
            r = requests.get(MODEL_URL, stream=True, timeout=90)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("[MODEL] Download complete")

        saved = joblib.load(MODEL_PATH)
        PIPELINE = saved["pipeline"]
        MLB = saved["mlb"]
        META = saved.get("meta", {})
        print(f"[MODEL] Loaded from: {MODEL_PATH}")
    except Exception as e:
        print("[MODEL LOAD ERROR]", e)
        print(traceback.format_exc())
        raise

class Req(BaseModel):
    lang: str = "en"
    text: str
    age: int | None = None
    sex: str | None = None
    vitals: dict | None = None

@app.get("/")
def health():
    return {"ok": True, "message": "POST /ai/symptom-check", "model_path": str(MODEL_PATH), "meta": META}

def triage_from(top):
    if not top: return {"level": "SELF_CARE", "why": ["No signal"]}
    t0 = top[0]
    if t0["id"] in {"ami","meningitis","heatstroke","dka","stroke","seizure"} and t0["prob"] > 0.35:
        return {"level": "EMERGENCY", "why": ["Potential life-threatening pattern"]}
    if t0["id"] in {"appendicitis","angina","dengue_fever","kidney_stones","cholera","typhoid"} and t0["prob"] > 0.35:
        return {"level": "URGENT_TODAY", "why": [f"{t0['name']} suspicion"]}
    if t0["prob"] < 0.25:
        return {"level": "SELF_CARE", "why": ["Low-risk pattern; monitor"]}
    return {"level": "GP_24_48H", "why": ["Moderate risk pattern"]}

@app.post("/ai/symptom-check")
def check(req: Req):
    try:
        ensure_model()
        proba = PIPELINE.predict_proba([req.text])[0]
        idx = np.argsort(proba)[::-1][:3]
        top = []
        for i in idx:
            cid = str(MLB.classes_[i]); p = float(proba[i])
            top.append({"id": cid, "name": cid.replace("_"," ").title(), "prob": p, "prob_pct": round(p*100,2)})
        return {"top_conditions": top, "triage": triage_from(top), "disclaimer": "Educational aid; not a medical diagnosis."}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
