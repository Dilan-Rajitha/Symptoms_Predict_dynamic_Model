import os
from pathlib import Path
import joblib
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any

# ---------- Setup ----------
MODEL_PATH = Path("models/model.joblib")
MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)

# ---------- Dynamic download ----------
MODEL_URL = "https://github.com/Dilan-Rajitha/Symptoms_Predict/raw/main/models/model.joblib"

if not MODEL_PATH.exists():
    print("Downloading model dynamically...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model download complete!")

# ---------- Load model ----------
saved = joblib.load(MODEL_PATH)
PIPELINE = saved["pipeline"]
MLB = saved["mlb"]

# ---------- FastAPI ----------
app = FastAPI(title="Symptoms Predict API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Request(BaseModel):
    lang: Optional[str] = "en"
    text: str
    age: Optional[int] = None
    sex: Optional[str] = None
    vitals: Optional[Dict[str, Any]] = None

def simple_triage(top):
    if not top:
        return {"level": "SELF_CARE", "why": ["No signal detected"]}
    t0 = top[0]
    if t0["id"] in {"ami","meningitis","heatstroke","dka","stroke","seizure"} and t0["prob"] > 0.35:
        return {"level": "EMERGENCY", "why": ["Potential life-threatening pattern"]}
    if t0["id"] in {"appendicitis","angina","dengue_fever","kidney_stones","cholera","typhoid"} and t0["prob"] > 0.35:
        return {"level": "URGENT_TODAY", "why": [f"{t0['name']} suspicion"]}
    if t0["prob"] < 0.25:
        return {"level": "SELF_CARE", "why": ["Low-risk pattern; monitor"]}
    return {"level": "GP_24_48H", "why": ["Moderate risk pattern"]}

@app.get("/")
def health():
    return {"ok": True, "message": "POST /ai/symptom-check"}

@app.post("/ai/symptom-check")
def check(req: Request):
    try:
        proba = PIPELINE.predict_proba([req.text])[0]
        idx = proba.argsort()[::-1][:3]
        top = []
        for i in idx:
            cid = str(MLB.classes_[i])
            p = float(proba[i])
            top.append({"id": cid, "name": cid.replace("_", " ").title(), "prob": p, "prob_pct": round(p*100,2)})
        triage = simple_triage(top)
        return {"top_conditions": top, "triage": triage, "disclaimer": "Educational aid; not a medical diagnosis."}
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
