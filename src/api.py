from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib, requests, os
from pathlib import Path
import numpy as np
import traceback

# Setup
app = FastAPI(title="Symptoms Predict API (Vercel)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow your RN app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("models/model.joblib")
MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)

# Dynamic download
if not MODEL_PATH.exists():
    url = "https://github.com/Dilan-Rajitha/Symptoms_Predict/raw/main/models/model.joblib"
    print("Downloading model...")
    r = requests.get(url, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Model downloaded successfully")

# Load model
try:
    saved = joblib.load(MODEL_PATH)
    PIPELINE = saved["pipeline"]
    MLB = saved["mlb"]
except Exception as e:
    print("Error loading model:", e)
    PIPELINE = None
    MLB = None

# Request model
class Request(BaseModel):
    lang: str = "en"
    text: str
    age: int = 0
    sex: str = "string"
    vitals: dict = {}

@app.get("/")
def health():
    return {"ok": True, "message": "POST /ai/symptom-check"}

@app.post("/ai/symptom-check")
def check(req: Request):
    try:
        if PIPELINE is None or MLB is None:
            return {"error": "Model not loaded."}
        proba = PIPELINE.predict_proba([req.text])[0]
        idx = np.argsort(proba)[::-1][:3]
        top = [{"id": str(MLB.classes_[i]), "prob": float(proba[i]), "prob_pct": round(float(proba[i])*100,2)} for i in idx]
        triage = {"level": "GP_24_48H"}  # optional: add your triage rules
        return {"top_conditions": top, "triage": triage, "disclaimer": "Educational aid; not medical advice."}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
