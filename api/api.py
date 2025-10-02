import os, tempfile, requests, joblib, numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

app = FastAPI(title="Symptoms Predict API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

LOCAL_MODEL = Path("models/model.joblib")
TMP_MODEL   = Path(tempfile.gettempdir()) / "model.joblib"
MODEL_PATH  = LOCAL_MODEL if LOCAL_MODEL.exists() else TMP_MODEL
MODEL_URL   = "https://github.com/Dilan-Rajitha/Symptoms_Predict/raw/main/models/model.joblib"

PIPELINE = MLB = None
def ensure_model():
    global PIPELINE, MLB
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        r = requests.get(MODEL_URL, stream=True, timeout=90); r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for ch in r.iter_content(8192):
                if ch: f.write(ch)
    saved = joblib.load(MODEL_PATH)
    PIPELINE, MLB = saved["pipeline"], saved["mlb"]

class Req(BaseModel):
    lang:str="en"; text:str; age:int|None=None; sex:str|None=None; vitals:dict|None=None

@app.get("/")
def health(): return {"ok": True, "model_path": str(MODEL_PATH)}

@app.post("/ai/symptom-check")
def check(req: Req):
    try:
        ensure_model()
        p = PIPELINE.predict_proba([req.text])[0]
        idx = np.argsort(p)[::-1][:3]
        top = [{"id": str(MLB.classes_[i]), "name": str(MLB.classes_[i]).replace("_"," ").title(),
                "prob": float(p[i]), "prob_pct": round(float(p[i])*100,2)} for i in idx]
        return {"top_conditions": top, "triage": {"level":"GP_24_48H"}, "disclaimer":"Educational aid; not a medical diagnosis."}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
