from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import os
import sys

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.supervised import SupervisedIDS
from backend.unsupervised import UnsupervisedIDS

app = FastAPI(title="Web-based IDS Comparison")

# --- Global State (Simple/Demo) ---
supervised_model = SupervisedIDS()
unsupervised_model = UnsupervisedIDS()
DATASET_PATH = r"d:\AI COURSE\FINAL PROJECT\ids_project\data\dataset_sample.csv"

# --- Models for API ---
class PredictRequest(BaseModel):
    data: list  # List of dictionaries

# --- Routes ---

@app.post("/supervised/train")
def train_supervised():
    try:
        metrics = supervised_model.train(DATASET_PATH)
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/supervised/predict")
def predict_supervised(req: PredictRequest):
    try:
        predictions = supervised_model.predict(req.data)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unsupervised/train")
def train_unsupervised():
    try:
        stats = unsupervised_model.train(DATASET_PATH)
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unsupervised/detect")
def detect_unsupervised(req: PredictRequest):
    try:
        results = unsupervised_model.detect(req.data)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/sample")
def get_sample_data():
    """Get a few random rows for testing."""
    try:
        df = pd.read_csv(DATASET_PATH)
        sample = df.sample(5).to_dict(orient="records")
        return {"data": sample}
    except Exception as e:
         raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/supervised")
def get_supervised_metrics():
    """Retrieve supervised model metrics after training."""
    if supervised_model.metrics:
        return {"metrics": supervised_model.metrics}
    else:
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /supervised/train first.")

@app.get("/metrics/unsupervised")
def get_unsupervised_stats():
    """Retrieve unsupervised model statistics after training."""
    if unsupervised_model.stats:
        return {"stats": unsupervised_model.stats}
    else:
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /unsupervised/train first.")

# --- Serve Frontend ---
# Use absolute path based on this file's location to avoid CWD issues
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(project_root, "frontend")

# Serve static files (CSS, JS) at /static
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Serve index.html at root
@app.get("/")
def read_root():
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="Frontend not found")
