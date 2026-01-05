import os
import pandas as pd
import io
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all communication (Fixes the "Nothing happens" bug)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

# --- Logic ---
@app.post("/detect/{mode}")
async def detect(mode: str, file: UploadFile = File(None)):
    # If file is uploaded, count rows; else default to 5
    count = 5
    if file:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        count = min(len(df), 20)
    
    # Generate mock results for the UI
    results = []
    for i in range(count):
        is_threat = i % 3 == 0
        results.append({
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
            "status": "Threat" if is_threat else "Safe",
            "statusClass": "threat" if is_threat else "safe",
            "confidence": "0.98",
            "details": f"Analysis via {mode}",
        })
    
    return {
        "summary": {
            "safe": {"value": count - (count//3), "label": "Safe", "color": "#10b981"},
            "threats": {"value": count//3, "label": "Threats", "color": "#ef4444"}
        },
        "results": results
    }

# --- Serve Frontend ---
# Make sure your folder is named 'frontend'
if os.path.exists(BASE_DIR / "frontend"):
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "frontend")), name="static")

@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "frontend" / "index.html")

if __name__ == "__main__":
    import uvicorn
    # Render provides a PORT environment variable
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)