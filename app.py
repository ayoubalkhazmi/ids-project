import os
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

# --- Intelligent Path Finder ---
def get_frontend_path():
    # List of possible names (Linux is case-sensitive!)
    choices = ["frontend/index.html", "Frontend/index.html", "index.html"]
    for choice in choices:
        path = BASE_DIR / choice
        if path.exists():
            return path
    return None

@app.get("/")
async def index():
    path = get_frontend_path()
    if path:
        return FileResponse(path)
    
    # If the file is REALLY missing, show a helpful message instead of crashing
    return HTMLResponse(content=f"""
        <h1>Frontend Missing</h1>
        <p>I looked for <b>index.html</b> in the frontend folder but couldn't find it.</p>
        <p>Current Directory Files: {os.listdir(BASE_DIR)}</p>
    """, status_code=404)

# Mount the static folder ONLY if it exists
frontend_folder = BASE_DIR / "frontend"
if not frontend_folder.exists():
    frontend_folder = BASE_DIR / "Frontend"

if frontend_folder.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_folder)), name="static")

# ... keep your @app.post("/detect/{mode}") logic here ...
