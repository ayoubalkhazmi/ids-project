from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys
import io
import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.supervised import SupervisedIDS
from backend.unsupervised import UnsupervisedIDS
from backend.hybrid import HybridIDS
from backend.discovery import discovery_engine
import io
from fastapi.responses import StreamingResponse
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

app = FastAPI(title="Network Security Analyzer - Demo")

# Enable CORS for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models at startup
supervised_model = SupervisedIDS()
unsupervised_model = UnsupervisedIDS()
hybrid_model = HybridIDS(supervised_model, unsupervised_model)
# Define data paths
DATA_DIR = r"d:\AI COURSE\FINAL PROJECT\ids_project\data"
FULL_SET = os.path.join(DATA_DIR, "dataset_sample.csv")
TRAIN_SET = os.path.join(DATA_DIR, "train_stratified.csv")
TEST_SET = os.path.join(DATA_DIR, "test_stratified.csv")

# Train models on startup
print("Training models on startup...")
try:
    # Prefer split datasets if available, else use full sample
    train_file = TRAIN_SET if os.path.exists(TRAIN_SET) else FULL_SET
    print(f"Training on: {os.path.basename(train_file)}")
    
    supervised_model.train(train_file)
    unsupervised_model.train(train_file)
    print("✓ Models trained successfully")
except Exception as e:
    print(f"Warning: Could not train models on startup: {e}")

# --- User-Friendly Analysis Endpoints ---

def translate_supervised_result(predictions, probabilities=None):
    """Translate ML output to user-friendly language"""
    results = []
    for i, pred in enumerate(predictions):
        # Convert to user-friendly status
        if pred == 'Normal':
            status = "Safe"
            statusClass = "safe"
            confidence = "High" if probabilities is None else f"{probabilities[i]:.0%}"
            details = "Normal network activity detected"
        else:
            status = f"Known Attack: {pred}"
            statusClass = "threat"
            confidence = "High" if probabilities is None else f"{probabilities[i]:.0%}"
            details = f"Behavior matches attack pattern for {pred}"
        
        results.append({
            "status": status,
            "statusClass": statusClass,
            "confidence": confidence,
            "details": details
        })
    
    return results

def translate_unsupervised_result(detections):
    """Translate anomaly scores to user-friendly language"""
    results = []
    for detection in detections:
        score = detection['score']
        is_anomaly = detection['prediction'] == 'Anomaly'
        
        # Translate to risk levels
        if is_anomaly:
            if score < -0.1:
                status = "High Risk"
                statusClass = "threat"
                confidence = "High"
                details = "Very unusual behavior detected"
            else:
                status = "Suspicious"
                statusClass = "suspicious"
                confidence = "Medium"
                details = "Slightly unusual behavior"
        else:
            status = "Normal"
            statusClass = "safe"
            confidence = "High"
            details = "Behavior matches normal patterns"
        
        results.append({
            "status": status,
            "statusClass": statusClass,
            "confidence": confidence,
            "details": details
        })
    
    return results

# Global results storage for export (in-memory for demo)
last_analysis_results = []

@app.post("/detect/signature")
async def detect_signature(file: UploadFile = File(...)):
    """Analyze traffic using Known Attack Detection"""
    try:
        global last_analysis_results
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = df.head(100)
        
        data = df.to_dict(orient='records')
        predictions = supervised_model.predict(data)
        
        raw_results = translate_supervised_result(predictions)
        
        last_analysis_results = []
        for r in raw_results:
            last_analysis_results.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": r["status"],
                "statusClass": r["statusClass"],
                "confidence": r["confidence"],
                "details": r["details"]
            })
            
        attack_count = sum(1 for r in last_analysis_results if r['statusClass'] == 'threat')
        safe_count = len(last_analysis_results) - attack_count
        
        return {
            "summary": {
                "total": {"value": len(last_analysis_results), "label": "Traffic Flows Analyzed", "type": "safe", "color": "#94a3b8", "icon": "database"},
                "safe": {"value": safe_count, "label": "Safe Traffic", "type": "safe", "color": "#10b981", "icon": "check-circle"},
                "threats": {"value": attack_count, "label": "Known Attacks Detected", "type": "danger", "color": "#ef4444", "icon": "shield-alert"}
            },
            "explanation": {
                "title": "Known Attack Detection",
                "text": f"This system identified {attack_count} flows matching known attack signatures."
            },
            "results": last_analysis_results,
            "warning": "This mode cannot detect unknown or modified attack methods."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/detect/anomaly")
async def detect_anomaly(file: UploadFile = File(...)):
    """Analyze traffic using Behavior Monitoring"""
    try:
        global last_analysis_results
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = df.head(100)
        
        data = df.to_dict(orient='records')
        detections = unsupervised_model.detect(data)
        
        raw_results = translate_unsupervised_result(detections)
        
        last_analysis_results = []
        for r in raw_results:
            last_analysis_results.append({
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": r["status"],
                "statusClass": r["statusClass"],
                "confidence": r["confidence"],
                "details": r["details"]
            })
            
        anomaly_count = sum(1 for r in last_analysis_results if r['statusClass'] in ['threat', 'suspicious'])
        normal_count = len(last_analysis_results) - anomaly_count
        
        return {
            "summary": {
                "total": {"value": len(last_analysis_results), "label": "Traffic Flows Analyzed", "type": "safe", "color": "#94a3b8", "icon": "database"},
                "normal": {"value": normal_count, "label": "Normal Behavior", "type": "safe", "color": "#10b981", "icon": "check-circle"},
                "anomalies": {"value": anomaly_count, "label": "Unusual Behavior Detected", "type": "warning", "color": "#f59e0b", "icon": "alert-triangle"}
            },
            "explanation": {
                "title": "Behavior Monitoring",
                "text": f"This system flagged {anomaly_count} flows as unusual compared to normal network activity."
            },
            "results": last_analysis_results,
            "warning": "This mode may flag legitimate unusual traffic (false alarms)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# --- Live Traffic Simulation ---

import asyncio
import random

class SimulationManager:
    def __init__(self):
        self.is_running = False
        self.data_pool = None
        self.latest_logs = []
        self.total_processed = 0
        self.safe_count = 0
        self.threat_count = 0
        self.blocked_ips = set() # Store IPs blocked by the user
        self.task = None

    def load_data(self):
        if self.data_pool is None:
            try:
                # Use Test Set for simulation (Unseen Data)
                sim_file = TEST_SET if os.path.exists(TEST_SET) else FULL_SET
                print(f"Simulating traffic from: {os.path.basename(sim_file)}")
                
                df = pd.read_csv(sim_file)
                self.data_pool = df.to_dict(orient='records')
            except Exception as e:
                print(f"Error loading simulation data: {e}")
                self.data_pool = []

    async def run_simulation(self):
        self.load_data()
        if not self.data_pool:
            return

        print("Starting simulation loop...")
        while self.is_running:
            try:
                # Pick a random row or sequential
                row = random.choice(self.data_pool)
                src_ip = str(row.get('Src_IP', '0.0.0.0'))

                # Check if IP is blocked
                if src_ip in self.blocked_ips:
                    # Logic for handling blocked IPs: we can either skip or log as 'Blocked'
                    # Let's log as 'Blocked' to show the user it's working
                    result = {
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "status": "Source Blocked",
                        "statusClass": "safe", # Don't count as threat once blocked
                        "confidence": "100%",
                        "detection_mode": "Firewall Rule",
                        "severity": "N/A",
                        "details": f"Traffic from {src_ip} dropped by firewall.",
                        "src_ip": src_ip
                    }
                else:
                    # Analyze it
                    result = hybrid_model.analyze([row])[0]
                    
                    # Add metadata & UI fields (Fixing missing statusClass)
                    result['timestamp'] = datetime.datetime.now().strftime("%H:%M:%S")
                    result['statusClass'] = "threat" if result["severity"] in ["High", "Critical"] else ("suspicious" if result["severity"] == "Warning" else "safe")
                    result['details'] = f"{result['detection_mode']} ({result['severity']})"
                    result['src_ip'] = src_ip # Add IP for UI actions
                    
                    # Update stats
                    self.total_processed += 1
                    if result['statusClass'] == 'safe':
                        self.safe_count += 1
                    elif result['statusClass'] == 'threat': # Count threats
                        self.threat_count += 1
                    
                # Add to buffer (keep last 50)
                self.latest_logs.append(result)
                if len(self.latest_logs) > 50:
                    self.latest_logs.pop(0)
                    
                # Simulate network delay (Faster for demo feel)
                await asyncio.sleep(random.uniform(0.1, 0.8))
                
            except Exception as e:
                print(f"Simulation Error: {e}")
                await asyncio.sleep(1) # Wait before retry

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.task = asyncio.create_task(self.run_simulation())
            return {"status": "started"}
        return {"status": "already_running"}

    def stop(self):
        self.is_running = False
        if self.task:
            self.task.cancel()
        return {"status": "stopped"}

    def get_updates(self):
        # Return current buffer and stats
        updates = {
            "logs": list(self.latest_logs), # Copy
            "stats": {
                "total": self.total_processed,
                "safe": self.safe_count,
                "threats": self.threat_count
            }
        }
        # Clear buffer so we don't send duplicates? 
        # Better: Frontend handles duplicates or we send only new ones.
        # For simplicity: We send the buffer and clear it from the "unseen" perspective.
        # However, clearing it here means if a polling misses, data is lost.
        # Let's simple return and clear.
        self.latest_logs = [] 
        return updates

sim_manager = SimulationManager()

@app.post("/monitor/start")
async def start_monitor():
    return sim_manager.start()

@app.post("/monitor/stop")
async def stop_monitor():
    return sim_manager.stop()

@app.get("/monitor/updates")
async def get_monitor_updates():
    return sim_manager.get_updates()

class BlockRequest(BaseModel):
    ip: str

@app.post("/monitor/block")
async def block_ip(req: BlockRequest):
    sim_manager.blocked_ips.add(req.ip)
    print(f"FIREWALL: Blocked IP {req.ip}")
    return {"status": "blocked", "ip": req.ip}

# --- End Live Simulation ---

@app.post("/detect/discovery")
async def detect_discovery(file: UploadFile = File(...)):
    """
    Advanced Behavioral Discovery:
    Performs PCA and DBSCAN on the uploaded file for visual analysis.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Limit to 500 records for visualization performance
        df_sample = df.head(500)
        data_list = df_sample.to_dict(orient='records')
        
        results = discovery_engine.get_discovery_map(data_list)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}

@app.post("/detect/hybrid")
async def detect_hybrid(file: UploadFile = File(...)):
    """Analyze traffic using Hybrid Mode (Merged)"""
    try:
        global last_analysis_results
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = df.head(100)
        
        data = df.to_dict(orient='records')
        raw_results = hybrid_model.analyze(data)
        
        # Map to UI format
        last_analysis_results = []
        for r in raw_results:
            last_analysis_results.append({
                "timestamp": r["timestamp"],
                "status": r["status"],
                "statusClass": "threat" if r["severity"] in ["High", "Critical"] else ("suspicious" if r["severity"] == "Warning" else "safe"),
                "confidence": f"{r['confidence']}%",
                "details": f"{r['detection_mode']} - Severity: {r['severity']}"
            })
            
        # Calculate summary
        threat_count = sum(1 for r in last_analysis_results if r['statusClass'] == 'threat')
        suspicious_count = sum(1 for r in last_analysis_results if r['statusClass'] == 'suspicious')
        safe_count = len(last_analysis_results) - threat_count - suspicious_count
        
        return {
            "summary": {
                "total": {"value": len(last_analysis_results), "label": "Traffic Flows Analyzed", "type": "safe", "color": "#94a3b8", "icon": "database"},
                "safe": {"value": safe_count, "label": "Safe Traffic", "type": "safe", "color": "#10b981", "icon": "check-circle"},
                "suspicious": {"value": suspicious_count, "label": "Unknown Behavioral Anomalies", "type": "warning", "color": "#f59e0b", "icon": "alert-triangle"},
                "threats": {"value": threat_count, "label": "Identified Threats", "type": "danger", "color": "#ef4444", "icon": "shield-alert"}
            },
            "explanation": {
                "title": "Hybrid Detection System",
                "text": f"This mode combined both Supervised and Behavior analysis. It first identified known attack signatures and then cross-verified safe-looking traffic for unusual behaviors."
            },
            "results": last_analysis_results,
            "warning": "Hybrid mode provides the best coverage by combining pattern matching with behavioral profiling."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hybrid analysis failed: {str(e)}")

@app.get("/export/csv")
def export_csv():
    """Export last results as CSV"""
    if not last_analysis_results:
        raise HTTPException(status_code=400, detail="No analysis results available")
    
    df = pd.DataFrame(last_analysis_results)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ids_report.csv"}
    )

@app.get("/export/pdf")
def export_pdf():
    """Export last results as PDF"""
    if not last_analysis_results:
        raise HTTPException(status_code=400, detail="No analysis results available")
    
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "Network Security Intrusion Detection Report")
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 70, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    p.drawString(50, height - 90, f"Total Flows Analyzed: {len(last_analysis_results)}")
    
    y = height - 130
    p.setFont("Helvetica-Bold", 10)
    p.drawString(50, y, "Timestamp")
    p.drawString(150, y, "Status")
    p.drawString(250, y, "Mode/Severity")
    p.line(50, y-5, 550, y-5)
    
    y -= 20
    p.setFont("Helvetica", 8)
    for r in last_analysis_results[:30]: # Limit PDF size for demo
        if y < 50:
            p.showPage()
            y = height - 50
        p.drawString(50, y, r["timestamp"])
        p.drawString(150, y, r["status"])
        p.drawString(250, y, r["details"])
        y -= 15
        
    p.showPage()
    p.save()
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=ids_report.pdf"}
    )

# Serve demo frontend at /demo
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
demo_path = os.path.join(project_root, "demo")

if os.path.exists(demo_path):
    app.mount("/static", StaticFiles(directory=demo_path), name="static")

@app.get("/demo")
def demo_page():
    index_path = os.path.join(demo_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        raise HTTPException(status_code=404, detail="Demo not found")

# Root redirects to demo
@app.get("/")
def root():
    return FileResponse(os.path.join(demo_path, "index.html"))
