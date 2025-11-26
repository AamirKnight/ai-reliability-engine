# app/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # 🟢 NEW: CORS import
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from collections import deque
from datetime import datetime
from typing import Optional
from app.services.orchestrator import ReliabilityOrchestrator
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# 🟢 Workflow imports
from app.services.workflow_engine import WorkflowEngine
from app.workflows.loan_underwriting import LOAN_WORKFLOW


# --- METRICS COLLECTOR ---
class MetricsCollector:
    """In-memory metrics storage (replace with Redis/DB in production)"""
    def __init__(self, max_size=1000):
        self.decisions = deque(maxlen=max_size)
    
    def record(self, decision_result: dict):
        """Store decision result with timestamp"""
        self.decisions.append({
            **decision_result,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_stats(self) -> dict:
        """Calculate aggregated statistics"""
        if not self.decisions:
            return {
                "message": "No data yet. Make some requests to /v1/analyze first.",
                "total_decisions": 0
            }
        
        total = len(self.decisions)
        successful = sum(1 for d in self.decisions if d.get("status") == "success")
        
        successful_decisions = [d for d in self.decisions if d.get("status") == "success"]
        avg_confidence = 0.0
        if successful_decisions:
            confidences = [
                d.get("confidence_metrics", {}).get("overall_confidence", 0)
                for d in successful_decisions
            ]
            avg_confidence = sum(confidences) / len(confidences)
        
        response_times = [
            d.get("response_time_ms", 0) 
            for d in self.decisions 
            if d.get("response_time_ms")
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "total_decisions": total,
            "successful_decisions": successful,
            "success_rate": f"{(successful/total)*100:.1f}%",
            "average_confidence": f"{avg_confidence:.3f}",
            "average_response_time_ms": f"{avg_response_time:.1f}",
            "blocked_by_confidence_gate": total - successful,
            "recent_decisions": list(self.decisions)[-10:]
        }


# Initialize global instances
orchestrator = ReliabilityOrchestrator()
metrics = MetricsCollector()
workflow_engine = WorkflowEngine()


# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup health check and graceful shutdown"""
    print("🏥 Performing Startup Health Check...")
    health = orchestrator.health_check()
    if health["status"] == "healthy":
        print(f"✅ AI System Online: Connected to {health['model']}")
    else:
        print(f"❌ CRITICAL: AI System Failure. Error: {health['error']}")
    yield
    print("🛑 Shutting down gracefully...")


# --- FASTAPI APP ---
app = FastAPI(
    title="Gemini Reliability Engine",
    description="Production-grade AI workflow with confidence scoring and metrics",
    version="2.0.0",
    lifespan=lifespan
)

# 🟢 CORS MIDDLEWARE - CRITICAL for Next.js communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",  # Alternative localhost
        "http://localhost:3001",  # Backup port
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# --- REQUEST MODELS ---
class RequestBody(BaseModel):
    prompt: str = Field(..., description="The trading analysis prompt")
    confidence_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


class WorkflowRequest(BaseModel):
    application_text: str = Field(..., description="The loan application text")
    confidence_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


# --- API ENDPOINTS ---

@app.get("/")
async def root():
    """API documentation pointer"""
    return {
        "message": "Gemini Reliability Engine",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "analyze": "/v1/analyze",
            "workflow": "/v1/workflow/execute",
            "metrics": "/v1/metrics",
            "dashboard": "/dashboard"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Verify the AI service is operational"""
    result = orchestrator.health_check()
    if result["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/v1/analyze")
async def analyze(body: RequestBody):
    """Single-step trading decision with reliability patterns"""
    result = await orchestrator.run_reliable_workflow(
        user_prompt=body.prompt,
        confidence_threshold=body.confidence_threshold
    )
    
    metrics.record(result)
    return result


@app.post("/v1/workflow/execute")
async def execute_workflow(body: WorkflowRequest):
    """
    Execute the 5-step loan underwriting workflow.
    
    Returns a WorkflowExecution object with:
    - execution_id: Unique workflow run identifier
    - status: "running" | "completed" | "failed"
    - step_executions: Dict of each step's result
    - context: Shared data between steps
    """
    initial_context = {
        "application_text": body.application_text
    }
    
    result = await workflow_engine.execute_workflow(
        workflow_def=LOAN_WORKFLOW,
        initial_context=initial_context,
        confidence_threshold=body.confidence_threshold
    )
    
    return result


@app.get("/v1/workflow/{execution_id}")
async def get_workflow_status(execution_id: str):
    """Get the status of a workflow execution by ID"""
    execution = workflow_engine.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    return execution


@app.get("/v1/metrics")
async def get_metrics():
    """View aggregated performance metrics"""
    return metrics.get_stats()


@app.post("/v1/metrics/reset")
async def reset_metrics():
    """Clear all stored metrics"""
    global metrics
    metrics = MetricsCollector()
    return {"message": "Metrics cleared successfully"}


@app.get("/dashboard")
async def dashboard():
    """Serve the metrics dashboard"""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    dashboard_path = os.path.join(static_dir, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return {"error": "Dashboard not found"}


# Mount static files LAST
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")