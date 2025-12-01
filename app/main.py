

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from collections import deque
from datetime import datetime
from typing import Optional
import os

from app.services.orchestrator import ReliabilityOrchestrator
from app.services.semantic_cache import SemanticCache, CachedOrchestrator
from app.services.workflow_engine import WorkflowEngine
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- 🚀 NEW TRADING IMPORTS ---
from app.workflows.trading_workflow import TRADING_WORKFLOW
from app.schemas.workflow import WorkflowExecution


# --- METRICS COLLECTOR ---
class MetricsCollector:
    """In-memory metrics storage"""
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
                "message": "No data yet.",
                "total_decisions": 0
            }
        
        total = len(self.decisions)
        successful = sum(1 for d in self.decisions if d.get("status") == "success")
        
        # Count cache hits
        cache_hits = sum(1 for d in self.decisions 
                        if d.get("meta", {}).get("cache_hit") == True)
        
        return {
            "total_decisions": total,
            "successful_decisions": successful,
            "success_rate": f"{(successful/total)*100:.1f}%" if total > 0 else "0%",
            "cache_hit_rate": f"{(cache_hits/total)*100:.1f}%" if total > 0 else "0%",
            "recent_decisions": list(self.decisions)[-10:]
        }


# ============================================================================
# CRITICAL: Initialize SINGLE shared cache instance
# ============================================================================
print("🔄 Initializing shared semantic cache...")
shared_semantic_cache = SemanticCache(
    similarity_threshold=0.80,  # Tuned for trading queries
    max_cache_size=1000,
    ttl_hours=24
)

# Initialize orchestrator with shared cache
base_orchestrator = ReliabilityOrchestrator(use_parallel=False)
orchestrator = CachedOrchestrator(base_orchestrator, shared_semantic_cache)

# Initialize metrics
metrics = MetricsCollector()

# Initialize workflow engine with the SAME shared cache
workflow_engine = WorkflowEngine(cache=shared_semantic_cache)


# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup health check and graceful shutdown"""
    print("🏥 Performing Startup Health Check...")
    health = orchestrator.health_check()
    if health["status"] == "healthy":
        print(f"✅ AI System Online: Connected to {health['model']}")
        print(f"📊 Rate Limit: {health.get('rate_limit', {})}")
        print(f"🧠 Semantic Cache: Initialized with {shared_semantic_cache.max_cache_size} max entries")
    else:
        print(f"❌ CRITICAL: AI System Failure. Error: {health['error']}")
    yield
    print("🛑 Shutting down gracefully...")
    print(f"📊 Final Cache Stats: {shared_semantic_cache.get_stats()}")


# --- FASTAPI APP ---
app = FastAPI(
    title="TradeGuard AI Engine",
    description="Institutional-grade stock analysis workflow with reliability features",
    version="3.0.0",
    lifespan=lifespan
)

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REQUEST MODELS ---
class RequestBody(BaseModel):
    prompt: str = Field(..., description="The trading analysis prompt")
    confidence_threshold: Optional[float] = Field(default=0.6, ge=0.0, le=1.0)
    use_cache: Optional[bool] = Field(default=True, description="Whether to use semantic cache")


class WorkflowRequest(BaseModel):
    application_text: str = Field(..., description="The stock query or analysis request")
    confidence_threshold: Optional[float] = Field(default=0.6, ge=0.0, le=1.0)


# --- API ENDPOINTS ---

@app.get("/")
async def root():
    """API documentation pointer"""
    cache_stats = orchestrator.get_cache_stats()
    return {
        "message": "TradeGuard AI Engine Online",
        "version": "3.0.0",
        "status": "online",
        "cache_stats": {
            "hit_rate": cache_stats["hit_rate"],
            "time_saved": cache_stats["time_saved_seconds"]
        }
    }


@app.get("/health")
async def health_check():
    """Verify the AI service is operational"""
    result = orchestrator.health_check()
    if result["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=result)
    result["cache"] = orchestrator.get_cache_stats()
    return result


@app.post("/v1/analyze")
async def analyze(body: RequestBody):
    """Single-step trading decision (Legacy endpoint)"""
    result = await orchestrator.run_reliable_workflow(
        user_prompt=body.prompt,
        confidence_threshold=body.confidence_threshold,
        use_cache=body.use_cache
    )
    metrics.record(result)
    return result


# 🚀 NEW TRADING WORKFLOW ENDPOINT
@app.post("/v1/workflow/execute", response_model=WorkflowExecution)
async def execute_workflow(body: WorkflowRequest):
    """
    Executes the 4-step Stock Trading Analysis Workflow.
    Returns the full execution history for the frontend timeline.
    """
    initial_context = {
        "application_text": body.application_text
    }
    
    # Execute the Trading Workflow
    execution = await workflow_engine.execute_workflow(
        workflow_def=TRADING_WORKFLOW,
        initial_context=initial_context,
        confidence_threshold=body.confidence_threshold
    )
    
    # Return the raw execution object so frontend can render steps
    return execution


@app.get("/v1/workflow/{execution_id}")
async def get_workflow_status(execution_id: str):
    """Get the detailed status of a workflow execution by ID"""
    execution = workflow_engine.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    return execution


@app.get("/v1/metrics")
async def get_metrics():
    """View aggregated performance metrics"""
    return metrics.get_stats()


# --- SEMANTIC CACHE ENDPOINTS ---

@app.get("/v1/cache/stats")
async def get_cache_stats():
    return orchestrator.get_cache_stats()

@app.post("/v1/cache/clear")
async def clear_cache():
    orchestrator.cache.invalidate()
    return {"message": "Cache cleared successfully"}
