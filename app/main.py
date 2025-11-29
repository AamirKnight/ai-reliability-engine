# app/main.py (FIXED - SHARED CACHE INSTANCE)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from collections import deque
from datetime import datetime
from typing import Optional
from app.services.orchestrator import ReliabilityOrchestrator
from app.services.semantic_cache import SemanticCache, CachedOrchestrator
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Workflow imports
from app.services.workflow_engine import WorkflowEngine
from app.workflows.loan_underwriting import LOAN_WORKFLOW
from app.services.workflow_aggregator import WorkflowResultAggregator
from app.schemas.loan_decision import FinalLoanDecision


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
                "message": "No data yet. Make some requests to /v1/analyze first.",
                "total_decisions": 0
            }
        
        total = len(self.decisions)
        successful = sum(1 for d in self.decisions if d.get("status") == "success")
        
        # Count cache hits
        cache_hits = sum(1 for d in self.decisions 
                        if d.get("meta", {}).get("cache_hit") == True)
        
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
            "cache_hit_rate": f"{(cache_hits/total)*100:.1f}%",
            "average_confidence": f"{avg_confidence:.3f}",
            "average_response_time_ms": f"{avg_response_time:.1f}",
            "blocked_by_confidence_gate": total - successful,
            "recent_decisions": list(self.decisions)[-10:]
        }


# ============================================================================
# CRITICAL: Initialize SINGLE shared cache instance
# ============================================================================
print("🔄 Initializing shared semantic cache...")
shared_semantic_cache = SemanticCache(
    similarity_threshold=0.70,  # Tuned for optimal hit rate
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
workflow_aggregator = WorkflowResultAggregator()


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
        print(f"   Similarity Threshold: {shared_semantic_cache.similarity_threshold}")
    else:
        print(f"❌ CRITICAL: AI System Failure. Error: {health['error']}")
    yield
    print("🛑 Shutting down gracefully...")
    print(f"📊 Final Cache Stats: {shared_semantic_cache.get_stats()}")


# --- FASTAPI APP ---
app = FastAPI(
    title="Gemini Reliability Engine with Semantic Cache",
    description="Production-grade AI workflow with confidence scoring, semantic caching, and metrics",
    version="2.2.0",
    lifespan=lifespan
)

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REQUEST MODELS ---
class RequestBody(BaseModel):
    prompt: str = Field(..., description="The trading analysis prompt")
    confidence_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    use_cache: Optional[bool] = Field(default=True, description="Whether to use semantic cache")


class WorkflowRequest(BaseModel):
    application_text: str = Field(..., description="The loan application text")
    confidence_threshold: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


# --- API ENDPOINTS ---

@app.get("/")
async def root():
    """API documentation pointer"""
    cache_stats = orchestrator.get_cache_stats()
    return {
        "message": "Gemini Reliability Engine with Semantic Cache",
        "version": "2.2.0",
        "status": "online",
        "features": {
            "semantic_caching": True,
            "confidence_scoring": True,
            "circuit_breaker": True,
            "rate_limiting": True,
            "workflow_engine": True
        },
        "cache_stats": {
            "hit_rate": cache_stats["hit_rate"],
            "api_calls_saved": cache_stats["api_calls_saved"],
            "time_saved": cache_stats["time_saved_seconds"]
        },
        "endpoints": {
            "health": "/health",
            "analyze": "/v1/analyze",
            "workflow": "/v1/workflow/execute",
            "metrics": "/v1/metrics",
            "cache_stats": "/v1/cache/stats",
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
    
    # Add cache info to health check
    result["cache"] = orchestrator.get_cache_stats()
    return result


@app.post("/v1/analyze")
async def analyze(body: RequestBody):
    """Single-step trading decision with reliability patterns and semantic caching"""
    result = await orchestrator.run_reliable_workflow(
        user_prompt=body.prompt,
        confidence_threshold=body.confidence_threshold,
        use_cache=body.use_cache
    )
    
    metrics.record(result)
    return result


@app.post("/v1/workflow/execute", response_model=FinalLoanDecision)
async def execute_workflow(body: WorkflowRequest):
    """
    Execute the 5-step loan underwriting workflow.
    Optimized for free tier with sequential execution and semantic caching.
    """
    initial_context = {
        "application_text": body.application_text
    }
    
    # Execute the workflow
    execution = await workflow_engine.execute_workflow(
        workflow_def=LOAN_WORKFLOW,
        initial_context=initial_context,
        confidence_threshold=body.confidence_threshold
    )
    
    # Generate human-readable final decision
    final_decision = workflow_aggregator.generate_loan_decision_summary(execution)
    
    return final_decision


@app.get("/v1/workflow/{execution_id}")
async def get_workflow_status(execution_id: str):
    """Get the detailed status of a workflow execution by ID"""
    execution = workflow_engine.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    final_decision = workflow_aggregator.generate_loan_decision_summary(execution)
    
    return {
        "execution": execution,
        "final_decision": final_decision
    }


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


# --- SEMANTIC CACHE ENDPOINTS ---

@app.get("/v1/cache/stats")
async def get_cache_stats():
    """View semantic cache performance metrics"""
    return orchestrator.get_cache_stats()


@app.get("/v1/cache/top")
async def get_top_cached():
    """View most frequently accessed cached prompts"""
    return {
        "top_prompts": orchestrator.cache.get_top_cached_prompts(limit=10)
    }


@app.post("/v1/cache/clear")
async def clear_cache():
    """Clear the entire cache"""
    orchestrator.cache.invalidate()
    return {"message": "Cache cleared successfully"}


@app.post("/v1/cache/invalidate")
async def invalidate_similar(prompt: str):
    """Invalidate cache entries similar to the provided prompt"""
    orchestrator.cache.invalidate(prompt)
    return {"message": f"Invalidated cache entries similar to: {prompt[:100]}..."}


@app.post("/v1/cache/export")
async def export_cache():
    """Export cache to file for persistence"""
    filepath = "cache_backup.json"
    orchestrator.cache.export_cache(filepath)
    return {"message": f"Cache exported to {filepath}"}


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