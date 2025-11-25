# app/main.py

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from collections import deque
from datetime import datetime
from typing import Optional
from app.services.orchestrator import ReliabilityOrchestrator


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
        
        # Calculate average confidence (only for successful decisions)
        successful_decisions = [d for d in self.decisions if d.get("status") == "success"]
        avg_confidence = 0.0
        if successful_decisions:
            confidences = [
                d.get("confidence_metrics", {}).get("overall_confidence", 0)
                for d in successful_decisions
            ]
            avg_confidence = sum(confidences) / len(confidences)
        
        # Calculate average response time
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
            "recent_decisions": list(self.decisions)[-10:]  # Last 10 for inspection
        }


# Initialize global instances
orchestrator = ReliabilityOrchestrator()
metrics = MetricsCollector()


# --- LIFESPAN (Startup/Shutdown Events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup health check and graceful shutdown"""
    print("🏥 Performing Startup Health Check...")
    health = orchestrator.health_check()
    if health["status"] == "healthy":
        print(f"✅ AI System Online: Connected to {health['model']}")
    else:
        print(f"❌ CRITICAL: AI System Failure. Error: {health['error']}")
        print("💡 TIP: Check your GEMINI_MODEL name in app/core/config.py")
    yield
    print("🛑 Shutting down gracefully...")


# --- FASTAPI APP ---
app = FastAPI(
    title="Gemini Reliability Engine",
    description="Production-grade AI workflow with confidence scoring and metrics",
    version="2.0.0",
    lifespan=lifespan
)


# --- REQUEST MODELS ---
class RequestBody(BaseModel):
    prompt: str = Field(..., description="The trading analysis prompt")
    confidence_threshold: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required (0.0-1.0). Lower = more permissive."
    )


# --- API ENDPOINTS ---

@app.get("/")
async def root():
    """API documentation pointer"""
    return {
        "message": "Gemini Reliability Engine",
        "endpoints": {
            "health": "/health - Check system health",
            "analyze": "/v1/analyze - Make a trading decision",
            "metrics": "/v1/metrics - View performance statistics"
        },
        "docs": "/docs - Interactive API documentation"
    }


@app.get("/health")
async def health_check():
    """
    Verify the AI service is operational.
    Returns 503 if unhealthy.
    """
    result = orchestrator.health_check()
    if result["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/v1/analyze")
async def analyze(body: RequestBody):
    """
    Analyze a trading prompt with confidence-based quality gates.
    
    Example request:
    ```json
    {
        "prompt": "Should I buy AAPL stock?",
        "confidence_threshold": 0.7
    }
    ```
    """
    result = orchestrator.run_reliable_workflow(
        user_prompt=body.prompt,
        confidence_threshold=body.confidence_threshold
    )
    
    # Record metrics for monitoring
    metrics.record(result)
    
    # Return error if workflow failed
    if result["status"] == "failure":
        raise HTTPException(
            status_code=500,
            detail={
                "error": result["error"],
                "message": "Decision workflow failed after retries"
            }
        )
    
    return result


@app.get("/v1/metrics")
async def get_metrics():
    """
    View aggregated performance metrics.
    
    Shows:
    - Success rate
    - Average confidence scores
    - Response times
    - Recent decisions
    """
    return metrics.get_stats()


@app.post("/v1/metrics/reset")
async def reset_metrics():
    """
    Clear all stored metrics (useful for testing).
    """
    global metrics
    metrics = MetricsCollector()
    return {"message": "Metrics cleared successfully"}