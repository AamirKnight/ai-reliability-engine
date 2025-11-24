from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.services.orchestrator import ReliabilityOrchestrator

app = FastAPI(title="Gemini 3 Pro Agent")
orchestrator = ReliabilityOrchestrator()

class RequestBody(BaseModel):
    prompt: str

@app.post("/v1/analyze")
async def analyze(body: RequestBody):
    result = orchestrator.run_reliable_workflow(body.prompt)
    if result["status"] == "failure":
        raise HTTPException(status_code=500, detail=result["error"])
    return result

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
# Ensure you are importing from the correct 'services' location now
from app.services.orchestrator import ReliabilityOrchestrator

orchestrator = ReliabilityOrchestrator()

# --- 1. LIFESPAN (Startup Event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🏥 Performing Startup Health Check...")
    health = orchestrator.health_check()
    if health["status"] == "healthy":
        print(f"✅ AI System Online: Connected to {health['model']}")
    else:
        print(f"❌ CRITICAL: AI System Failure. Error: {health['error']}")
        print("💡 TIP: Check your GEMINI_MODEL name in app/core/config.py")
    yield
    print("🛑 Shutting down...")

app = FastAPI(title="Gemini Reliability Engine", lifespan=lifespan)

class RequestBody(BaseModel):
    prompt: str

# --- 2. HEALTH ENDPOINT ---
@app.get("/health")
async def health_check():
    """
    Call this to verify the AI service is alive.
    """
    result = orchestrator.health_check()
    if result["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=result)
    return result

@app.post("/v1/analyze")
async def analyze(body: RequestBody):
    result = orchestrator.run_reliable_workflow(body.prompt)
    if result["status"] == "failure":
        raise HTTPException(status_code=500, detail=result["error"])
    return result