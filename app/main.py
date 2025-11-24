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