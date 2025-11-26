import sys
import os
import asyncio
import time

# FIX: Add project root to python path. 
# We traverse up 3 levels: scripts -> app -> project_root
# This ensures we can do "from app.services..."
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.orchestrator import ReliabilityOrchestrator
from app.services.circuit_breaker import CircuitState

async def run_diagnostics():
    print("🔎 STARTING DEEP SYSTEM DIAGNOSTIC...\n")
    
    orch = ReliabilityOrchestrator()
    report = {}

    # 1. API KEY CHECK
    if orch.client.api_key:
        print("✅ [1/5] API Key Configured")
        report["Config"] = "PASS"
    else:
        print("❌ [1/5] API Key Missing")
        report["Config"] = "FAIL"

    # 2. CIRCUIT BREAKER STATE
    if orch.cb.state == CircuitState.CLOSED:
        print("✅ [2/5] Circuit Breaker Initialized (State: CLOSED)")
        report["CircuitBreaker"] = "PASS"
    else:
        print(f"❌ [2/5] Circuit Breaker Invalid State: {orch.cb.state}")
        report["CircuitBreaker"] = "FAIL"

    # 3. CONNECTIVITY (Sync Check)
    print("🔄 [3/5] Testing Google Gemini Connectivity...")
    health = orch.health_check()
    if health["status"] == "healthy":
        print(f"✅ Connectivity OK ({health.get('model')})")
        report["Connectivity"] = "PASS"
    else:
        print(f"❌ Connectivity Failed: {health.get('error')}")
        report["Connectivity"] = "FAIL"

    # 4. PARALLEL EXECUTION & VOTING
    print("\n🧠 [4/5] Testing Parallel Execution & Voting (Async)...")
    print("   (This runs 3 simultaneous agents...)")
    try:
        start = time.time()
        result = await orch.run_reliable_workflow(
            "Is now a good time to buy gold?", 
            confidence_threshold=0.5 # Low threshold to ensure pass for testing
        )
        duration = time.time() - start
        
        if result["status"] == "success":
            meta = result["meta"]
            print(f"✅ Workflow Succeeded in {duration:.2f}s")
            print(f"   - Agents Used: {meta['parallel_agents']}")
            print(f"   - Confidence: {result['confidence_metrics']['overall_confidence']:.2f}")
            print(f"   - Consensus: {result['data']['action']}")
            report["Workflow"] = "PASS"
        else:
            print(f"❌ Workflow Failed: {result.get('error')}")
            report["Workflow"] = "FAIL"
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR during workflow: {e}")
        report["Workflow"] = "CRITICAL FAIL"

    # 5. ERROR SIMULATION (Optional)
    print("\n⚡ [5/5] Circuit Breaker Logic Check...")
    # Manually trigger a failure
    orch.cb._on_failure()
    if orch.cb.failure_count == 1:
        print("✅ Failure correctly counted")
        report["Logic"] = "PASS"
    else:
        print("❌ Failure counting broken")
        report["Logic"] = "FAIL"

    print("\n" + "="*30)
    print("DIAGNOSTIC REPORT")
    print("="*30)
    all_pass = True
    for k, v in report.items():
        print(f"{k:<15}: {v}")
        if "FAIL" in v: all_pass = False
    
    if all_pass:
        print("\n🚀 ALL SYSTEMS OPERATIONAL.")
    else:
        print("\n⚠️ SYSTEM ISSUES DETECTED.")

if __name__ == "__main__":
    asyncio.run(run_diagnostics())