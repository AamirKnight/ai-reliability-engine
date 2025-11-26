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
    # We rely on the health check to indirectly test client initialization
    try:
        # Check if the client was initialized with a key
        if orch.client.api_key:
            print("✅ [1/5] API Key Configured")
            report["Config"] = "PASS"
        else:
            print("❌ [1/5] API Key Missing (Client initialized without key)")
            report["Config"] = "FAIL"
    except Exception:
        print("❌ [1/5] Client initialization failed.")
        report["Config"] = "FAIL"


    # 2. CIRCUIT BREAKER & CONNECTIVITY
    print("🔄 [2/5 & 3/5] Testing Connectivity & Circuit Breaker...")
    # Health check is delegated to the AgentWrapper/CircuitBreaker
    health = orch.health_check()
    
    # Check 2: Circuit Breaker State (Reported by health_check)
    cb_state = health.get("circuit_state", "UNKNOWN")
    if cb_state == CircuitState.CLOSED.value:
        print("✅ [2/5] Circuit Breaker Initialized (State: CLOSED)")
        report["CircuitBreaker"] = "PASS"
    else:
        print(f"❌ [2/5] Circuit Breaker Invalid State: {cb_state}")
        report["CircuitBreaker"] = "FAIL"

    # Check 3: Connectivity
    if health["status"] == "healthy":
        print(f"✅ [3/5] Google Gemini Connected ({health.get('model')})")
        report["Connectivity"] = "PASS"
    else:
        print(f"❌ [3/5] Connectivity Failed: {health.get('error')}")
        report["Connectivity"] = "FAIL"


    # 4. PARALLEL EXECUTION & VOTING
    print("\n🧠 [4/5] Testing Full Reliability Workflow (Async)...")
    print("   (This runs 3 simultaneous agents...)")
    try:
        start = time.time()
        result = await orch.run_reliable_workflow(
            "Is now a good time to buy gold, considering recent inflation and geopolitical tension?", 
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
    print("\n⚡ [5/5] Circuit Breaker Logic Check (Manual Failure Trigger)...")
    # Manually trigger a failure
    orch.cb._on_failure()
    if orch.cb.failure_count == 1:
        print("✅ Failure correctly counted (Count: 1)")
        report["Logic"] = "PASS"
    else:
        print(f"❌ Failure counting broken (Count: {orch.cb.failure_count})")
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