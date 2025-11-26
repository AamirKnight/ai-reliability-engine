import sys
import os
import asyncio
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.workflow_engine import WorkflowEngine
from app.workflows.loan_underwriting import LOAN_WORKFLOW

async def run_single_test(failure_rate: float = 0.0):
    """
    Run a single workflow with specified failure rate.
    """
    engine = WorkflowEngine()
    
    # Sample loan application
    initial_context = {
        "application_text": """
        Loan Application:
        - Name: John Smith
        - Annual Income: $85,000
        - Employment: Software Engineer at Tech Corp (5 years)
        - Loan Amount Requested: $300,000
        - Purpose: Home Purchase
        - Existing Debts: $15,000 (car loan + credit cards)
        """
    }
    
    print(f"\n🎯 Testing with {failure_rate*100}% failure rate...")
    
    result = await engine.execute_workflow(
        workflow_def=LOAN_WORKFLOW,
        initial_context=initial_context,
        confidence_threshold=0.65
    )
    
    return result

async def run_comparison_test():
    """
    Run workflows with and without reliability features.
    """
    print("\n" + "="*80)
    print("🧪 RELIABILITY COMPARISON TEST")
    print("="*80)
    
    # Test 1: No failures (baseline)
    print("\n📊 Test 1: Baseline (No Failures)")
    baseline = await run_single_test(failure_rate=0.0)
    
    # Test 2: 20% failure rate
    print("\n📊 Test 2: 20% Failure Rate")
    moderate = await run_single_test(failure_rate=0.20)
    
    # Test 3: 40% failure rate (extreme)
    print("\n📊 Test 3: 40% Failure Rate (Extreme)")
    extreme = await run_single_test(failure_rate=0.40)
    
    # Summary
    print("\n" + "="*80)
    print("📈 RESULTS SUMMARY")
    print("="*80)
    
    results = [
        ("Baseline (0% failures)", baseline),
        ("Moderate (20% failures)", moderate),
        ("Extreme (40% failures)", extreme)
    ]
    
    for name, result in results:
        success_steps = sum(1 for s in result.step_executions.values() if s.status == "success")
        total_steps = len(result.step_executions)
        
        print(f"\n{name}:")
        print(f"  Status: {result.status}")
        print(f"  Steps Completed: {success_steps}/{total_steps}")
        print(f"  Success Rate: {(success_steps/total_steps*100):.1f}%")

if __name__ == "__main__":
    asyncio.run(run_comparison_test())