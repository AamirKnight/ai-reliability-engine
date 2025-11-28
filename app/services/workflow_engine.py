# app/services/workflow_engine.py

import uuid
import asyncio
from datetime import datetime
from typing import Dict, Optional
from app.schemas.workflow import (
    WorkflowDefinition, 
    WorkflowExecution, 
    StepExecution, 
    StepStatus
)
from app.services.orchestrator import ReliabilityOrchestrator
from app.schemas.loan_decision import LoanStepDecision  # NEW


class WorkflowEngine:
    def __init__(self):
        self.orchestrator = ReliabilityOrchestrator()
        self.active_executions: Dict[str, WorkflowExecution] = {}
    
    async def execute_workflow(
        self, 
        workflow_def: WorkflowDefinition,
        initial_context: Dict = None,
        confidence_threshold: float = 0.7
    ) -> WorkflowExecution:
        """
        Execute a complete workflow from start to finish.
        """
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow_def.workflow_id,
            status="running",
            context=initial_context or {},
            created_at=datetime.now().isoformat()
        )
        
        self.active_executions[execution_id] = execution
        
        print(f"\n{'='*60}")
        print(f"🚀 WORKFLOW STARTED: {workflow_def.name}")
        print(f"   Execution ID: {execution_id}")
        print(f"   Total Steps: {len(workflow_def.steps)}")
        print(f"{'='*60}\n")
        
        try:
            for step_def in workflow_def.steps:
                # Check if dependencies are met
                if not self._dependencies_met(step_def, execution):
                    print(f"⏭️  Skipping {step_def.name} (dependencies not met)")
                    execution.step_executions[step_def.step_id] = StepExecution(
                        step_id=step_def.step_id,
                        status=StepStatus.SKIPPED
                    )
                    continue
                
                # Execute the step
                execution.current_step = step_def.step_id
                step_result = await self._execute_step(
                    step_def, 
                    execution, 
                    confidence_threshold
                )
                
                execution.step_executions[step_def.step_id] = step_result
                
                # If step failed and is required, stop workflow
                if step_result.status == StepStatus.FAILED and step_def.required:
                    execution.status = "failed"
                    print(f"\n❌ WORKFLOW FAILED at step: {step_def.name}")
                    break
                
                # Update context with step results
                if step_result.result:
                    execution.context[step_def.step_id] = step_result.result
            
            # Mark as completed if we got through all steps
            if execution.status == "running":
                execution.status = "completed"
                execution.completed_at = datetime.now().isoformat()
                print(f"\n✅ WORKFLOW COMPLETED SUCCESSFULLY")
        
        except Exception as e:
            execution.status = "failed"
            print(f"\n💥 WORKFLOW CRASHED: {str(e)}")
        
        return execution
    
    async def _execute_step(
        self,
        step_def,
        execution: WorkflowExecution,
        confidence_threshold: float
    ) -> StepExecution:
        """
        Execute a single step with retry logic.
        Uses LoanStepDecision schema for loan workflows.
        """
        step_exec = StepExecution(
            step_id=step_def.step_id,
            status=StepStatus.RUNNING,
            started_at=datetime.now().isoformat()
        )
        
        print(f"\n{'─'*60}")
        print(f"⚙️  Step {step_def.step_id}: {step_def.name}")
        print(f"{'─'*60}")
        
        for attempt in range(step_def.retry_count):
            try:
                step_exec.attempts = attempt + 1
                
                # Build prompt with context
                prompt = self._build_prompt(step_def, execution)
                
                print(f"   Attempt {attempt + 1}/{step_def.retry_count}...")
                
                # Use LoanStepDecision schema for loan workflow steps
                result = await self.orchestrator.run_reliable_workflow(
                    user_prompt=prompt,
                    confidence_threshold=confidence_threshold,
                    response_schema=LoanStepDecision  # NEW: Use proper schema
                )
                
                if result["status"] == "success":
                    step_exec.status = StepStatus.SUCCESS
                    step_exec.result = result["data"]
                    step_exec.confidence_score = result["confidence_metrics"]["overall_confidence"]
                    step_exec.completed_at = datetime.now().isoformat()
                    
                    print(f"   ✅ Step succeeded (confidence: {step_exec.confidence_score:.2f})")
                    return step_exec
                else:
                    print(f"   ⚠️  Attempt {attempt + 1} failed: {result.get('error')}")
            
            except Exception as e:
                print(f"   ❌ Attempt {attempt + 1} crashed: {str(e)}")
                step_exec.error = str(e)
            
            # Wait before retry (exponential backoff)
            if attempt < step_def.retry_count - 1:
                await asyncio.sleep(2 ** attempt)
        
        # All retries exhausted
        step_exec.status = StepStatus.FAILED
        step_exec.completed_at = datetime.now().isoformat()
        print(f"   ❌ Step failed after {step_def.retry_count} attempts")
        
        return step_exec
    
    def _dependencies_met(
        self, 
        step_def, 
        execution: WorkflowExecution
    ) -> bool:
        """
        Check if all dependency steps completed successfully.
        """
        if not step_def.depends_on:
            return True
        
        for dep_id in step_def.depends_on:
            dep_exec = execution.step_executions.get(dep_id)
            if not dep_exec or dep_exec.status != StepStatus.SUCCESS:
                return False
        
        return True
    
    def _build_prompt(
        self, 
        step_def, 
        execution: WorkflowExecution
    ) -> str:
        """
        Build the prompt by injecting context from previous steps.
        """
        prompt = step_def.prompt_template
        
        # Replace placeholders with context values
        for key, value in execution.context.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                # Convert dict to readable string
                value_str = str(value) if not isinstance(value, dict) else str(value)
                prompt = prompt.replace(placeholder, value_str)
        
        return prompt
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Retrieve a workflow execution by ID."""
        return self.active_executions.get(execution_id)