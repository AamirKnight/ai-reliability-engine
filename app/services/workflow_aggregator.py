# app/services/workflow_aggregator.py

from typing import Dict
from app.schemas.workflow import WorkflowExecution, StepStatus
from app.schemas.loan_decision import FinalLoanDecision, LoanStepDecision

class WorkflowResultAggregator:
    """
    Aggregates all workflow step results into a final human-readable conclusion.
    """
    
    def generate_loan_decision_summary(
        self, 
        execution: WorkflowExecution
    ) -> FinalLoanDecision:
        """
        Converts a WorkflowExecution into a comprehensive loan decision.
        """
        # Extract key data from steps
        extract_step = execution.step_executions.get("step_1_extract")
        income_step = execution.step_executions.get("step_2_income")
        dti_step = execution.step_executions.get("step_3_dti")
        credit_step = execution.step_executions.get("step_4_credit")
        final_step = execution.step_executions.get("step_5_decision")
        
        # Determine final decision
        if not final_step or final_step.status != StepStatus.SUCCESS:
            final_decision = "MANUAL_REVIEW"
            executive_summary = "Workflow incomplete. Manual review required due to step failures."
        else:
            final_result = final_step.result
            action = final_result.get("action", "HOLD")
            
            if action == "BUY":
                final_decision = "APPROVED"
            elif action == "SELL":
                final_decision = "REJECTED"
            else:
                final_decision = "MANUAL_REVIEW"
            
            # Build executive summary from reasoning
            executive_summary = final_result.get("reasoning", "Decision based on comprehensive analysis.")
        
        # Extract applicant info from step 1
        applicant_name = "Unknown"
        loan_amount = 0.0
        
        if extract_step and extract_step.result:
            reasoning = extract_step.result.get("reasoning", "")
            # Simple extraction (in production, use structured extraction)
            if "name:" in reasoning.lower():
                parts = reasoning.split("name:")
                if len(parts) > 1:
                    applicant_name = parts[1].split(",")[0].strip()[:50]
            
            if "$" in reasoning:
                import re
                amounts = re.findall(r'\$([0-9,]+)', reasoning)
                if amounts:
                    try:
                        loan_amount = float(amounts[0].replace(",", ""))
                    except:
                        loan_amount = 0.0
        
        # Collect risk factors and strengths
        risk_factors = []
        strengths = []
        
        if income_step and income_step.result:
            action = income_step.result.get("action", "")
            if action == "SELL":
                risk_factors.append("Insufficient or unverified income")
            elif action == "BUY":
                strengths.append("Strong verified income")
        
        if dti_step and dti_step.result:
            action = dti_step.result.get("action", "")
            if action == "SELL":
                risk_factors.append("High debt-to-income ratio")
            elif action == "BUY":
                strengths.append("Healthy debt-to-income ratio")
        
        if credit_step and credit_step.result:
            action = credit_step.result.get("action", "")
            if action == "SELL":
                risk_factors.append("Elevated credit risk profile")
            elif action == "BUY":
                strengths.append("Low credit risk profile")
        
        # Calculate overall confidence
        confidences = []
        for step_exec in execution.step_executions.values():
            if step_exec.confidence_score:
                confidences.append(step_exec.confidence_score)
        
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Build step summary
        step_summary = {}
        for step_id, step_exec in execution.step_executions.items():
            step_summary[step_id] = {
                "status": step_exec.status.value,
                "confidence": step_exec.confidence_score,
                "key_finding": step_exec.result.get("reasoning", "") if step_exec.result else None
            }
        
        return FinalLoanDecision(
            applicant_name=applicant_name,
            loan_amount=loan_amount,
            final_decision=final_decision,
            overall_confidence=overall_confidence,
            executive_summary=executive_summary,
            risk_factors=risk_factors,
            strengths=strengths,
            step_summary=step_summary
        )