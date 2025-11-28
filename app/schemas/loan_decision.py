# app/schemas/loan_decision.py

from pydantic import BaseModel, Field, field_validator
from typing import Literal

class LoanStepDecision(BaseModel):
    """Schema for individual loan workflow steps"""
    step_type: str = Field(..., description="Type of step (EXTRACT, INCOME_CHECK, DTI, CREDIT_RISK, FINAL)")
    status: Literal["PASS", "FAIL", "REVIEW"] = Field(..., description="The step outcome")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Explanation for this step's decision")
    key_findings: str = Field(default="", description="Critical data extracted in this step")

    @field_validator('confidence')
    @classmethod
    def check_confidence_threshold(cls, v: float) -> float:
        if v < 0.1:
            raise ValueError("Confidence is too low (below 10%). Request rejected.")
        return v


class FinalLoanDecision(BaseModel):
    """Schema for the final aggregated loan decision"""
    applicant_name: str = Field(..., description="Name of the loan applicant")
    loan_amount: float = Field(..., description="Requested loan amount in dollars")
    
    final_decision: Literal["APPROVED", "REJECTED", "MANUAL_REVIEW"] = Field(
        ..., 
        description="The final loan decision"
    )
    
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    
    executive_summary: str = Field(
        ..., 
        description="2-3 sentence summary of why this decision was made"
    )
    
    risk_factors: list[str] = Field(
        default_factory=list,
        description="List of key risk factors identified"
    )
    
    strengths: list[str] = Field(
        default_factory=list,
        description="List of applicant strengths"
    )
    
    step_summary: dict = Field(
        default_factory=dict,
        description="Summary of each workflow step's outcome"
    )