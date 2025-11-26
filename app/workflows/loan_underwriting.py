# app/workflows/loan_underwriting.py

from app.schemas.workflow import WorkflowDefinition, WorkflowStep

# 5-step simplified loan underwriting workflow
LOAN_WORKFLOW = WorkflowDefinition(
    workflow_id="loan_underwriting_v1",
    name="Loan Underwriting Workflow",
    description="5-step loan application analysis",
    steps=[
        WorkflowStep(
            step_id="step_1_extract",
            name="Extract Applicant Information",
            prompt_template="""
You are a document processing specialist analyzing a loan application.

Application Details:
{application_text}

Extract and analyze the key information. Return your analysis in the required format.

Return JSON with:
- symbol: "LOAN_APP"
- action: "EXTRACT"
- confidence: Your confidence in extraction accuracy (0-1)
- reasoning: Concise summary including: applicant name, annual income, loan amount requested, and employment status
            """,
            depends_on=[],
            retry_count=3
        ),
        
        WorkflowStep(
            step_id="step_2_income",
            name="Verify Income Documents",
            prompt_template="""
You are an income verification analyst.

Previous Analysis:
{step_1_extract}

Based on the extracted information, verify if the stated income is credible and sufficient for the loan request.

Return JSON with:
- symbol: "INCOME_CHECK"
- action: "BUY" (if income is strong), "SELL" (if insufficient), or "HOLD" (if needs review)
- confidence: Your confidence in the verification (0-1)
- reasoning: Brief explanation of your income verification decision (2-3 sentences)
            """,
            depends_on=["step_1_extract"],
            retry_count=3
        ),
        
        WorkflowStep(
            step_id="step_3_dti",
            name="Calculate Debt-to-Income Ratio",
            prompt_template="""
You are a financial analyst calculating debt-to-income ratios.

Context:
- Application Data: {step_1_extract}
- Income Verification: {step_2_income}

Calculate and assess the debt-to-income ratio for this loan.

Return JSON with:
- symbol: "DTI_RATIO"
- action: "BUY" (if DTI is acceptable <43%), "SELL" (if high risk), or "HOLD" (if borderline)
- confidence: Your confidence in the calculation (0-1)
- reasoning: Brief DTI assessment explaining why the ratio is acceptable or concerning
            """,
            depends_on=["step_1_extract", "step_2_income"],
            retry_count=3
        ),
        
        WorkflowStep(
            step_id="step_4_credit",
            name="Assess Credit Risk Profile",
            prompt_template="""
You are a credit risk analyst.

Complete Application Analysis:
- Initial Data: {step_1_extract}
- Income Status: {step_2_income}
- DTI Analysis: {step_3_dti}

Assess the overall credit risk profile for this applicant.

Return JSON with:
- symbol: "CREDIT_RISK"
- action: "BUY" (low risk), "HOLD" (medium risk), or "SELL" (high risk)
- confidence: Your confidence in the risk assessment (0-1)
- reasoning: Brief explanation of the key credit risk factors
            """,
            depends_on=["step_1_extract", "step_2_income", "step_3_dti"],
            retry_count=3
        ),
        
        WorkflowStep(
            step_id="step_5_decision",
            name="Generate Final Loan Decision",
            prompt_template="""
You are the final loan decision authority.

Complete Analysis Review:
1. Application: {step_1_extract}
2. Income: {step_2_income}
3. DTI: {step_3_dti}
4. Credit Risk: {step_4_credit}

Make the final loan decision based on all previous analyses.

Return JSON with:
- symbol: "FINAL_DECISION"
- action: "BUY" (APPROVE loan), "SELL" (REJECT loan), or "HOLD" (needs manual review)
- confidence: Your confidence in the final decision (0-1)
- reasoning: Comprehensive explanation of the final decision citing key factors from all analyses
            """,
            depends_on=["step_1_extract", "step_2_income", "step_3_dti", "step_4_credit"],
            retry_count=2
        )
    ]
)