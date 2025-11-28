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

Extract and analyze the key information from this loan application.

Return JSON with:
- step_type: "EXTRACT"
- status: "PASS" (if extraction successful), "FAIL" (if data missing), or "REVIEW" (if unclear)
- confidence: Your confidence in extraction accuracy (0-1)
- reasoning: Concise summary of what you found
- key_findings: A summary including applicant name, annual income, loan amount requested, employment status, and existing debts
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
- step_type: "INCOME_CHECK"
- status: "PASS" (if income is strong), "FAIL" (if insufficient), or "REVIEW" (if needs manual review)
- confidence: Your confidence in the verification (0-1)
- reasoning: Brief explanation of your income verification decision (2-3 sentences)
- key_findings: State the verified income amount and your assessment
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

Calculate and assess the debt-to-income (DTI) ratio for this loan. A DTI below 43% is generally acceptable.

Return JSON with:
- step_type: "DTI"
- status: "PASS" (if DTI is acceptable <43%), "FAIL" (if high risk >50%), or "REVIEW" (if borderline 43-50%)
- confidence: Your confidence in the calculation (0-1)
- reasoning: Brief DTI assessment explaining the calculated ratio and why it's acceptable or concerning
- key_findings: State the calculated DTI percentage and your conclusion
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

Assess the overall credit risk profile for this applicant based on all available information.

Return JSON with:
- step_type: "CREDIT_RISK"
- status: "PASS" (low risk), "REVIEW" (medium risk), or "FAIL" (high risk)
- confidence: Your confidence in the risk assessment (0-1)
- reasoning: Brief explanation of the key credit risk factors (2-3 sentences)
- key_findings: Overall risk level and primary concerns or strengths
            """,
            depends_on=["step_1_extract", "step_2_income", "step_3_dti"],
            retry_count=3
        ),
        
        WorkflowStep(
            step_id="step_5_decision",
            name="Generate Final Loan Decision",
            prompt_template="""
You are the final loan decision authority making the ultimate approval decision.

Complete Analysis Review:
1. Application: {step_1_extract}
2. Income: {step_2_income}
3. DTI: {step_3_dti}
4. Credit Risk: {step_4_credit}

Make the final loan decision based on all previous analyses. Consider:
- If all previous steps passed, the loan should be APPROVED
- If any critical step failed, the loan should be REJECTED
- If there are concerns but not dealbreakers, it needs MANUAL REVIEW

Return JSON with:
- step_type: "FINAL"
- status: "PASS" (APPROVE loan), "FAIL" (REJECT loan), or "REVIEW" (needs manual review)
- confidence: Your confidence in the final decision (0-1)
- reasoning: Comprehensive explanation of the final decision citing key factors from all analyses (3-4 sentences)
- key_findings: Final recommendation with brief justification
            """,
            depends_on=["step_1_extract", "step_2_income", "step_3_dti", "step_4_credit"],
            retry_count=2
        )
    ]
)