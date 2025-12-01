from app.schemas.workflow import WorkflowDefinition, WorkflowStep

TRADING_WORKFLOW = WorkflowDefinition(
    workflow_id="stock_analysis_v1",
    name="Institutional Trading Analysis",
    description="4-step comprehensive market analysis",
    steps=[
        WorkflowStep(
            step_id="step_1_sentiment",
            name="Market Sentiment Analysis",
            prompt_template="""
You are a sentiment analyst. Analyze the following request:
"{application_text}"

Determine the current market sentiment and news impact for this asset.

Return JSON with:
- step_type: "SENTIMENT"
- status: "PASS" (if sentiment is positive/bullish), "FAIL" (if negative/bearish), or "REVIEW" (if mixed)
- confidence: Score between 0-1
- reasoning: Explain the key news drivers and sentiment
- key_findings: Summary of bullish vs bearish signals
            """,
            depends_on=[],
            retry_count=3
        ),
        
        WorkflowStep(
            step_id="step_2_technicals",
            name="Technical Indicator Check",
            prompt_template="""
You are a technical analyst.
Context: {step_1_sentiment}

Analyze the technical setup (Trends, Support/Resistance, RSI, MACD) implies by the request or general market knowledge for the asset.

Return JSON with:
- step_type: "TECHNICALS"
- status: "PASS" (if technicals suggest Entry/Buy), "FAIL" (if technicals suggest Exit/Sell), or "REVIEW" (Neutral)
- confidence: Score between 0-1
- reasoning: Explanation of chart patterns and indicators
- key_findings: Key price levels (Support/Resistance)
            """,
            depends_on=["step_1_sentiment"],
            retry_count=3
        ),
        
        WorkflowStep(
            step_id="step_3_risk",
            name="Risk & Volatility Assessment",
            prompt_template="""
You are a risk manager.
Context:
- Sentiment: {step_1_sentiment}
- Technicals: {step_2_technicals}

Assess the risk profile. Is this trade too volatile?

Return JSON with:
- step_type: "RISK"
- status: "PASS" (Low/Manageable Risk), "FAIL" (High Risk/Gambling), "REVIEW" (Moderate Risk)
- confidence: Score between 0-1
- reasoning: Analysis of volatility and downside potential
- key_findings: Recommended Stop Loss area or risk warning
            """,
            depends_on=["step_1_sentiment", "step_2_technicals"],
            retry_count=3
        ),
        
        WorkflowStep(
            step_id="step_4_final",
            name="Final Trade Execution",
            prompt_template="""
You are the Head Trader.
Review all analysis:
1. Sentiment: {step_1_sentiment}
2. Technicals: {step_2_technicals}
3. Risk: {step_3_risk}

Make the final call. 
- PASS = BUY / LONG
- FAIL = SELL / SHORT / AVOID
- REVIEW = HOLD / WAIT

Return JSON with:
- step_type: "FINAL"
- status: "PASS" (Buy), "FAIL" (Sell/Avoid), or "REVIEW" (Hold)
- confidence: Overall confidence score
- reasoning: Final executive summary combining all factors
- key_findings: The final verdict and trade structure
            """,
            depends_on=["step_1_sentiment", "step_2_technicals", "step_3_risk"],
            retry_count=2
        )
    ]
)