from pydantic import BaseModel, Field, field_validator

class TradeDecision(BaseModel):
    symbol: str = Field(..., description="The ticker symbol (e.g., AAPL)")
    action: str = Field(..., pattern="^(BUY|SELL|HOLD)$", description="The trading action")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(..., description="Short explanation for the decision")

    # LOGIC CHECK: This runs BEFORE your code sees the data
    @field_validator('confidence')
    @classmethod
    def check_confidence_threshold(cls, v: float) -> float:
        if v < 0.1:
            raise ValueError("Confidence is too low (below 10%). Request rejected.")
        return v