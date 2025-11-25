# app/services/confidence_engine.py

from typing import Dict, Any
from pydantic import BaseModel

class ConfidenceMetrics(BaseModel):
    """Stores all confidence components"""
    schema_validity: float  # Did it match the expected structure?
    semantic_coherence: float  # Does the reasoning make sense?
    output_consistency: float  # Matches expected patterns?
    historical_performance: float  # Track record of this agent
    overall_confidence: float  # Weighted average

class ConfidenceEngine:
    def __init__(self):
        # Track success rates over time
        self.success_history = []
        
    def calculate_confidence(
        self, 
        decision: Dict[str, Any],
        validation_passed: bool,
        response_time_ms: float
    ) -> ConfidenceMetrics:
        """
        Multi-signal confidence scoring system
        """
        
        # 1. Schema Validity (30% weight)
        schema_score = 1.0 if validation_passed else 0.0
        
        # 2. Semantic Coherence (30% weight)
        semantic_score = self._check_semantic_coherence(decision)
        
        # 3. Output Consistency (20% weight)
        consistency_score = self._check_output_consistency(decision)
        
        # 4. Historical Performance (20% weight)
        historical_score = self._get_historical_performance()
        
        # Weighted average
        overall = (
            schema_score * 0.3 +
            semantic_score * 0.3 +
            consistency_score * 0.2 +
            historical_score * 0.2
        )
        
        return ConfidenceMetrics(
            schema_validity=schema_score,
            semantic_coherence=semantic_score,
            output_consistency=consistency_score,
            historical_performance=historical_score,
            overall_confidence=overall
        )
    
    def _check_semantic_coherence(self, decision: Dict[str, Any]) -> float:
        """
        Check if the reasoning aligns with the action
        """
        action = decision.get('action', '').upper()
        reasoning = decision.get('reasoning', '').lower()
        confidence = decision.get('confidence', 0)
        
        score = 1.0
        
        # Rule 1: BUY should mention positive terms
        if action == "BUY":
            positive_terms = ['growth', 'strong', 'positive', 'upward', 'bullish']
            if not any(term in reasoning for term in positive_terms):
                score -= 0.3
        
        # Rule 2: SELL should mention negative terms
        elif action == "SELL":
            negative_terms = ['decline', 'weak', 'negative', 'downward', 'bearish']
            if not any(term in reasoning for term in negative_terms):
                score -= 0.3
        
        # Rule 3: High confidence should have longer reasoning
        if confidence > 0.8 and len(reasoning) < 50:
            score -= 0.2
        
        # Rule 4: Low confidence should express uncertainty
        if confidence < 0.5:
            uncertain_terms = ['uncertain', 'unclear', 'mixed', 'volatile']
            if not any(term in reasoning for term in uncertain_terms):
                score -= 0.2
        
        return max(0.0, score)
    
    def _check_output_consistency(self, decision: Dict[str, Any]) -> float:
        """
        Check for internal consistency in the decision
        """
        action = decision.get('action', '').upper()
        confidence = decision.get('confidence', 0)
        
        # Strong actions should have high confidence
        if action in ["BUY", "SELL"] and confidence < 0.5:
            return 0.6  # Inconsistent
        
        # HOLD can have any confidence
        if action == "HOLD":
            return 0.9
        
        # Normal case
        return 0.85
    
    def _get_historical_performance(self) -> float:
        """
        Track success rate over last N decisions
        """
        if len(self.success_history) == 0:
            return 0.7  # Neutral starting point
        
        recent = self.success_history[-20:]  # Last 20 decisions
        return sum(recent) / len(recent)
    
    def record_outcome(self, success: bool):
        """
        Update historical performance tracker
        """
        self.success_history.append(1.0 if success else 0.0)
        
        # Keep only last 100 records
        if len(self.success_history) > 100:
            self.success_history.pop(0)
    
    def should_proceed(
        self, 
        confidence_metrics: ConfidenceMetrics,
        threshold: float = 0.7
    ) -> tuple[bool, str]:
        """
        Gate: Should we execute this decision?
        
        Returns: (proceed: bool, reason: str)
        """
        if confidence_metrics.overall_confidence >= threshold:
            return True, "Confidence threshold met"
        
        # Provide specific reason for rejection
        if confidence_metrics.schema_validity < 0.8:
            return False, "Schema validation concerns"
        
        if confidence_metrics.semantic_coherence < 0.6:
            return False, "Reasoning does not align with action"
        
        if confidence_metrics.historical_performance < 0.5:
            return False, "Agent has poor historical track record"
        
        return False, f"Overall confidence too low: {confidence_metrics.overall_confidence:.2f}"