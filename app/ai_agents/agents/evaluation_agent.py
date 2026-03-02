from typing import Optional

from app.services.quality_evaluator import QualityEvaluator


class EvaluationAgent:
    """Evaluate correction output quality for LLM agent."""

    def __init__(self):
        self.evaluator = QualityEvaluator()

    def evaluate(
        self,
        original_text: str,
        corrected_text: str,
        method: str,
        metadata: Optional[dict] = None,
        model_name: str = "gpt-5.2",
    ) -> dict:
        return self.evaluator.evaluate_correction_quality(
            original_text=original_text,
            corrected_text=corrected_text,
            method=method,
            metadata=metadata or {},
            model_name=model_name,
        )
