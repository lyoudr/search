from typing import Any, Dict

from app.services.medical_documents_services import MedicalTermExtractor


class WhisperAgentTool:
    """Base tool class used by WhisperAgent."""

    name: str = "base_tool"

    def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class TermExtractorTool(WhisperAgentTool):
    name = "term_extractor"

    def __init__(self):
        self.extractor = MedicalTermExtractor()

    def execute(self, text: str, model_name: str) -> Dict[str, Any]:
        terms = self.extractor.extract_terms(text=text, model_name=model_name)
        return {"terms": terms, "count": len(terms)}
