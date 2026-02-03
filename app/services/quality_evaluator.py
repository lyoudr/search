"""
Quality Evaluator
Evaluates the quality of corrected text to help agent make decisions
"""
from typing import Dict, Any, Optional
import re


class QualityEvaluator:
    """Evaluates quality of transcription corrections"""
    
    def __init__(self):
        # Common medical term patterns (simplified - you can expand this)
        self.medical_patterns = [
            r'[血紅蛋白|白血球|血小板|骨髓|淋巴|細胞]',
            r'[癌症|腫瘤|惡性|良性]',
            r'[診斷|治療|症狀|病徵]',
        ]
    
    def evaluate_correction_quality(
        self,
        original_text: str,
        corrected_text: str,
        method: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a correction.
        
        :param original_text: Original Whisper transcription
        :param corrected_text: Corrected text
        :param method: Method used for correction
        :param metadata: Additional metadata (e.g., documents retrieved)
        :return: Quality score and assessment
        """
        if not corrected_text:
            return {
                "score": 0.0,
                "confidence": "low",
                "issues": ["No corrected text generated"],
                "recommendation": "try_alternative"
            }
        
        # Basic quality checks
        issues = []
        score = 1.0
        
        # Check if text is too short (might be truncated)
        if len(corrected_text) < len(original_text) * 0.5:
            issues.append("Corrected text is significantly shorter than original")
            score -= 0.3
        
        # Check if text is too long (might have hallucinations)
        if len(corrected_text) > len(original_text) * 1.5:
            issues.append("Corrected text is significantly longer than original")
            score -= 0.2
        
        # Check if correction actually changed anything
        if corrected_text.strip() == original_text.strip():
            issues.append("No changes made to original text")
            score -= 0.2
        
        # Check for retrieval quality (if RAG was used)
        if metadata:
            if method in ["medical_document_rag", "combined_rag"]:
                docs_retrieved = metadata.get("documents_retrieved", 0)
                if docs_retrieved == 0:
                    issues.append("No medical documents retrieved")
                    score -= 0.3
                elif docs_retrieved < 2:
                    issues.append("Very few medical documents retrieved")
                    score -= 0.1
            
            if method in ["hematology_rag", "combined_rag"]:
                entries_retrieved = metadata.get("entries_retrieved", 0)
                if entries_retrieved == 0:
                    issues.append("No hematology entries retrieved")
                    score -= 0.3
                elif entries_retrieved < 2:
                    issues.append("Very few hematology entries retrieved")
                    score -= 0.1
        
        # Normalize score to 0-1 range
        score = max(0.0, min(1.0, score))
        
        # Determine confidence level
        if score >= 0.8:
            confidence = "high"
            recommendation = "accept"
        elif score >= 0.6:
            confidence = "medium"
            recommendation = "accept_or_refine"
        else:
            confidence = "low"
            recommendation = "try_alternative"
        
        return {
            "score": score,
            "confidence": confidence,
            "issues": issues,
            "recommendation": recommendation,
            "method": method
        }
    
    def should_try_alternative(
        self,
        quality_assessment: Dict[str, Any],
        methods_tried: list
    ) -> bool:
        """
        Determine if agent should try an alternative method.
        
        :param quality_assessment: Quality assessment from evaluate_correction_quality
        :param methods_tried: List of methods already tried
        :return: True if should try alternative
        """
        if quality_assessment["recommendation"] == "try_alternative":
            return True
        
        if quality_assessment["confidence"] == "low" and len(methods_tried) < 2:
            return True
        
        return False
    
    def suggest_next_method(
        self,
        current_method: str,
        methods_tried: list,
        transcription_text: str
    ) -> Optional[str]:
        """
        Suggest the next method to try based on what's been tried.
        
        :param current_method: Current method that was used
        :param methods_tried: List of methods already tried
        :param transcription_text: Original transcription text
        :return: Suggested next method name, or None if no good options
        """
        available_methods = [
            "direct_llm",
            "medical_document_rag",
            "hematology_rag",
            "combined_rag"
        ]
        
        # If we've tried everything, return None
        if len(methods_tried) >= len(available_methods):
            return None
        
        # Strategy: try more comprehensive methods if simple ones failed
        if "direct_llm" in methods_tried and "medical_document_rag" not in methods_tried:
            return "medical_document_rag"
        
        if "medical_document_rag" in methods_tried and "hematology_rag" not in methods_tried:
            # Check if text seems hematology-related
            if any(re.search(pattern, transcription_text) for pattern in self.medical_patterns):
                return "hematology_rag"
        
        if len(methods_tried) >= 2 and "combined_rag" not in methods_tried:
            return "combined_rag"
        
        # Default: try the next available method
        for method in available_methods:
            if method not in methods_tried:
                return method
        
        return None
