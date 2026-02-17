"""
Quality Evaluator
Evaluates the quality of corrected text to help agent make decisions
"""
from typing import Dict, Any, Optional
import re
import json

from app.services.model_manager import model_manager


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
        metadata: Optional[Dict[str, Any]] = None,
        model_name: str = "gpt-5.2"
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of a correction using LLM to detect typos.
        
        :param original_text: Original Whisper transcription (kept for compatibility, not used)
        :param corrected_text: Corrected text to evaluate
        :param method: Method used for correction
        :param metadata: Additional metadata (e.g., documents retrieved)
        :param model_name: LLM model to use for evaluation (default: "gpt-5.2")
        :return: Quality score and assessment
        """
        if not corrected_text:
            return {
                "score": 0.0,
                "confidence": "low",
                "issues": ["No corrected text generated"],
                "recommendation": "try_alternative"
            }
        
        issues = []
        score = 1.0
        
        # Use LLM to evaluate correction quality by detecting typos in corrected_text only
        llm_evaluation = self._evaluate_with_llm(
            corrected_text=corrected_text,
            model_name=model_name
        )
        
        # Extract LLM evaluation results
        chinese_typos = llm_evaluation.get("chinese_typos", [])
        english_typos = llm_evaluation.get("english_typos", [])
        overall_quality = llm_evaluation.get("overall_quality", "unknown")
        llm_score = llm_evaluation.get("score", 0.5)
        
        # Calculate score based on LLM evaluation
        # Start with LLM's score (0.0 to 1.0)
        score = llm_score
        
        # Add penalties for detected typos
        if chinese_typos:
            penalty = min(0.3, len(chinese_typos) * 0.1)
            score -= penalty
            issues.append(f"Detected {len(chinese_typos)} Chinese typo(s): {', '.join(chinese_typos[:3])}")
        
        if english_typos:
            penalty = min(0.3, len(english_typos) * 0.1)
            score -= penalty
            issues.append(f"Detected {len(english_typos)} English medical term typo(s): {', '.join(english_typos[:3])}")
        
        # Check for retrieval quality (if RAG was used)
        if metadata:
            if method in ["medical_document_rag", "combined_rag"]:
                docs_retrieved = metadata.get("documents_retrieved", 0)
                if docs_retrieved == 0:
                    issues.append("No medical documents retrieved")
                    score -= 0.2
                elif docs_retrieved < 2:
                    issues.append("Very few medical documents retrieved")
                    score -= 0.1
            
            if method in ["hematology_rag", "combined_rag"]:
                entries_retrieved = metadata.get("entries_retrieved", 0)
                if entries_retrieved == 0:
                    issues.append("No hematology entries retrieved")
                    score -= 0.2
                elif entries_retrieved < 2:
                    issues.append("Very few hematology entries retrieved")
                    score -= 0.1
            
            if method in ["hematology_vocabulary", "combined_rag"]:
                vocab_terms_retrieved = metadata.get("vocab_terms_retrieved", 0)
                if vocab_terms_retrieved == 0:
                    issues.append("No hematology vocabulary terms retrieved")
                    score -= 0.2
                elif vocab_terms_retrieved < 2:
                    issues.append("Very few hematology vocabulary terms retrieved")
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
            "score": round(score, 2),
            "confidence": confidence,
            "issues": issues,
            "recommendation": recommendation,
            "method": method,
            "llm_evaluation": {
                "chinese_typos": chinese_typos,
                "english_typos": english_typos,
                "overall_quality": overall_quality
            }
        }
    
    def _evaluate_with_llm(
        self,
        corrected_text: str,
        model_name: str = "gpt-5.2"
    ) -> Dict[str, Any]:
        """
        Use LLM to evaluate correction quality by detecting typos in corrected_text.
        
        :param corrected_text: Corrected text to evaluate
        :param model_name: LLM model to use
        :return: Dictionary with evaluation results
        """
        prompt = f"""你是一位醫療文本質量評估專家。請檢查以下文本中是否有錯字：

文本內容：
{corrected_text}

請仔細檢查：
1. **中文錯字**：檢查文本中的中文是否有錯字（包括同音字、形近字等錯誤）
2. **英文專業術語錯字**：檢查文本中的英文醫療專業術語是否有拼寫錯誤

請以 JSON 格式回答，包含以下欄位：
{{
    "chinese_typos": ["錯字1", "錯字2", ...],  // 如果沒有錯字，返回空陣列 []
    "english_typos": ["錯誤的英文術語1", "錯誤的英文術語2", ...],  // 如果沒有錯字，返回空陣列 []
    "overall_quality": "excellent|good|fair|poor",  // 整體質量評估（excellent=無錯字，good=極少錯字，fair=有一些錯字，poor=很多錯字）
    "score": 0.0-1.0,  // 質量分數，1.0 表示完美無錯，0.0 表示有很多錯誤
    "explanation": "簡短說明評估結果"
}}

只返回 JSON，不要其他文字。"""

        try:
            response = model_manager.generate_text(
                model_name=model_name,
                prompt=prompt,
                max_length=512,
                temperature=0.1
            )
            
            # Try to parse JSON from response
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            try:
                evaluation = json.loads(response)
                return {
                    "chinese_typos": evaluation.get("chinese_typos", []),
                    "english_typos": evaluation.get("english_typos", []),
                    "overall_quality": evaluation.get("overall_quality", "unknown"),
                    "score": float(evaluation.get("score", 0.5)),
                    "explanation": evaluation.get("explanation", "")
                }
            except json.JSONDecodeError:
                # Fallback: try to extract score from text
                print(f"⚠️  Failed to parse LLM JSON response: {response}")
                # Default to medium score if parsing fails
                return {
                    "chinese_typos": [],
                    "english_typos": [],
                    "overall_quality": "unknown",
                    "score": 0.5,
                    "explanation": "LLM evaluation parsing failed"
                }
        
        except Exception as e:
            print(f"⚠️  LLM evaluation failed: {e}")
            # Fallback to default score
            return {
                "chinese_typos": [],
                "english_typos": [],
                "overall_quality": "unknown",
                "score": 0.5,
                "explanation": f"LLM evaluation error: {str(e)}"
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
            "hematology_vocabulary",
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
