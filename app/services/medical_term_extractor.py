"""
Medical Term Extractor Service
Extracts medical terminology from transcriptions using LLM for vector storage
"""
from typing import List, Set
import re

from app.services.model_manager import model_manager


class MedicalTermExtractor:
    """Service for extracting medical terms from transcriptions using LLM"""
    
    def extract_terms(self, text: str, model_name: str = "gpt-5.2") -> List[str]:
        """
        Extract medical terms from text using LLM.
        
        :param text: Transcription text
        :param model_name: LLM model to use for extraction (default: "gpt-5.2")
        :return: List of extracted medical terms
        """
        prompt = (
            "你是一位醫療術語專家。請從以下醫療轉錄文本中提取所有醫療專有術語。\n"
            "請只提取專業的醫療術語、疾病名稱、藥物名稱、檢查項目、治療方法等。\n"
            "不要提取一般詞彙或非醫療相關的詞。\n"
            "請將提取的術語以逗號分隔，每行一個術語，不要編號，不要其他說明。\n\n"
            f"轉錄文本：{text}\n\n"
            "提取的醫療術語："
        )
        
        try:
            response = model_manager.generate_text(
                model_name=model_name,
                prompt=prompt,
                max_length=512,
                temperature=0.1  # Low temperature for more consistent extraction
            )
            
            # Parse response to extract terms
            terms = []
            for line in response.strip().split('\n'):
                line = line.strip()
                # Remove numbering if present (e.g., "1. 術語" -> "術語")
                line = re.sub(r'^\d+[\.、．]\s*', '', line)
                # Split by comma if multiple terms on one line
                for term in line.split(','):
                    term = term.strip()
                    if term and len(term) > 1:  # Filter out single characters
                        terms.append(term)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term in terms:
                if term not in seen:
                    seen.add(term)
                    unique_terms.append(term)
            
            return unique_terms
        
        except Exception as e:
            raise ValueError(f"Failed to extract medical terms with LLM: {e}")
    
    def extract_unique_terms(self, texts: List[str], model_name: str = "gpt-5.2") -> Set[str]:
        """
        Extract unique medical terms from multiple texts.
        
        :param texts: List of transcription texts
        :param model_name: LLM model to use
        :return: Set of unique medical terms
        """
        all_terms = set()
        
        for text in texts:
            terms = self.extract_terms(text, model_name)
            all_terms.update(terms)
        
        return all_terms

