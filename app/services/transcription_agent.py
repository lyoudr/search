"""
Transcription Correction Agent
An intelligent agent that dynamically selects and combines tools to correct medical transcriptions
"""
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from app.services.agent_tools import (
    get_available_tools,
    DirectLLMTool,
    MedicalDocumentRAGTool,
    HematologyRAGTool,
    CombinedRAGTool
)
from app.services.quality_evaluator import QualityEvaluator


class TranscriptionAgent:
    """
    Agent that orchestrates transcription correction using multiple tools.
    
    The agent:
    1. Analyzes the transcription to determine the best strategy
    2. Tries different tools based on quality assessments
    3. Can combine multiple tools for better results
    4. Iterates if quality is low
    """
    
    def __init__(self, max_iterations: int = 3):
        """
        Initialize the agent.
        
        :param max_iterations: Maximum number of correction attempts
        """
        self.tools = {tool.name: tool for tool in get_available_tools()}
        self.quality_evaluator = QualityEvaluator()
        self.max_iterations = max_iterations
    
    def correct_transcription(
        self,
        whisper_text: str,
        transcription_id: int,
        model_name: str = "gpt-4",
        initial_strategy: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Correct a transcription using agent-based approach.
        
        :param whisper_text: Whisper transcribed text
        :param transcription_id: Transcription ID
        :param model_name: LLM model to use
        :param initial_strategy: Initial strategy to try (None for auto-select)
        :param kwargs: Additional parameters for tools
        :return: Dictionary with corrected text and metadata
        """
        methods_tried = []
        best_result = None
        best_score = 0.0
        
        # Determine initial strategy
        if initial_strategy is None:
            initial_strategy = self._select_initial_strategy(whisper_text)
        
        current_method = initial_strategy
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Skip if we've already tried this method
            if current_method in methods_tried:
                # Get next suggested method
                current_method = self.quality_evaluator.suggest_next_method(
                    current_method,
                    methods_tried,
                    whisper_text
                )
                if current_method is None:
                    break
            
            methods_tried.append(current_method)
            
            # Execute the tool
            result = self._execute_tool(
                current_method,
                whisper_text=whisper_text,
                transcription_id=transcription_id,
                model_name=model_name,
                **kwargs
            )
            
            if not result.get("success"):
                # Tool failed, try next method
                continue
            
            # Evaluate quality
            quality = self.quality_evaluator.evaluate_correction_quality(
                original_text=whisper_text,
                corrected_text=result.get("corrected_text"),
                method=current_method,
                metadata=result
            )
            
            result["quality"] = quality
            
            # Track best result
            if quality["score"] > best_score:
                best_score = quality["score"]
                best_result = result
            
            # Check if we should continue
            if quality["recommendation"] == "accept":
                # Good enough, return this result
                return {
                    "corrected_text": result["corrected_text"],
                    "method": current_method,
                    "quality": quality,
                    "methods_tried": methods_tried,
                    "iterations": iteration,
                    "final": True
                }
            
            # Check if we should try alternative
            if not self.quality_evaluator.should_try_alternative(quality, methods_tried):
                # Quality is acceptable, return best result
                break
            
            # Get next method to try
            current_method = self.quality_evaluator.suggest_next_method(
                current_method,
                methods_tried,
                whisper_text
            )
            
            if current_method is None:
                # No more methods to try
                break
        
        # Return best result found
        if best_result:
            return {
                "corrected_text": best_result["corrected_text"],
                "method": best_result["method"],
                "quality": best_result["quality"],
                "methods_tried": methods_tried,
                "iterations": iteration,
                "final": False
            }
        
        # Fallback: try direct LLM if nothing else worked
        if "direct_llm" not in methods_tried:
            result = self._execute_tool(
                "direct_llm",
                whisper_text=whisper_text,
                transcription_id=transcription_id,
                model_name=model_name,
                **kwargs
            )
            if result.get("success"):
                return {
                    "corrected_text": result["corrected_text"],
                    "method": "direct_llm",
                    "quality": {"score": 0.5, "confidence": "low"},
                    "methods_tried": methods_tried + ["direct_llm"],
                    "iterations": iteration + 1,
                    "final": False,
                    "fallback": True
                }
        
        # Last resort: return original text
        return {
            "corrected_text": whisper_text,
            "method": "none",
            "quality": {"score": 0.0, "confidence": "low", "issues": ["All methods failed"]},
            "methods_tried": methods_tried,
            "iterations": iteration,
            "final": False,
            "error": "All correction methods failed"
        }
    
    def _select_initial_strategy(self, whisper_text: str) -> str:
        """
        Select initial strategy based on transcription content.
        
        :param whisper_text: Transcription text
        :return: Strategy name
        """
        # Simple heuristic: check if text seems complex or has medical terms
        # For now, start with medical document RAG as it's a good balance
        # You can enhance this with more sophisticated analysis
        
        text_lower = whisper_text.lower()
        
        # Check for hematology-specific terms
        hematology_keywords = ["血", "細胞", "骨髓", "淋巴", "白血球", "血小板"]
        if any(keyword in text_lower for keyword in hematology_keywords):
            return "hematology_rag"
        
        # Default to medical document RAG
        return "medical_document_rag"
    
    def _execute_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        :param tool_name: Name of the tool to execute
        :param kwargs: Parameters for the tool
        :return: Tool execution result
        """
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "method": tool_name
            }
        
        tool = self.tools[tool_name]
        
        try:
            result = tool.execute(**kwargs)
            result["method"] = tool_name
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "method": tool_name
            }
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available correction strategies"""
        return list(self.tools.keys())
