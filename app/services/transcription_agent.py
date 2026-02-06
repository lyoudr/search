"""
Transcription Correction Agent
An intelligent agent that dynamically selects and combines tools to correct medical transcriptions
"""

import asyncio
from typing import Dict, Any, Optional, List

from app.services.agent_tools import (
    get_available_tools,
)
from app.services.quality_evaluator import QualityEvaluator


class TranscriptionAgent:
    """
    Agent that orchestrates transcription correction using multiple tools.
    
    The agent:
    1. Analyzes the transcription to determine the best strategy
    2. Runs multiple tools (potentially in parallel) based on quality assessments
    3. Chooses the best result using the QualityEvaluator
    """
    
    def __init__(self, max_iterations: int = 1, max_parallel_tools: int = 4):
        """
        Initialize the agent.
        
        :param max_iterations: Maximum number of correction rounds (kept for compatibility, default 1)
        :param max_parallel_tools: Maximum number of tools to run in parallel
        """
        # Tools are registered by name
        self.tools = {tool.name: tool for tool in get_available_tools()}
        self.quality_evaluator = QualityEvaluator()
        self.max_iterations = max_iterations
        self.max_parallel_tools = max_parallel_tools
    
    # -------------------------------------------------------------------------
    # Public API (synchronous wrapper)
    # -------------------------------------------------------------------------
    def correct_transcription(
        self,
        whisper_text: str,
        transcription_id: int,
        model_name: str = "gpt-4",
        initial_strategy: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for correct_transcription_async.
        
        This keeps the existing API compatible with synchronous callers
        (e.g., FastAPI routes that are defined as normal `def` functions).
        """
        return asyncio.run(
            self.correct_transcription_async(
                whisper_text=whisper_text,
                transcription_id=transcription_id,
                model_name=model_name,
                initial_strategy=initial_strategy,
                **kwargs,
            )
        )
    
    # -------------------------------------------------------------------------
    # Public API (async / concurrent version)
    # -------------------------------------------------------------------------
    async def correct_transcription_async(
        self,
        whisper_text: str,
        transcription_id: int,
        model_name: str = "gpt-4",
        initial_strategy: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Correct a transcription using an async, agent-based approach.
        
        Strategy:
        1. Decide an initial strategy based on the transcription content
        2. Select a set of candidate tools (up to max_parallel_tools), prioritizing the initial strategy
        3. Run all selected tools concurrently (each in a separate thread, since tool.execute() is sync)
        4. Use QualityEvaluator to score each result and pick the best one
        5. Optionally, could iterate multiple rounds, but by default we do a single parallel round
        """
        methods_tried: List[str] = []
        best_result: Optional[Dict[str, Any]] = None
        best_score: float = 0.0
        iteration = 0
        
        # Determine initial strategy
        if initial_strategy is None:
            initial_strategy = self._select_initial_strategy(whisper_text)
        
        # Available tool names (as registered in self.tools)
        available_methods: List[str] = list(self.tools.keys())
        
        # Simple ordering: put initial_strategy first (if it exists), then the rest
        ordered_methods: List[str] = []
        if initial_strategy in available_methods:
            ordered_methods.append(initial_strategy)
        ordered_methods.extend(
            [m for m in available_methods if m != initial_strategy]
        )
        
        # We currently do a single parallel round (iteration = 1),
        # but we keep the variable in case we want to expand later.
        while iteration < self.max_iterations:
            iteration += 1
            
            # Select a batch of methods to run in parallel for this iteration
            batch_methods = [
                m for m in ordered_methods if m not in methods_tried
            ][: self.max_parallel_tools]
            
            if not batch_methods:
                break
            
            methods_tried.extend(batch_methods)
            
            # Run all tools in this batch concurrently
            tasks = [
                self._execute_tool_async(
                    method_name,
                    whisper_text=whisper_text,
                    transcription_id=transcription_id,
                    model_name=model_name,
                    **kwargs,
                )
                for method_name in batch_methods
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Evaluate quality for each successful result
            for method_name, result in zip(batch_methods, results):
                if isinstance(result, Exception):
                    # Capture execution error as a failed tool
                    continue
                
                if not result.get("success"):
                    # Tool failed, skip
                    continue
                
                corrected_text = result.get("corrected_text")
                if not corrected_text:
                    continue
                
                quality = self.quality_evaluator.evaluate_correction_quality(
                    original_text=whisper_text,
                    corrected_text=corrected_text,
                    method=method_name,
                    metadata=result,
                )
                
                result["quality"] = quality
                result["method"] = method_name
                
                # Track best result so far
                if quality["score"] > best_score:
                    best_score = quality["score"]
                    best_result = result
                
                # If any result is clearly acceptable, we can stop early
                if quality["recommendation"] == "accept":
                    return {
                        "corrected_text": corrected_text,
                        "method": method_name,
                        "quality": quality,
                        "methods_tried": methods_tried,
                        "iterations": iteration,
                        "final": True,
                    }
            
            # If we've reached here without an 'accept', decide whether to continue
            if best_result is not None:
                # If best score is reasonably good, we can stop
                if best_result["quality"]["score"] >= 0.8:
                    return {
                        "corrected_text": best_result["corrected_text"],
                        "method": best_result["method"],
                        "quality": best_result["quality"],
                        "methods_tried": methods_tried,
                        "iterations": iteration,
                        "final": True,
                    }
            
            # If max_iterations == 1 (default) or we have tried all methods, break
            remaining = [m for m in available_methods if m not in methods_tried]
            if not remaining:
                break
        
        # Return best result found if any
        if best_result:
            return {
                "corrected_text": best_result["corrected_text"],
                "method": best_result["method"],
                "quality": best_result["quality"],
                "methods_tried": methods_tried,
                "iterations": iteration,
                "final": False,
            }
        
        # Fallback: try a simple direct LLM correction if such a tool exists
        fallback_method_name = None
        # Prefer a tool whose name contains 'direct_llm'
        for name in available_methods:
            if "direct_llm" in name:
                fallback_method_name = name
                break
        
        if fallback_method_name:
            fallback_result = await self._execute_tool_async(
                fallback_method_name,
                whisper_text=whisper_text,
                transcription_id=transcription_id,
                model_name=model_name,
                **kwargs,
            )
            if fallback_result.get("success") and fallback_result.get("corrected_text"):
                return {
                    "corrected_text": fallback_result["corrected_text"],
                    "method": fallback_method_name,
                    "quality": {"score": 0.5, "confidence": "low"},
                    "methods_tried": methods_tried + [fallback_method_name],
                    "iterations": iteration + 1,
                    "final": False,
                    "fallback": True,
                }
        
        # Last resort: return original text
        return {
            "corrected_text": whisper_text,
            "method": "none",
            "quality": {
                "score": 0.0,
                "confidence": "low",
                "issues": ["All methods failed"],
            },
            "methods_tried": methods_tried,
            "iterations": iteration,
            "final": False,
            "error": "All correction methods failed",
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
            # Prefer hematology_vocabulary for vocabulary-focused corrections
            # (it directly searches the hematology-vocab index)
            # Fall back to hematology_rag if vocabulary tool is not available
            if "hematology_vocabulary" in self.tools:
                return "hematology_vocabulary"
            return "hematology_rag"

        # Default to medical document RAG
        return "medical_document_rag"

    def _execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
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
                "method": tool_name,
            }

        tool = self.tools[tool_name]
        
        try:
            result = tool.execute(**kwargs)
            result["method"] = tool_name
            return result
        except Exception as e:
            return {"success": False, "error": str(e), "method": tool_name}

    async def _execute_tool_async(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Async wrapper around _execute_tool.
        
        Runs the synchronous tool.execute(...) in a thread so that multiple tools
        can be executed concurrently without blocking the event loop.
        """
        return await asyncio.to_thread(self._execute_tool, tool_name, **kwargs)

    def get_available_strategies(self) -> List[str]:
        """Get list of available correction strategies"""
        return list(self.tools.keys())
