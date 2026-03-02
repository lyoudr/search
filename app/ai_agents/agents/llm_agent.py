from typing import Optional

from app.services.correction_core import (
    run_llm_agent_correction,
)
from app.services.hematology_services import HematologyRetriever
from app.services.medical_documents_services import MedicalDocumentRetriever
from app.ai_agents.agents.evaluation_agent import EvaluationAgent
from app.ai_agents.tools.shared_vector_tools import (
    ChunkTool,
    EmbeddingTool,
    PineconeQueryTool,
)


class LLMAgent:
    """LLM correction agent that orchestrates shared tools + retrievers."""

    def __init__(self):
        self.medical_retriever = MedicalDocumentRetriever()
        self.hematology_retriever = HematologyRetriever()
        self.chunk_tool = ChunkTool(chunk_size=256, chunk_overlap=40)
        self.embedding_tool = EmbeddingTool(model_name="openai")
        self.vocab_query_tool = PineconeQueryTool(index_name="hematology-vocab")
        self.evaluation_agent = EvaluationAgent()

        # Lazy import to avoid circular import with app.services.llm
        from app.ai_agents.tools.correction_tools import (
            BatchCorrectWhisperTextTool,
            BatchCorrectWhisperTextWithHematologyTool,
        )

        self.tools = {
            "batch_correct_whisper_text": BatchCorrectWhisperTextTool(),
            "batch_correct_whisper_text_with_hematology": BatchCorrectWhisperTextWithHematologyTool(),
        }

    def correct(
        self,
        whisper_text: str,
        model_name: str,
        strategy: str,
        transcription_id: Optional[int] = None,
        top_k_queries: int = 2,
        top_k_documents: int = 3,
        top_k_hematology: int = 5,
    ) -> dict:
        return run_llm_agent_correction(
            whisper_text=whisper_text,
            model_name=model_name,
            strategy=strategy,
            evaluation_agent=self.evaluation_agent,
            chunk_tool=self.chunk_tool,
            embedding_tool=self.embedding_tool,
            vocab_query_tool=self.vocab_query_tool,
            medical_retriever=self.medical_retriever,
            hematology_retriever=self.hematology_retriever,
            transcription_id=transcription_id,
            top_k_queries=top_k_queries,
            top_k_documents=top_k_documents,
            top_k_hematology=top_k_hematology,
        )

    def execute_tool(self, tool_name: str, **kwargs) -> dict:
        if tool_name not in self.tools:
            raise ValueError(f"Unknown LLM agent tool: {tool_name}")
        return self.tools[tool_name].execute(**kwargs)
