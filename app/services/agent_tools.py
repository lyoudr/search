"""
Agent Tools
Tools that the agent can use to correct medical transcriptions
"""

from typing import List, Dict, Any

from app.services.model_manager import model_manager
from app.services.medical_document_retriever import MedicalDocumentRetriever
from app.services.hematology_retriever import HematologyRetriever


class AgentTool:
    """Base class for agent tools"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool and return results"""
        raise NotImplementedError


class DirectLLMTool(AgentTool):
    """Tool for direct LLM correction without RAG"""

    def __init__(self):
        super().__init__(
            name="direct_llm_correction",
            description="Correct transcription using LLM directly without any retrieval. Fast but may miss medical terminology.",
        )

    def execute(
        self, whisper_text: str, model_name: str = "gpt-4", **kwargs
    ) -> Dict[str, Any]:
        """
        Execute direct LLM correction.

        :param whisper_text: Whisper transcribed text
        :param model_name: LLM model to use
        :return: Dictionary with 'corrected_text' and 'method'
        """
        base_prompt = (
            "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
            "1. 不補上標點符號\n"
            "2. 只修正詞彙錯誤\n\n"
            f"原文：{whisper_text}\n"
            f"修正："
        )

        try:
            corrected_text = model_manager.generate_text(
                model_name=model_name,
                prompt=base_prompt,
                max_length=512,
                temperature=0.1,
            )

            return {
                "corrected_text": corrected_text,
                "method": "direct_llm",
                "success": True,
                "tools_used": ["direct_llm"],
            }
        except Exception as e:
            return {
                "corrected_text": None,
                "method": "direct_llm",
                "success": False,
                "error": str(e),
                "tools_used": ["direct_llm"],
            }


class MedicalDocumentRAGTool(AgentTool):
    """Tool for RAG correction using medical documents"""

    def __init__(self):
        super().__init__(
            name="medical_document_rag",
            description="Retrieve relevant medical documents from Pinecone and use them as context for LLM correction. Good for general medical terminology.",
        )
        self.retriever = MedicalDocumentRetriever()

    def execute(
        self,
        whisper_text: str,
        transcription_id: int,
        model_name: str = "gpt-4",
        top_k_queries: int = 3,
        top_k_documents: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute medical document RAG correction.

        :param whisper_text: Whisper transcribed text
        :param transcription_id: Transcription ID to find medical terms
        :param model_name: LLM model to use
        :param top_k_queries: Number of medical terms to retrieve
        :param top_k_documents: Number of documents per term
        :return: Dictionary with 'corrected_text' and 'method'
        """
        base_prompt = (
            "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
            "1. 不補上標點符號\n"
            "2. 只修正詞彙錯誤\n\n"
        )

        try:
            # Retrieve relevant medical documents
            documents = self.retriever.retrieve_documents_for_correction(
                transcription_id=transcription_id,
                transcription_text=whisper_text,
                top_k_queries=top_k_queries,
                top_k_documents=top_k_documents,
            )

            if documents:
                context = "\n\n".join(
                    [f"參考文檔 {i+1}：{doc}" for i, doc in enumerate(documents)]
                )
                prompt = (
                    f"{base_prompt}"
                    f"以下是一些醫療文檔作為參考：\n{context}\n\n"
                    f"原文：{whisper_text}\n"
                    f"修正："
                )
            else:
                # No documents found, fallback to direct LLM
                prompt = f"{base_prompt}原文：{whisper_text}\n修正："

            corrected_text = model_manager.generate_text(
                model_name=model_name, prompt=prompt, max_length=512, temperature=0.1
            )

            return {
                "corrected_text": corrected_text,
                "method": "medical_document_rag",
                "success": True,
                "documents_retrieved": len(documents),
                "tools_used": ["medical_document_rag"],
            }
        except Exception as e:
            return {
                "corrected_text": None,
                "method": "medical_document_rag",
                "success": False,
                "error": str(e),
                "tools_used": ["medical_document_rag"],
            }


class HematologyRAGTool(AgentTool):
    """Tool for RAG correction using hematology dictionary"""

    def __init__(self):
        super().__init__(
            name="hematology_rag",
            description="Retrieve relevant entries from hematology dictionary and use them as context for LLM correction. Best for hematology-specific terminology.",
        )
        self.retriever = HematologyRetriever()

    def execute(
        self,
        whisper_text: str,
        transcription_id: int,
        model_name: str = "gpt-4",
        top_k_queries: int = 2,
        top_k: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute hematology dictionary RAG correction.

        :param whisper_text: Whisper transcribed text
        :param transcription_id: Transcription ID to find medical terms
        :param model_name: LLM model to use
        :param top_k_queries: Number of medical terms to retrieve
        :param top_k: Number of hematology entries per term
        :return: Dictionary with 'corrected_text' and 'method'
        """
        base_prompt = (
            "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
            "1. 不補上標點符號\n"
            "2. 只修正詞彙錯誤\n\n"
        )

        try:
            # Retrieve relevant hematology dictionary entries
            hematology_entries = self.retriever.retrieve_entries_for_correction(
                transcription_id=transcription_id,
                transcription_text=whisper_text,
                top_k_queries=top_k_queries,
                top_k=top_k,
            )

            if hematology_entries:
                context = "\n\n".join(
                    [
                        f"參考範例 {i+1}：{entry}"
                        for i, entry in enumerate(hematology_entries)
                    ]
                )
                prompt = (
                    f"{base_prompt}"
                    f"以下是一些血液學醫學詞典範例作為參考：\n{context}\n\n"
                    f"原文：{whisper_text}\n"
                    f"修正："
                )
            else:
                # No entries found, fallback to direct LLM
                prompt = f"{base_prompt}原文：{whisper_text}\n修正："

            corrected_text = model_manager.generate_text(
                model_name=model_name, prompt=prompt, max_length=512, temperature=0.1
            )

            return {
                "corrected_text": corrected_text,
                "method": "hematology_rag",
                "success": True,
                "entries_retrieved": len(hematology_entries),
                "tools_used": ["hematology_rag"],
            }
        except Exception as e:
            return {
                "corrected_text": None,
                "method": "hematology_rag",
                "success": False,
                "error": str(e),
                "tools_used": ["hematology_rag"],
            }


class CombinedRAGTool(AgentTool):
    """Tool that combines both medical documents and hematology dictionary"""

    def __init__(self):
        super().__init__(
            name="combined_rag",
            description="Combine both medical documents and hematology dictionary for comprehensive correction. Most thorough but slower.",
        )
        self.medical_retriever = MedicalDocumentRetriever()
        self.hematology_retriever = HematologyRetriever()

    def execute(
        self,
        whisper_text: str,
        transcription_id: int,
        model_name: str = "gpt-4",
        top_k_queries: int = 2,
        top_k_documents: int = 3,
        top_k_hematology: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute combined RAG correction using both sources.

        :param whisper_text: Whisper transcribed text
        :param transcription_id: Transcription ID to find medical terms
        :param model_name: LLM model to use
        :param top_k_queries: Number of medical terms to retrieve
        :param top_k_documents: Number of medical documents per term
        :param top_k_hematology: Number of hematology entries per term
        :return: Dictionary with 'corrected_text' and 'method'
        """
        base_prompt = (
            "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
            "1. 不補上標點符號\n"
            "2. 只修正詞彙錯誤\n\n"
        )

        try:
            # Retrieve from both sources
            medical_docs = self.medical_retriever.retrieve_documents_for_correction(
                transcription_id=transcription_id,
                transcription_text=whisper_text,
                top_k_queries=top_k_queries,
                top_k_documents=top_k_documents,
            )

            hematology_entries = (
                self.hematology_retriever.retrieve_entries_for_correction(
                    transcription_id=transcription_id,
                    transcription_text=whisper_text,
                    top_k_queries=top_k_queries,
                    top_k=top_k_hematology,
                )
            )

            # Build context from both sources
            context_parts = []

            if medical_docs:
                context_parts.append("醫療文檔參考：")
                context_parts.extend(
                    [
                        f"  - {doc[:200]}..." if len(doc) > 200 else f"  - {doc}"
                        for doc in medical_docs[:3]
                    ]
                )

            if hematology_entries:
                context_parts.append("\n血液學詞典參考：")
                context_parts.extend(
                    [
                        f"  - {entry[:200]}..." if len(entry) > 200 else f"  - {entry}"
                        for entry in hematology_entries[:3]
                    ]
                )

            if context_parts:
                context = "\n".join(context_parts)
                prompt = (
                    f"{base_prompt}"
                    f"以下是一些醫療參考資料：\n{context}\n\n"
                    f"原文：{whisper_text}\n"
                    f"修正："
                )
            else:
                # No context found, fallback to direct LLM
                prompt = f"{base_prompt}原文：{whisper_text}\n修正："

            corrected_text = model_manager.generate_text(
                model_name=model_name, prompt=prompt, max_length=512, temperature=0.1
            )

            return {
                "corrected_text": corrected_text,
                "method": "combined_rag",
                "success": True,
                "documents_retrieved": len(medical_docs),
                "entries_retrieved": len(hematology_entries),
                "tools_used": ["medical_document_rag", "hematology_rag"],
            }
        except Exception as e:
            return {
                "corrected_text": None,
                "method": "combined_rag",
                "success": False,
                "error": str(e),
                "tools_used": ["medical_document_rag", "hematology_rag"],
            }


def get_available_tools() -> List[AgentTool]:
    """Get list of all available agent tools"""
    return [
        DirectLLMTool(),
        MedicalDocumentRAGTool(),
        HematologyRAGTool(),
        CombinedRAGTool(),
    ]
