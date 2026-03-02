"""
Agent Tools
Tools that the agent can use to correct medical transcriptions
"""

from typing import List, Dict, Any

from app.services.medical_documents_services import MedicalDocumentRetriever
from app.services.hematology_services import HematologyRetriever
from app.ai_agents.tools.shared_vector_tools import EmbeddingTool, PineconeQueryTool
from app.services.correction_core import (
    build_correction_prompt,
    build_numbered_lines,
    extract_unique_metadata_terms,
    generate_correction_text,
)


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
        self, whisper_text: str, model_name: str = "gpt-5.2", **kwargs
    ) -> Dict[str, Any]:
        """
        Execute direct LLM correction.

        :param whisper_text: Whisper transcribed text
        :param model_name: LLM model to use
        :return: Dictionary with 'corrected_text' and 'method'
        """
        prompt = build_correction_prompt(whisper_text=whisper_text)

        try:
            corrected_text = generate_correction_text(
                model_name=model_name, prompt=prompt
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
        model_name: str = "gpt-5.2",
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
        try:
            # Retrieve relevant medical documents
            documents = self.retriever.retrieve_documents_for_correction(
                transcription_id=transcription_id,
                transcription_text=whisper_text,
                top_k_queries=top_k_queries,
                top_k_documents=top_k_documents,
            )

            if documents:
                prompt = build_correction_prompt(
                    whisper_text=whisper_text,
                    context_header="以下是一些醫療文檔作為參考：",
                    context_lines=build_numbered_lines(documents, "參考文檔"),
                )
            else:
                # No documents found, fallback to direct LLM
                prompt = build_correction_prompt(whisper_text=whisper_text)

            corrected_text = generate_correction_text(
                model_name=model_name, prompt=prompt
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
        model_name: str = "gpt-5.2",
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
        try:
            # Retrieve relevant hematology dictionary entries
            hematology_entries = self.retriever.retrieve_entries_for_correction(
                transcription_id=transcription_id,
                transcription_text=whisper_text,
                top_k_queries=top_k_queries,
                top_k=top_k,
            )

            if hematology_entries:
                prompt = build_correction_prompt(
                    whisper_text=whisper_text,
                    context_header="以下是一些血液學醫學詞典範例作為參考：",
                    context_lines=build_numbered_lines(hematology_entries, "參考範例"),
                )
            else:
                # No entries found, fallback to direct LLM
                prompt = build_correction_prompt(whisper_text=whisper_text)

            corrected_text = generate_correction_text(
                model_name=model_name, prompt=prompt
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


class HematologyVocabularyTool(AgentTool):
    """Tool for vocabulary correction using hematology-vocab index"""

    def __init__(self):
        super().__init__(
            name="hematology_vocabulary",
            description="Retrieve relevant vocabulary terms from hematology-vocab index and use them as context for LLM correction. Best for vocabulary-specific corrections.",
        )
        self.embedding_tool = EmbeddingTool(model_name="openai")
        self.vocab_query_tool = PineconeQueryTool(index_name="hematology-vocab")

    def execute(
        self,
        whisper_text: str,
        transcription_id: int,
        model_name: str = "gpt-5.2",
        top_k: int = 10,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute vocabulary correction using hematology-vocab index.

        :param whisper_text: Whisper transcribed text
        :param transcription_id: Transcription ID (for consistency with other tools)
        :param model_name: LLM model to use
        :param top_k: Number of vocabulary terms to retrieve from hematology-vocab index
        :return: Dictionary with 'corrected_text' and 'method'
        """
        try:
            # Create embedding for the transcription text
            query_embedding = self.embedding_tool.embed_text(whisper_text)

            # Query hematology-vocab index directly
            vocab_results = self.vocab_query_tool.execute(
                query_vector=query_embedding, top_k=top_k, include_metadata=True
            )["matches"]

            # Extract vocabulary terms from results
            vocab_terms = extract_unique_metadata_terms(vocab_results)

            if vocab_terms:
                prompt = build_correction_prompt(
                    whisper_text=whisper_text,
                    context_header="以下是一些血液學醫學詞彙作為參考：",
                    context_lines=build_numbered_lines(vocab_terms, "參考詞彙"),
                )
            else:
                # No vocabulary terms found, fallback to direct LLM
                prompt = build_correction_prompt(whisper_text=whisper_text)

            corrected_text = generate_correction_text(
                model_name=model_name, prompt=prompt
            )

            return {
                "corrected_text": corrected_text,
                "method": "hematology_vocabulary",
                "success": True,
                "vocab_terms_retrieved": len(vocab_terms),
                "tools_used": ["hematology_vocabulary"],
            }
        except Exception as e:
            return {
                "corrected_text": None,
                "method": "hematology_vocabulary",
                "success": False,
                "error": str(e),
                "tools_used": ["hematology_vocabulary"],
            }


class CombinedRAGTool(AgentTool):
    """Tool that combines medical documents, hematology dictionary, and hematology vocabulary"""

    def __init__(self):
        super().__init__(
            name="combined_rag",
            description="Combine medical documents, hematology dictionary, and hematology vocabulary for comprehensive correction. Most thorough but slower.",
        )
        self.medical_retriever = MedicalDocumentRetriever()
        self.hematology_retriever = HematologyRetriever()
        self.embedding_tool = EmbeddingTool(model_name="openai")
        self.vocab_query_tool = PineconeQueryTool(index_name="hematology-vocab")

    def execute(
        self,
        whisper_text: str,
        transcription_id: int,
        model_name: str = "gpt-5.2",
        top_k_queries: int = 2,
        top_k_documents: int = 3,
        top_k_hematology: int = 3,
        top_k_vocab: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Execute combined RAG correction using all three sources.

        :param whisper_text: Whisper transcribed text
        :param transcription_id: Transcription ID to find medical terms
        :param model_name: LLM model to use
        :param top_k_queries: Number of medical terms to retrieve
        :param top_k_documents: Number of medical documents per term
        :param top_k_hematology: Number of hematology entries per term
        :param top_k_vocab: Number of hematology vocabulary terms to retrieve
        :return: Dictionary with 'corrected_text' and 'method'
        """
        try:
            # Retrieve from all three sources
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

            # Retrieve from hematology-vocab index
            query_embedding = self.embedding_tool.embed_text(whisper_text)
            vocab_results = self.vocab_query_tool.execute(
                query_vector=query_embedding, top_k=top_k_vocab, include_metadata=True
            )["matches"]

            # Extract vocabulary terms from results
            vocab_terms = extract_unique_metadata_terms(vocab_results)

            # Build context from all three sources
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

            if vocab_terms:
                context_parts.append("\n血液學詞彙參考：")
                context_parts.extend([f"  - {term}" for term in vocab_terms[:5]])

            if context_parts:
                context = "\n".join(context_parts)
                prompt = build_correction_prompt(
                    whisper_text=whisper_text,
                    context_header="以下是一些醫療參考資料：",
                    context_lines=[context],
                )
            else:
                # No context found, fallback to direct LLM
                prompt = build_correction_prompt(whisper_text=whisper_text)

            corrected_text = generate_correction_text(
                model_name=model_name, prompt=prompt
            )

            return {
                "corrected_text": corrected_text,
                "method": "combined_rag",
                "success": True,
                "documents_retrieved": len(medical_docs),
                "entries_retrieved": len(hematology_entries),
                "vocab_terms_retrieved": len(vocab_terms),
                "tools_used": [
                    "medical_document_rag",
                    "hematology_rag",
                    "hematology_vocabulary",
                ],
            }
        except Exception as e:
            return {
                "corrected_text": None,
                "method": "combined_rag",
                "success": False,
                "error": str(e),
                "tools_used": [
                    "medical_document_rag",
                    "hematology_rag",
                    "hematology_vocabulary",
                ],
            }


def get_available_tools() -> List[AgentTool]:
    """Get list of all available agent tools"""
    return [
        DirectLLMTool(),
        MedicalDocumentRAGTool(),
        HematologyRAGTool(),
        HematologyVocabularyTool(),
        CombinedRAGTool(),
    ]
