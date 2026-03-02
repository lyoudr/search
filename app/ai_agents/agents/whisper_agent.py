from typing import Any, Dict

from app.ai_agents.tools.shared_vector_tools import (
    ChunkTool,
    EmbeddingTool,
    PineconeUpsertTool,
)
from app.ai_agents.tools.whisper_tools import TermExtractorTool


class WhisperAgent:
    """Agent that orchestrates whisper post-processing via modular tools."""

    def __init__(
        self,
        extraction_model: str = "gpt-5.2",
        embedding_model: str = "openai",
        chunk_size: int = 128,
        chunk_overlap: int = 20,
        pinecone_index_name: str = "query-index",
    ):
        self.extraction_model = extraction_model
        self.term_extractor_tool = TermExtractorTool()
        self.chunk_tool = ChunkTool(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_tool = EmbeddingTool(model_name=embedding_model)
        self.pinecone_tool = PineconeUpsertTool(index_name=pinecone_index_name)

    def process_transcription(
        self,
        transcription_id: int,
        audio_file_id: int,
        engine: str,
        text: str,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "term_count": 0,
            "chunk_count": 0,
            "term_vectors_upserted": 0,
            "chunk_vectors_upserted": 0,
        }

        term_result = self.term_extractor_tool.execute(
            text=text, model_name=self.extraction_model
        )
        terms = term_result["terms"]
        summary["term_count"] = term_result["count"]

        if terms:
            term_embeddings = self.embedding_tool.execute(texts=terms)
            term_metadatas = [
                {
                    "term": term,
                    "type": "medical_term",
                    "transcription_id": int(transcription_id),
                    "audio_file_id": audio_file_id,
                    "engine": engine,
                }
                for term in terms
            ]
            term_upsert = self.pinecone_tool.execute(
                vectors=term_embeddings["vectors"],
                texts=terms,
                metadatas=term_metadatas,
                id_prefix=f"term_{transcription_id}",
            )
            summary["term_vectors_upserted"] = term_upsert["upserted"]

        chunk_result = self.chunk_tool.execute(
            text=text,
            metadata={
                "type": "transcription_chunk",
                "transcription_id": int(transcription_id),
                "audio_file_id": audio_file_id,
                "engine": engine,
            },
        )
        chunks = chunk_result["chunks"]
        summary["chunk_count"] = chunk_result["count"]

        if chunks:
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_metadatas = [chunk["metadata"] for chunk in chunks]
            chunk_embeddings = self.embedding_tool.execute(texts=chunk_texts)
            chunk_upsert = self.pinecone_tool.execute(
                vectors=chunk_embeddings["vectors"],
                texts=chunk_texts,
                metadatas=chunk_metadatas,
                id_prefix=f"chunk_{transcription_id}",
            )
            summary["chunk_vectors_upserted"] = chunk_upsert["upserted"]

        return summary
