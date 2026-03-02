import uuid
from typing import Any, Dict, List, Optional

from app.services.document_services import TextChunker
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


class SharedTool:
    """Base class for reusable tools across agents."""

    name: str = "shared_tool"

    def execute(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError


class ChunkTool(SharedTool):
    name = "chunk"

    def __init__(self, chunk_size: int = 128, chunk_overlap: int = 20):
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def execute(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        chunks = self.chunker.chunk_text(text=text, metadata=metadata or {})
        return {"chunks": chunks, "count": len(chunks)}


class EmbeddingTool(SharedTool):
    name = "embedding"

    def __init__(self, model_name: str = "openai"):
        self.embedding_service = EmbeddingService(model_name=model_name)

    def execute(self, texts: List[str]) -> Dict[str, Any]:
        vectors = self.embedding_service.embed_batch(texts) if texts else []
        return {"vectors": vectors, "count": len(vectors)}

    def embed_text(self, text: str) -> List[float]:
        return self.embedding_service.embed_text(text)


class PineconeUpsertTool(SharedTool):
    name = "pinecone_upsert"

    def __init__(self, index_name: str):
        self.pinecone_service = PineconeService(index_name=index_name)

    def execute(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        id_prefix: str,
    ) -> Dict[str, Any]:
        if not texts:
            return {"upserted": 0, "index_name": self.pinecone_service.index_name}
        ids = [f"{id_prefix}_{uuid.uuid4().hex}" for _ in range(len(texts))]
        self.pinecone_service.upsert_vectors(
            vectors=vectors,
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )
        return {"upserted": len(texts), "index_name": self.pinecone_service.index_name}


class PineconeQueryTool(SharedTool):
    name = "pinecone_query"

    def __init__(self, index_name: str):
        self.pinecone_service = PineconeService(index_name=index_name)

    def execute(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> Dict[str, Any]:
        matches = self.pinecone_service.query(
            query_vector=query_vector,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata,
        )
        return {"matches": matches, "count": len(matches)}
