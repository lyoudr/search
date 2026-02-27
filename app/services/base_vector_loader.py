import uuid
from typing import Any, Dict, List

from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


class BaseVectorLoader:
    """Shared base class for text-to-vector loading workflows."""

    def __init__(self, embedding_model: str, index_name: str):
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.pinecone_service = PineconeService(index_name=index_name)

    def upsert_text_records(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        id_prefix: str,
        item_label: str,
    ) -> Dict[str, Any]:
        if not texts:
            return {"num_entries": 0, "status": "no_entries"}

        print(f"🔢 Creating embeddings for {len(texts)} {item_label}...")
        embeddings = self.embedding_service.embed_batch(texts)
        print(f"✅ Created {len(embeddings)} embeddings")

        ids = [f"{id_prefix}_{uuid.uuid4().hex}" for _ in range(len(texts))]
        print(
            f"💾 Storing {len(embeddings)} vectors in Pinecone index "
            f"'{self.pinecone_service.index_name}'..."
        )
        self.pinecone_service.upsert_vectors(
            vectors=embeddings,
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )
        print(
            f"✅ Successfully loaded {len(embeddings)} entries into "
            f"'{self.pinecone_service.index_name}'"
        )
        return {"num_entries": len(embeddings), "status": "success"}
