"""
Pinecone Service
Handles storing and querying embeddings in Pinecone vector database
"""
from typing import List, Dict, Optional
import uuid

from pinecone import Pinecone, ServerlessSpec
from app.config.settings import get_settings

settings = get_settings()


class PineconeService:
    """Service for interacting with Pinecone vector database"""
    
    def __init__(self, index_name: str = "medical-documents"):
        """
        Initialize Pinecone service.
        
        :param index_name: Name of the Pinecone index
        """
        self.index_name = index_name
        self.pc = None
        self.index = None
        
        # Initialize Pinecone client
        if hasattr(settings, 'PINECONE_API_KEY') and settings.PINECONE_API_KEY:
            self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        else:
            raise ValueError("Pinecone API key not configured. Set PINECONE_API_KEY in .env")
        
        # Get or create index
        self._ensure_index()
    
    def _ensure_index(self, dimension: int = 1536):
        """
        Ensure the index exists, create if it doesn't.
        
        :param dimension: Dimension of embeddings (1536 for OpenAI text-embedding-3-small)
        """
        # List existing indexes
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            # Create new index
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  # Change to your preferred region
                )
            )
            print(f"✅ Created Pinecone index: {self.index_name}")
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        print(f"✅ Connected to Pinecone index: {self.index_name}")
    
    def upsert_vectors(
        self,
        vectors: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Upsert vectors into Pinecone.
        
        :param vectors: List of embedding vectors
        :param texts: List of text content
        :param metadatas: Optional list of metadata dictionaries
        :param ids: Optional list of IDs (will generate UUIDs if not provided)
        """
        if not self.index:
            raise ValueError("Index not initialized")
        
        if len(vectors) != len(texts):
            raise ValueError("Number of vectors must match number of texts")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{}] * len(vectors)
        
        # Add text to metadata
        for i, metadata in enumerate(metadatas):
            metadata['text'] = texts[i]
        
        # Prepare vectors for upsert
        vectors_to_upsert = []
        for i, (vector, text, metadata, vector_id) in enumerate(zip(vectors, texts, metadatas, ids)):
            vectors_to_upsert.append({
                'id': vector_id,
                'values': vector,
                'metadata': metadata
            })
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        print(f"✅ Upserted {len(vectors_to_upsert)} vectors to Pinecone")
    
    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[Dict]:
        """
        Query Pinecone for similar vectors.
        
        :param query_vector: Query embedding vector
        :param top_k: Number of results to return
        :param filter: Optional metadata filter
        :param include_metadata: Whether to include metadata in results
        :return: List of matching results with scores and metadata
        """
        if not self.index:
            raise ValueError("Index not initialized")
        
        query_response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter,
            include_metadata=include_metadata
        )
        
        results = []
        for match in query_response.matches:
            results.append({
                'id': match.id,
                'score': match.score,
                'metadata': match.metadata if include_metadata else None
            })
        
        return results
    
    def delete_vectors(self, ids: List[str]):
        """
        Delete vectors from Pinecone.
        
        :param ids: List of vector IDs to delete
        """
        if not self.index:
            raise ValueError("Index not initialized")
        
        self.index.delete(ids=ids)
        print(f"✅ Deleted {len(ids)} vectors from Pinecone")
    
    def delete_all(self, namespace: Optional[str] = None):
        """
        Delete all vectors from the index.
        
        :param namespace: Optional namespace to delete from
        """
        if not self.index:
            raise ValueError("Index not initialized")
        
        self.index.delete(delete_all=True, namespace=namespace)
        print(f"✅ Deleted all vectors from Pinecone index")

