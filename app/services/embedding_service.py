"""
Embedding Service
Handles creating embeddings for text using OpenAI or HuggingFace models
"""
from typing import List

from app.config.settings import get_settings

settings = get_settings()


class EmbeddingService:
    """Service for creating text embeddings"""
    
    def __init__(self, model_name: str = "openai"):
        """
        Initialize embedding service.
        
        :param model_name: Embedding model to use
            - "openai": Use OpenAI text-embedding-3-small (default)
            - "huggingface": Use HuggingFace BAAI/bge-m3
        """
        self.model_name = model_name
        self._openai_client = None
        
        if model_name == "openai":
            from openai import OpenAI
            if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                self._openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            else:
                raise ValueError("OpenAI API key not configured")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Create embedding for a single text.
        
        :param text: Text to embed
        :return: Embedding vector as list of floats
        """
        if self.model_name == "openai":
            return self._embed_openai(text)
        else:
            raise ValueError(f"Unknown embedding model: {self.model_name}")
    
    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Create embeddings for multiple texts.
        
        :param texts: List of texts to embed
        :param batch_size: Batch size for processing
        :return: List of embedding vectors
        """
        if self.model_name == "openai":
            return self._embed_batch_openai(texts, batch_size)
        else:
            raise ValueError(f"Unknown embedding model: {self.model_name}")
    
    def _embed_openai(self, text: str) -> List[float]:
        """Create embedding using OpenAI API"""
        if not self._openai_client:
            raise ValueError("OpenAI client not initialized")
        
        response = self._openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def _embed_batch_openai(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Create embeddings for batch using OpenAI API"""
        if not self._openai_client:
            raise ValueError("OpenAI client not initialized")
        
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self._openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

