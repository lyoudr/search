"""
Medical Term Vector Store Service
Stores extracted medical terms as vectors in query_index for RAG-enhanced LLM correction
"""
from typing import List, Dict, Optional
import uuid

from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


class MedicalTermVectorStore:
    """Service for storing and retrieving medical terms as vectors in query_index"""
    
    def __init__(
        self,
        embedding_model: str = "openai",
        query_index_name: str = "query-index"
    ):
        """
        Initialize medical term vector store.
        
        :param embedding_model: Embedding model to use
        :param query_index_name: Name of Pinecone index for queries (medical terms)
        """
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.pinecone_service = PineconeService(index_name=query_index_name)
    
    def store_terms(
        self,
        terms: List[str],
        transcription_id: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Store medical terms as vectors in Pinecone.
        
        :param terms: List of medical terms to store
        :param transcription_id: Optional transcription ID this term came from
        :param metadata: Optional additional metadata
        """
        if not terms:
            return
        
        # Create embeddings for all terms
        embeddings = self.embedding_service.embed_batch(terms)
        
        # Prepare metadata
        metadatas = []
        ids = []
        
        for i, term in enumerate(terms):
            term_metadata = {
                'term': term,
                'type': 'medical_term',
                **(metadata or {})
            }
            
            if transcription_id is not None:
                # Ensure transcription_id is stored as int for Pinecone filter matching
                term_metadata['transcription_id'] = int(transcription_id)
            
            metadatas.append(term_metadata)
            ids.append(f"term_{transcription_id}_{i}_{uuid.uuid4().hex[:8]}" if transcription_id else f"term_{uuid.uuid4().hex}")
        
        # Store in Pinecone
        self.pinecone_service.upsert_vectors(
            vectors=embeddings,
            texts=terms,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Stored {len(terms)} medical terms in Pinecone")
    
    def retrieve_similar_terms(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Retrieve similar medical terms from vector store.
        Used for RAG-enhanced LLM correction.
        
        :param query: Query text (could be a misspelled term or context)
        :param top_k: Number of similar terms to retrieve
        :param filter: Optional metadata filter
        :return: List of similar terms with scores
        """
        # Create embedding for query
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search in Pinecone
        results = self.pinecone_service.query(
            query_vector=query_embedding,
            top_k=top_k,
            filter=filter or {'type': 'medical_term'}
        )
        
        return results
    
    def get_relevant_terms_for_correction(
        self,
        transcription_text: str,
        top_k: int = 20
    ) -> List[str]:
        """
        Get relevant medical terms for LLM correction.
        Uses the transcription text to find similar terms in the vector store.
        
        :param transcription_text: Transcription text to correct
        :param top_k: Number of terms to retrieve
        :return: List of relevant medical terms
        """
        # Retrieve similar terms based on the transcription
        results = self.retrieve_similar_terms(
            query=transcription_text,
            top_k=top_k
        )
        
        # Extract unique terms from results
        terms = []
        seen = set()
        for result in results:
            term = result.get('metadata', {}).get('term')
            if term and term not in seen:
                terms.append(term)
                seen.add(term)
        
        return terms

