"""
Hematology Retriever Service
Retrieves relevant entries from hematology dictionary index for RAG-enhanced LLM correction
"""
from typing import List, Optional
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


class HematologyRetriever:
    """Service for retrieving medical entries from hematology dictionary index"""
    
    def __init__(
        self,
        embedding_model: str = "openai",
        query_index_name: str = "query-index",
        hematology_index_name: str = "hematology"
    ):
        """
        Initialize Hematology retriever.
        
        :param embedding_model: Embedding model to use
        :param query_index_name: Name of Pinecone index for queries (medical terms)
        :param hematology_index_name: Name of Pinecone index for hematology dictionary
        """
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.query_index = PineconeService(index_name=query_index_name)
        self.hematology_index = PineconeService(index_name=hematology_index_name)
    
    def retrieve_entries_for_correction(
        self,
        transcription_id: int,
        transcription_text: str,
        top_k_queries: int = 2,
        top_k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant hematology dictionary entries.
        
        Process:
        1. Search query_index by transcription_id to get stored medical terms (keywords)
        2. Use these keywords to search hematology index
        3. Return relevant dictionary entries
        
        :param transcription_id: Transcription ID to find queries in query_index
        :param transcription_text: Transcription text (used as query vector for query_index search)
        :param top_k_queries: Number of queries (terms) to retrieve from query_index
        :param top_k: Number of hematology entries to retrieve per query
        :return: List of relevant dictionary entries
        """
        try:
            # Step 1: Search query_index by transcription_id to get stored medical terms
            transcription_embedding = self.embedding_service.embed_text(transcription_text)
            
            query_results = self.query_index.query(
                query_vector=transcription_embedding,
                top_k=top_k_queries * 2,  # Get more to ensure we have enough after filtering
                filter={
                    'transcription_id': {'$eq': int(transcription_id)},
                    'type': {'$eq': 'medical_term'}
                },
                include_metadata=True
            )
            print(f"🔍 Query results for transcription_id={transcription_id}: {len(query_results)} results")
            
            # Extract terms (keywords) from results
            terms = []
            for result in query_results:
                term = result.get('metadata', {}).get('term')
                if term and term not in terms:
                    terms.append(term)
                    if len(terms) >= top_k_queries:
                        break
            
            if not terms:
                print(f"⚠️  No medical terms found in query_index for transcription {transcription_id}")
                return []
            
            print(f"🔍 Found {len(terms)} queries from query_index (transcription_id={transcription_id}): {', '.join(terms)}")
            
            # Step 2: Use these terms to search hematology index
            all_entries = []
            seen_entries = set()
            
            for term in terms:
                # Create embedding for the term (keyword)
                term_embedding = self.embedding_service.embed_text(term)
                
                # Query hematology index
                hematology_results = self.hematology_index.query(
                    query_vector=term_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                
                # Extract text entries from hematology results
                for result in hematology_results:
                    text = result.get('metadata', {}).get('text', '')
                    if not text:
                        # Fallback: use term and standard_term if text not available
                        term_val = result.get('metadata', {}).get('term', '')
                        standard_term = result.get('metadata', {}).get('standard_term', '')
                        if standard_term:
                            text = f"{term_val} ({standard_term})"
                        else:
                            text = term_val
                    
                    if text and text not in seen_entries:
                        all_entries.append(text)
                        seen_entries.add(text)
            
            print(f"✅ Retrieved {len(all_entries)} unique entries from hematology dictionary")
            return all_entries[:top_k * top_k_queries]  # Limit total results
            
        except Exception as e:
            print(f"⚠️  Failed to retrieve entries from hematology dictionary: {e}")
            return []

