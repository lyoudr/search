"""
MTSamples Retriever Service
Retrieves relevant transcriptions from mtsamples index for RAG-enhanced LLM correction
"""
from typing import List, Optional
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


class MTSamplesRetriever:
    """Service for retrieving medical transcriptions from mtsamples index"""
    
    def __init__(
        self,
        embedding_model: str = "openai",
        query_index_name: str = "query-index",
        mtsamples_index_name: str = "mtsamples"
    ):
        """
        Initialize MTSamples retriever.
        
        :param embedding_model: Embedding model to use
        :param query_index_name: Name of Pinecone index for queries (medical terms)
        :param mtsamples_index_name: Name of Pinecone index for MTSamples
        """
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.query_index = PineconeService(index_name=query_index_name)
        self.mtsamples_index = PineconeService(index_name=mtsamples_index_name)
    
    def retrieve_transcriptions_for_correction(
        self,
        transcription_id: int,
        transcription_text: str,
        top_k_queries: int = 2,
        medical_specialty: Optional[str] = "Hematology - Oncology",
        top_k: int = 5
    ) -> List[str]:
        """
        Retrieve relevant medical transcriptions from mtsamples index.
        
        Process:
        1. Search query_index by transcription_id to get stored medical terms (keywords)
        2. Use these keywords to search mtsamples index (filtered by medical_specialty)
        3. Return relevant keywords from mtsamples
        
        :param transcription_id: Transcription ID to find queries in query_index
        :param transcription_text: Transcription text (used as query vector for query_index search)
        :param top_k_queries: Number of queries (terms) to retrieve from query_index
        :param medical_specialty: Filter by medical specialty (default: "Hematology - Oncology")
        :param top_k: Number of mtsamples transcriptions to retrieve per query
        :return: List of relevant keywords from mtsamples
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
            
            # Step 2: Use these terms to search mtsamples index
            all_keywords = []
            seen_keywords = set()
            
            # Prepare filter for mtsamples
            filter_dict = None
            if medical_specialty:
                filter_dict = {
                    'medical_specialty': {'$eq': str(medical_specialty)}
                }
            
            for term in terms:
                # Create embedding for the term (keyword)
                term_embedding = self.embedding_service.embed_text(term)
                
                # Query mtsamples index
                mtsamples_results = self.mtsamples_index.query(
                    query_vector=term_embedding,
                    top_k=top_k,
                    filter=filter_dict,
                    include_metadata=True
                )
                
                # Extract keywords from mtsamples results
                for result in mtsamples_results:
                    keywords = result.get('metadata', {}).get('keywords', '')
                    if keywords and keywords not in seen_keywords:
                        all_keywords.append(keywords)
                        seen_keywords.add(keywords)
            
            print(f"✅ Retrieved {len(all_keywords)} unique keywords from mtsamples (specialty: {medical_specialty or 'all'})")
            return all_keywords[:top_k * top_k_queries]  # Limit total results
            
        except Exception as e:
            print(f"⚠️  Failed to retrieve transcriptions from mtsamples: {e}")
            return []

