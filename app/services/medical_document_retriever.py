"""
Medical Document Retriever Service
Retrieves relevant documents from medical_documents index using queries from query_index
"""
from typing import List, Dict
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


class MedicalDocumentRetriever:
    """Service for retrieving medical documents using queries from query_index"""
    
    def __init__(
        self,
        embedding_model: str = "openai",
        query_index_name: str = "query-index",
        document_index_name: str = "medical-documents"
    ):
        """
        Initialize medical document retriever.
        
        :param embedding_model: Embedding model to use
        :param query_index_name: Name of Pinecone index for queries (medical terms)
        :param document_index_name: Name of Pinecone index for medical documents
        """
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.query_index = PineconeService(index_name=query_index_name)
        self.document_index = PineconeService(index_name=document_index_name)
    
    def retrieve_documents_for_correction(
        self,
        transcription_id: int,
        transcription_text: str,
        top_k_queries: int = 2,
        top_k_documents: int = 3
    ) -> List[str]:
        """
        Retrieve relevant medical documents for LLM correction.
        
        Process:
        1. Search query_index by transcription_id to get stored medical terms (query_text)
        2. Use these query_text (terms) to search medical_documents index (ground truth)
        3. Return relevant document texts
        
        :param transcription_id: Transcription ID to find queries in query_index
        :param transcription_text: Transcription text (used as query vector for query_index search)
        :param top_k_queries: Number of queries (terms) to retrieve from query_index
        :param top_k_documents: Number of documents to retrieve per query from medical-documents
        :return: List of relevant document texts from medical-documents
        """
        try:
            # Step 1: Search query_index by transcription_id to get stored medical terms
            # Use transcription_text embedding as query vector, filter by transcription_id
            transcription_embedding = self.embedding_service.embed_text(transcription_text)
            
            query_results = self.query_index.query(
                query_vector=transcription_embedding,
                top_k=top_k_queries * 2,  # Get more to ensure we have enough after filtering
                filter={
                    'transcription_id': transcription_id,
                    'type': 'medical_term'
                },
                include_metadata=True
            )
            
            # Extract terms (query_text) from results
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
            
            # Step 2: Use these query_text (terms) to search medical-documents (ground truth)
            all_document_texts = []
            seen_texts = set()
            
            for term in terms:
                # Create embedding for the term (query_text)
                term_embedding = self.embedding_service.embed_text(term)
                
                # Query medical_documents index (ground truth)
                doc_results = self.document_index.query(
                    query_vector=term_embedding,
                    top_k=top_k_documents,
                    include_metadata=True
                )
                
                # Extract document texts from medical-documents
                for result in doc_results:
                    text = result.get('metadata', {}).get('text')
                    if text and text not in seen_texts:
                        all_document_texts.append(text)
                        seen_texts.add(text)
            
            print(f"✅ Retrieved {len(all_document_texts)} unique documents from medical-documents (ground truth)")
            return all_document_texts[:top_k_documents * top_k_queries]  # Limit total results
            
        except Exception as e:
            print(f"⚠️  Failed to retrieve documents: {e}")
            return []

