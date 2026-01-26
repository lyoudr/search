"""
Transcription Term Processor Service
Extracts medical terms from transcriptions using LLM and stores them as vectors in query_index
"""
from sqlalchemy.orm import Session
from typing import Optional

from app.repositories import transcription_repository
from app.services.medical_term_extractor import MedicalTermExtractor
from app.services.medical_term_vector_store import MedicalTermVectorStore


class TranscriptionTermProcessor:
    """Service for processing transcriptions to extract and store medical terms using LLM"""
    
    def __init__(self, embedding_model: str = "openai"):
        """
        Initialize transcription term processor.
        Stores extracted terms in query_index.
        
        :param embedding_model: Embedding model for vector storage
        """
        self.term_extractor = MedicalTermExtractor()
        self.term_store = MedicalTermVectorStore(embedding_model=embedding_model)
    
    def process_transcription(
        self,
        db: Session,
        transcription_id: int,
        extraction_model: str = "gpt-4o"
    ):
        """
        Extract medical terms from a transcription using LLM and store them as vectors.
        
        :param db: Database session
        :param transcription_id: Transcription ID to process
        :param extraction_model: LLM model to use for term extraction (default: "gpt-4o")
        """
        # Get transcription
        transcription = transcription_repository.get_transcription_by_id(db, transcription_id)
        if not transcription:
            raise ValueError(f"Transcription {transcription_id} not found")
        
        # Extract medical terms using LLM
        print(f"🔍 Extracting medical terms from transcription ID {transcription_id} using LLM...")
        terms = self.term_extractor.extract_terms(
            transcription.text,
            model_name=extraction_model
        )
        
        if not terms:
            print(f"⚠️  No medical terms extracted from transcription ID {transcription_id}")
            return
        
        print(f"✅ Extracted {len(terms)} medical terms: {', '.join(terms[:10])}{'...' if len(terms) > 10 else ''}")
        
        # Store terms as vectors
        self.term_store.store_terms(
            terms=terms,
            transcription_id=transcription_id,
            metadata={
                'audio_file_id': transcription.audio_file_id,
                'engine': transcription.engine
            }
        )
    
    def process_all_transcriptions(
        self,
        db: Session,
        limit: Optional[int] = None,
        extraction_model: str = "gpt-4o"
    ):
        """
        Process all transcriptions to extract and store medical terms.
        
        :param db: Database session
        :param limit: Optional limit on number of transcriptions to process
        :param extraction_model: LLM model to use for term extraction (default: "gpt-4o")
        """
        transcriptions = transcription_repository.get_all_transcriptions(db)
        
        if limit:
            transcriptions = transcriptions[:limit]
        
        print(f"📋 Processing {len(transcriptions)} transcriptions...")
        
        processed = 0
        for transcription in transcriptions:
            try:
                self.process_transcription(
                    db,
                    transcription.id,
                    extraction_model=extraction_model
                )
                processed += 1
            except Exception as e:
                print(f"❌ Failed to process transcription ID {transcription.id}: {e}")
        
        print(f"✅ Processed {processed}/{len(transcriptions)} transcriptions")

