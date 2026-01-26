from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Optional

from app.config.settings import get_settings
from app.models import get_db
from app.schemas.req_res.audio import (
    ParseRequest, 
    ParseResponse,
    AnalyzeRequest,
    WordErrorRateResponse,
    LLMCorrectResponse
)
from app.services.parse_audio import convert_to_wav
from app.services.split_audio import split_audio as split_audio_service
from app.services.speech2text import whisper_to_text
from app.services.llm import batch_correct_whisper_text
from app.services.import_audio_files import import_audio_files_from_splited
from app.repositories import (
    transcription_repository,
    llm_output_repository,
    evaluation_repository
)
from app.services.wer import wer

router = APIRouter(tags=["audio"], prefix="/audio")
settings = get_settings()

@router.post(
    "/parse",
    summary="Parse audio files to .wav",
)
def parse_audio(
    payload: ParseRequest
):
    convert_to_wav(
        f"{settings.SOURCE_DIR}/{payload.input_dir}", 
        f"{settings.SOURCE_DIR}/{payload.output_dir}"
    )
    return ParseResponse(status="Success parse audio file to .wav")

@router.post(
    "/split",
    summary="Split audio files into chunks",
)
def split_audio(
    payload: ParseRequest
):
    split_audio_service(
        f"{settings.SOURCE_DIR}/{payload.input_dir}", 
        f"{settings.SOURCE_DIR}/{payload.output_dir}"
    )
    return ParseResponse(status="Success split audio file into chunks")


@router.post(
    "/whisper/analyze",
    summary="Whisper analyze audio files and extract medical terms",
)
def whisper_analyze(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    extraction_model: str = "gpt-4o"
):
    """
    Analyze audio files with Whisper.
    Automatically extracts medical terms using LLM and stores them in query_index.
    
    :param payload: Request payload with input_dir
    :param db: Database session
    :param extraction_model: LLM model to use for term extraction (default: "gpt-4o")
    """
    whisper_to_text(
        db,
        f"{settings.SOURCE_DIR}/{payload.input_dir}",
        extraction_model=extraction_model
    )
    return ParseResponse(status="Success analyze audio file with Whisper and extract medical terms")


@router.post(
    "/whisper_wer",
    summary="Calculate WER for Whisper transcriptions",
)
def calculate_whisper_wer(
    db: Session = Depends(get_db)
):
    """
    Calculate WER for transcriptions that have ground truth.
    Note: This requires evaluations to be created with ground_truth.
    """
    # Get all transcriptions
    transcriptions = transcription_repository.get_all_transcriptions(db)
    
    updated_count = 0
    for transcription in transcriptions:
        # Get LLM outputs for this transcription
        llm_outputs = llm_output_repository.get_llm_outputs_by_transcription(db, transcription.id)
        
        for llm_output in llm_outputs:
            # Get or create evaluation
            evaluation = evaluation_repository.get_evaluation_by_llm_output(db, llm_output.id)
            
            if evaluation and evaluation.ground_truth:
                # Calculate WER for Whisper transcription vs ground truth
                whisper_wer_value = wer(evaluation.ground_truth, transcription.text)
                
                # Update evaluation
                evaluation_repository.update_evaluation_wer(
                    db, evaluation.id, whisper_wer=whisper_wer_value
                )
                updated_count += 1
    
    return WordErrorRateResponse(status=f"Updated WER for {updated_count} evaluations successfully.")


@router.post(
    "/llm",
    summary="Use LLM to correct text (with or without RAG)",
)
def correct_text(
    db: Session = Depends(get_db),
    llm_model_name: str = "gpt-4",
    prompt_version: str = "v1",
    limit: int = 10,
    use_rag: bool = False,
    top_k_queries: int = 3,
    top_k_documents: int = 5
):
    """
    Correct Whisper transcriptions using LLM.
    
    Two modes:
    1. RAG + LLM: Uses query_index and medical-documents for enhanced correction
    2. Direct LLM: Uses LLM directly without RAG
    
    :param db: Database session
    :param llm_model_name: LLM model to use (default: "gpt-4")
    :param prompt_version: Version of the prompt (default: "v1")
    :param limit: Maximum number of transcriptions to process
    :param use_rag: Whether to use RAG (default: False - direct LLM correction)
    :param top_k_queries: Number of queries to retrieve from query_index (only if use_rag=True)
    :param top_k_documents: Number of documents per query from medical-documents (only if use_rag=True)
    """
    batch_correct_whisper_text(
        db,
        llm_model_name,
        prompt_version,
        limit,
        use_rag=use_rag,
        top_k_queries=top_k_queries,
        top_k_documents=top_k_documents
    )
    return LLMCorrectResponse(status="Process whisper text successfully.")


@router.post(
    "/llm_wer",
    summary="Calculate WER for LLM outputs",
)
def calculate_llm_wer(
    db: Session = Depends(get_db)
):
    """
    Calculate WER for LLM outputs that have ground truth.
    Calculates both llm_wer (for direct LLM correction) and llm_rag_wer (for RAG-enhanced correction).
    """
    # Get LLM outputs with ground truth
    llm_outputs = llm_output_repository.get_llm_outputs_with_ground_truth(db)
    
    updated_count = 0
    for llm_output in llm_outputs:
        evaluation = evaluation_repository.get_evaluation_by_llm_output(db, llm_output.id)
        
        if evaluation and evaluation.ground_truth:
            # Calculate WER for direct LLM correction (text field)
            llm_wer_value = None
            if llm_output.text:
                llm_wer_value = wer(evaluation.ground_truth, llm_output.text)
            
            # Calculate WER for RAG-enhanced LLM correction (text_with_rag field)
            llm_rag_wer_value = None
            if llm_output.text_with_rag:
                llm_rag_wer_value = wer(evaluation.ground_truth, llm_output.text_with_rag)
            
            # Update evaluation with both WER values
            evaluation_repository.update_evaluation_wer(
                db, evaluation.id, llm_wer=llm_wer_value, llm_rag_wer=llm_rag_wer_value
            )
            updated_count += 1
    
    return WordErrorRateResponse(status=f"Updated WER for {updated_count} LLM outputs successfully.")


@router.post(
    "/import/splited",
    summary="Import all audio files from splited folder into audio_files table",
)
def import_splited_audio_files(
    db: Session = Depends(get_db),
    splited_dir: Optional[str] = None,
    get_duration: bool = True
):
    """
    Scan the splited folder and import all audio file paths into the audio_files table.
    
    :param splited_dir: Path to splited directory (relative to SOURCE_DIR, default: "splited")
    :param get_duration: Whether to get audio file duration (requires ffmpeg)
    :return: Import statistics
    """
    stats = import_audio_files_from_splited(db, splited_dir, get_duration)
    return {
        "status": "Success",
        "message": f"Imported {stats['created']} audio files",
        "statistics": stats
    }

