from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

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
from app.services.llm import batch_correct_whisper_text_with_gpt4
from app.repositories import audio_repository

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
    summary="Whisper analyze audio files",
)
def whisper_analyze(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db)  # Assuming you have a function to get DB session
):
    whisper_to_text(
        db,
        f"{settings.SOURCE_DIR}/{payload.input_dir}"
    )
    return ParseResponse(status="Success analyze audio file with Whisper")


@router.post(
    "/whisper_wer",
    summary="Whisper analyze audio files with WER",
)
def whisper_wer(
    db: Session = Depends(get_db)  # Assuming you have a function to get DB session
):
    records = audio_repository.get_whisper_analyze_records(db) 
    audio_repository.update_whisper_analyze_wer(db, records)
    return WordErrorRateResponse(status = "Update WER successfully.")


@router.post(
    "/llm",
    summary="Use GPT-4 to correct text",
)
def correct_text(
    db: Session = Depends(get_db)
):
    batch_correct_whisper_text_with_gpt4(db)
    return LLMCorrectResponse(status = "Process whisper text successfully.")


@router.post(
    "/llm_wer",
    summary="LLM analyzed WER",
)
def llm_wer(
    db: Session = Depends(get_db) 
):
    records = audio_repository.get_llm_analyze_records(db)
    audio_repository.update_llm_analyze_wer(db, records)
    return WordErrorRateResponse(status = "Update WER successfully.")