from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import Optional
import io
import pandas as pd

from app.config.settings import get_settings
from app.models import get_db
from app.schemas.req_res.audio import (
    ParseRequest,
    ParseResponse,
)
from app.services.audio_core import convert_to_wav
from app.services.audio_core import split_audio as split_audio_service
from app.services.import_audio_files import import_audio_files_from_splited
from app.repositories import (
    transcription_repository,
)


router = APIRouter(tags=["audio"], prefix="/audio")
settings = get_settings()


@router.post(
    "/parse",
    summary="Parse audio files to .wav",
)
def parse_audio(payload: ParseRequest):
    convert_to_wav(
        f"{settings.SOURCE_DIR}/{payload.input_dir}",
        f"{settings.SOURCE_DIR}/{payload.output_dir}",
    )
    return ParseResponse(status="Success parse audio file to .wav")


@router.post(
    "/split",
    summary="Split audio files into chunks",
)
def split_audio(payload: ParseRequest):
    split_audio_service(
        f"{settings.SOURCE_DIR}/{payload.input_dir}",
        f"{settings.SOURCE_DIR}/{payload.output_dir}",
    )
    return ParseResponse(status="Success split audio file into chunks")


@router.get(
    "/export/xlsx",
    summary="Export transcriptions (whisper / google / aws / dr_ai) to XLSX",
)
def export_transcriptions_xlsx(
    audio_file_id: Optional[int] = None,
    db: Session = Depends(get_db),
):
    """
    Export an XLSX file with columns:
    1. whisper_text   (Transcription.text)
    2. google_text    (LLMOutput.text_with_google)
    3. aws_text       (LLMOutput.text_with_aws)
    4. dr_ai_text     (LLMOutput.text_with_dr_ai)
    """
    rows = transcription_repository.export_transcriptions_text(
        db, audio_file_id=audio_file_id
    )

    data = [
        {
            "whisper_text": r.whisper_text,
            "google_text": r.google_text,
            "aws_text": r.aws_text,
            "dr_ai_text": r.dr_ai_text,
        }
        for r in rows
    ]

    df = pd.DataFrame(data)

    output = io.BytesIO()
    # Requires openpyxl in requirements.txt
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=transcriptions.xlsx"},
    )


@router.post(
    "/import/splited",
    summary="Import all audio files from splited folder into audio_files table",
)
def import_splited_audio_files(
    db: Session = Depends(get_db),
    splited_dir: Optional[str] = None,
    get_duration: bool = True,
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
        "statistics": stats,
    }
