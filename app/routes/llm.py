from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from app.config.settings import get_settings
from app.models import get_db
from app.schemas.req_res.audio import (
    AnalyzeRequest,
    ParseResponse,
    LLMCorrectResponse,
    GoogleTranscribeRequest,
    AwsTranscribeRequest,
)
from app.services.audio_core import whisper_to_text
from app.services.correction_core import batch_correct_whisper_text_with_agent
from app.services.transcribe import transcribe_audio_aws, transcribe_audio_google
from app.ai_agents.agents.llm_agent import LLMAgent
from app.repositories import llm_output_repository


router = APIRouter(tags=["llm"], prefix="/llm")
settings = get_settings()


@router.post(
    "",
    summary="Use LLM to correct text (with or without RAG)",
)
def correct_text(
    db: Session = Depends(get_db),
    llm_model_name: str = "gpt-5.2",
    prompt_version: str = "v1",
    limit: int = 10,
    use_rag: bool = False,
    top_k_queries: int = 3,
    top_k_documents: int = 5,
):
    llm_agent = LLMAgent()
    llm_agent.execute_tool(
        "batch_correct_whisper_text",
        db=db,
        llm_model_name=llm_model_name,
        prompt_version=prompt_version,
        limit=limit,
        use_rag=use_rag,
        top_k_queries=top_k_queries,
        top_k_documents=top_k_documents,
    )
    mode_str = "with RAG" if use_rag else "without RAG"
    return LLMCorrectResponse(status=f"Process whisper text successfully ({mode_str}).")


@router.post(
    "/hematology",
    summary="Use LLM to correct text with Hematology Dictionary RAG",
)
def correct_text_with_hematology(
    db: Session = Depends(get_db),
    llm_model_name: str = "gpt-5.2",
    prompt_version: str = "v1",
    limit: int = 10,
    top_k_queries: int = 2,
    top_k: int = 5,
):
    llm_agent = LLMAgent()
    llm_agent.execute_tool(
        "batch_correct_whisper_text_with_hematology",
        db=db,
        llm_model_name=llm_model_name,
        prompt_version=prompt_version,
        limit=limit,
        top_k_queries=top_k_queries,
        top_k=top_k,
    )
    return LLMCorrectResponse(
        status="Process whisper text successfully (with Hematology Dictionary RAG)."
    )


@router.post(
    "/agent",
    summary="Use Agent to correct text with dynamic tool selection",
)
def correct_text_with_agent(
    db: Session = Depends(get_db),
    llm_model_name: str = "gpt-5.2",
    prompt_version: str = "v1",
    limit: int = 10,
    max_iterations: int = 3,
    initial_strategy: Optional[str] = None,
):
    batch_correct_whisper_text_with_agent(
        db=db,
        llm_model_name=llm_model_name,
        prompt_version=prompt_version,
        limit=limit,
        max_iterations=max_iterations,
        initial_strategy=initial_strategy,
    )
    return LLMCorrectResponse(status="Process whisper text successfully (with Agent).")


@router.post(
    "/whisper/analyze",
    summary="Whisper analyze audio files and extract medical terms",
)
def whisper_analyze(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    extraction_model: str = "gpt-5.2",
):
    whisper_to_text(
        db,
        f"{settings.SOURCE_DIR}/{payload.input_dir}",
        extraction_model=extraction_model,
    )
    return ParseResponse(
        status="Success analyze audio file with Whisper and extract medical terms"
    )


@router.post(
    "/google-transcribe-batch",
    summary="Transcribe all LLM outputs with null text_with_google via Google Speech-to-Text",
)
def google_transcribe_batch(
    payload: GoogleTranscribeRequest,
    db: Session = Depends(get_db),
):
    rows = llm_output_repository.get_llm_outputs_with_null_text_with_google(db)
    processed = 0
    failed = 0
    errors = []

    for llm_output in rows:
        if not llm_output.transcription:
            failed += 1
            errors.append(
                {"llm_output_id": llm_output.id, "error": "No transcription linked"}
            )
            continue
        audio_file = llm_output.transcription.audio_file
        if not audio_file or not audio_file.file_path:
            failed += 1
            errors.append(
                {"llm_output_id": llm_output.id, "error": "No audio file or path"}
            )
            continue

        try:
            transcript = transcribe_audio_google(
                audio_path=audio_file.file_path,
                language_code=payload.language_code,
                sample_rate_hertz=payload.sample_rate_hertz,
            )
            llm_output_repository.update_llm_output(
                db, llm_output.id, text_with_google=transcript
            )
            processed += 1
        except FileNotFoundError as e:
            failed += 1
            errors.append({"llm_output_id": llm_output.id, "error": str(e)})
        except Exception as e:
            failed += 1
            errors.append({"llm_output_id": llm_output.id, "error": f"{e!s}"})

    return {
        "status": "success",
        "total_found": len(rows),
        "processed": processed,
        "failed": failed,
        "errors": errors,
    }


@router.post(
    "/aws-transcribe-batch",
    summary="Transcribe all LLM outputs with null text_with_aws via Amazon Transcribe",
)
def aws_transcribe_batch(
    payload: AwsTranscribeRequest,
    db: Session = Depends(get_db),
):
    if not settings.AWS_S3_BUCKET:
        raise HTTPException(
            status_code=503,
            detail="AWS_S3_BUCKET is not configured",
        )
    rows = llm_output_repository.get_llm_outputs_with_null_text_with_aws(db)
    processed = 0
    failed = 0
    errors = []

    for llm_output in rows:
        if not llm_output.transcription:
            failed += 1
            errors.append(
                {"llm_output_id": llm_output.id, "error": "No transcription linked"}
            )
            continue
        audio_file = llm_output.transcription.audio_file
        if not audio_file or not audio_file.file_path:
            failed += 1
            errors.append(
                {"llm_output_id": llm_output.id, "error": "No audio file or path"}
            )
            continue
        try:
            transcript = transcribe_audio_aws(
                audio_path=audio_file.file_path,
                language_code=payload.language_code,
                media_format=payload.media_format,
                region_name=settings.AWS_REGION,
                bucket=settings.AWS_S3_BUCKET,
            )
            llm_output_repository.update_llm_output(
                db, llm_output.id, text_with_aws=transcript
            )
            processed += 1
        except FileNotFoundError as e:
            failed += 1
            errors.append({"llm_output_id": llm_output.id, "error": str(e)})
        except Exception as e:
            failed += 1
            errors.append({"llm_output_id": llm_output.id, "error": f"{e!s}"})

    return {
        "status": "success",
        "total_found": len(rows),
        "processed": processed,
        "failed": failed,
        "errors": errors,
    }
