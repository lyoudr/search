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
    AnalyzeRequest,
    WordErrorRateResponse,
    LLMCorrectResponse,
)
from app.services.parse_audio import convert_to_wav
from app.services.split_audio import split_audio as split_audio_service
from app.services.speech2text import whisper_to_text
from app.services.llm import (
    batch_correct_whisper_text,
    batch_correct_whisper_text_with_hematology,
    batch_correct_whisper_text_with_agent,
)
from app.services.import_audio_files import import_audio_files_from_splited
from app.repositories import (
    transcription_repository,
    llm_output_repository,
    evaluation_repository,
)
from app.services.wer import wer

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


@router.post(
    "/whisper/analyze",
    summary="Whisper analyze audio files and extract medical terms",
)
def whisper_analyze(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    extraction_model: str = "gpt-5.2",
):
    """
    Analyze audio files with Whisper.
    Automatically extracts medical terms using LLM and stores them in query_index.

    :param payload: Request payload with input_dir
    :param db: Database session
    :param extraction_model: LLM model to use for term extraction (default: "gpt-5.2")
    """
    whisper_to_text(
        db,
        f"{settings.SOURCE_DIR}/{payload.input_dir}",
        extraction_model=extraction_model,
    )
    return ParseResponse(
        status="Success analyze audio file with Whisper and extract medical terms"
    )


@router.post(
    "/whisper_wer",
    summary="Calculate WER for Whisper transcriptions",
)
def calculate_whisper_wer(db: Session = Depends(get_db)):
    """
    Calculate WER for transcriptions that have ground truth.
    Note: This requires evaluations to be created with ground_truth.
    """
    # Get all transcriptions
    transcriptions = transcription_repository.get_all_transcriptions(db)

    updated_count = 0
    for transcription in transcriptions:
        # Get LLM outputs for this transcription
        llm_outputs = llm_output_repository.get_llm_outputs_by_transcription(
            db, transcription.id
        )

        for llm_output in llm_outputs:
            # Get or create evaluation
            evaluation = evaluation_repository.get_evaluation_by_llm_output(
                db, llm_output.id
            )

            if evaluation and evaluation.ground_truth:
                # Calculate WER for Whisper transcription vs ground truth
                whisper_wer_value = wer(evaluation.ground_truth, transcription.text)

                # Update evaluation
                evaluation_repository.update_evaluation_wer(
                    db, evaluation.id, whisper_wer=whisper_wer_value
                )
                updated_count += 1

    return WordErrorRateResponse(
        status=f"Updated WER for {updated_count} evaluations successfully."
    )


@router.post(
    "/llm",
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
    """
    Correct Whisper transcriptions using LLM.

    Two modes:
    1. use_rag=False: Direct LLM correction (without RAG) - stores in 'text' column
    2. use_rag=True: LLM correction with RAG (using medical documents) - stores in 'text_with_rag' column

    :param db: Database session
    :param llm_model_name: LLM model to use (default: "gpt-5.2")
    :param prompt_version: Version of the prompt (default: "v1")
    :param limit: Maximum number of transcriptions to process
    :param use_rag: Whether to use RAG (Retrieval-Augmented Generation) with medical documents (default: False)
    :param top_k_queries: Number of queries (terms) to retrieve from query_index (only if use_rag=True)
    :param top_k_documents: Number of documents per query from medical-documents (only if use_rag=True)
    """
    batch_correct_whisper_text(
        db,
        llm_model_name,
        prompt_version,
        limit,
        use_rag=use_rag,
        top_k_queries=top_k_queries,
        top_k_documents=top_k_documents,
    )
    mode_str = "with RAG" if use_rag else "without RAG"
    return LLMCorrectResponse(status=f"Process whisper text successfully ({mode_str}).")


@router.post(
    "/llm/hematology",
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
    """
    Correct Whisper transcriptions using LLM with Hematology Dictionary RAG.

    Process:
    1. Use transcription_id to query query_index and get stored medical terms (keywords)
    2. Use these keywords to search hematology dictionary index
    3. Use retrieved hematology dictionary entries as context for LLM correction
    4. Store results in 'text_with_hematology' column.

    :param db: Database session
    :param llm_model_name: LLM model to use (default: "gpt-5.2")
    :param prompt_version: Version of the prompt (default: "v1")
    :param limit: Maximum number of transcriptions to process
    :param top_k_queries: Number of queries (terms) to retrieve from query_index
    :param top_k: Number of hematology dictionary entries to retrieve per query
    """
    batch_correct_whisper_text_with_hematology(
        db,
        llm_model_name,
        prompt_version,
        limit,
        top_k_queries=top_k_queries,
        top_k=top_k,
    )
    return LLMCorrectResponse(
        status=f"Process whisper text successfully (with Hematology Dictionary RAG)."
    )


@router.post(
    "/llm/agent",
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
    """
    Correct Whisper transcriptions using Agent-based approach.

    The agent dynamically selects and combines tools (direct LLM, medical RAG,
    hematology RAG, combined RAG) to achieve the best correction quality.

    Process:
    1. Agent analyzes the transcription
    2. Selects initial strategy (or uses provided one)
    3. Tries different tools based on quality assessments
    4. Iterates if quality is low
    5. Stores results in 'text_agent' column

    :param db: Database session
    :param llm_model_name: LLM model to use (default: "gpt-5.2")
    :param prompt_version: Version of the prompt (default: "v1")
    :param limit: Maximum number of transcriptions to process
    :param max_iterations: Maximum number of correction attempts per transcription
    :param initial_strategy: Initial strategy to try (None for auto-select)
                            Options: "direct_llm", "medical_document_rag", "hematology_rag", "combined_rag"
    """
    batch_correct_whisper_text_with_agent(
        db,
        llm_model_name,
        prompt_version,
        limit,
        max_iterations=max_iterations,
        initial_strategy=initial_strategy,
    )
    return LLMCorrectResponse(status=f"Process whisper text successfully (with Agent).")


@router.post(
    "/llm_wer",
    summary="Calculate WER for LLM outputs",
)
def calculate_llm_wer(db: Session = Depends(get_db)):
    """
    Calculate WER for LLM outputs that have ground truth.
    Calculates llm_wer (for direct LLM correction), llm_rag_wer (for RAG-enhanced correction),
    llm_hematology_wer (for Hematology Dictionary RAG-enhanced correction),
    llm_agent_wer (for agent-based correction),
    and google_wer / aws_wer / dr_ai_wer for cloud STT outputs.
    """
    llm_outputs = llm_output_repository.get_llm_outputs_with_ground_truth(db)

    updated_count = 0
    for llm_output in llm_outputs:
        evaluation = evaluation_repository.get_evaluation_by_llm_output(
            db, llm_output.id
        )

        if evaluation and evaluation.ground_truth:
            gt = evaluation.ground_truth

            # Calculate WER for direct LLM correction (text field)
            llm_wer_value = wer(gt, llm_output.text) if llm_output.text else None

            # Calculate WER for RAG-enhanced LLM correction (text_with_rag field)
            llm_rag_wer_value = (
                wer(gt, llm_output.text_with_rag)
                if llm_output.text_with_rag
                else None
            )

            # Calculate WER for Hematology Dictionary RAG-enhanced LLM correction
            llm_hematology_wer_value = (
                wer(gt, llm_output.text_with_hematology)
                if llm_output.text_with_hematology
                else None
            )

            # Calculate WER for agent-based LLM correction (text_agent field)
            llm_agent_wer_value = (
                wer(gt, llm_output.text_agent) if llm_output.text_agent else None
            )

            # Calculate WER for cloud STT outputs
            google_wer_value = (
                wer(gt, llm_output.text_with_google)
                if llm_output.text_with_google
                else None
            )
            aws_wer_value = (
                wer(gt, llm_output.text_with_aws)
                if llm_output.text_with_aws
                else None
            )
            dr_ai_wer_value = (
                wer(gt, llm_output.text_with_dr_ai)
                if llm_output.text_with_dr_ai
                else None
            )

            # Update evaluation with all WER values
            evaluation_repository.update_evaluation_wer(
                db,
                evaluation.id,
                llm_wer=llm_wer_value,
                llm_rag_wer=llm_rag_wer_value,
                llm_hematology_wer=llm_hematology_wer_value,
                llm_agent_wer=llm_agent_wer_value,
                google_wer=google_wer_value,
                aws_wer=aws_wer_value,
                dr_ai_wer=dr_ai_wer_value,
            )
            updated_count += 1

    return WordErrorRateResponse(
        status=f"Updated WER for {updated_count} LLM outputs successfully."
    )


@router.get(
    "/wer/llm_model_comparison",
    summary="Compare average llm_wer across selected LLM models",
)
def get_llm_model_wer_comparison(db: Session = Depends(get_db)):
    """
    Compare average llm_wer for selected LLM models based on evaluations table.
    Only compares direct LLM correction metric: evaluations.llm_wer.
    """
    target_models = [
        "gpt-4o",
        "gpt-5.2",
        "qwen2.5-7b-instruct",
        "llama-3-8b-instruct",
    ]

    rows = evaluation_repository.get_avg_llm_wer_by_model_names(db, target_models)
    rows_map = {row.model_name: row for row in rows}

    models = []
    total_wer_sum = 0.0
    total_wer_count = 0

    for model_name in target_models:
        row = rows_map.get(model_name)
        avg_llm_wer = float(row.avg_llm_wer) if row and row.avg_llm_wer is not None else None
        wer_count = int(row.wer_count) if row else 0

        if avg_llm_wer is not None and wer_count > 0:
            total_wer_sum += avg_llm_wer * wer_count
            total_wer_count += wer_count

        models.append(
            {
                "model_name": model_name,
                "avg_llm_wer": avg_llm_wer,
                "wer_count": wer_count,
            }
        )

    overall_avg_llm_wer = (
        total_wer_sum / total_wer_count if total_wer_count > 0 else None
    )

    return {
        "status": "success",
        "metric": "llm_wer",
        "overall_avg_llm_wer": overall_avg_llm_wer,
        "total_wer_count": total_wer_count,
        "models": models,
    }


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
    "/wer_statistics/recalculate", summary="Recalculate total WER statistics (API2)"
)
def recalculate_wer_statistics(db: Session = Depends(get_db)):
    """
    Recalculate total WER statistics from all evaluations and create a new record.
    This is API2 - manually trigger recalculation of current total WER.

    Calculates:
    - Total number of evaluations with ground truth
    - Average WER for each method (whisper, llm, llm_rag, llm_hematology, llm_agent)
    - Count of evaluations with each WER type calculated

    :return: Recalculated statistics
    """
    from app.services.wer_statistics_calculator import recalculate_wer_statistics

    stats = recalculate_wer_statistics(db)

    return {
        "status": "success",
        "message": "WER statistics recalculated successfully",
        "statistics": stats,
    }


@router.get("/wer_statistics/latest", summary="Get latest WER statistics")
def get_latest_wer_statistics(db: Session = Depends(get_db)):
    """
    Get the latest WER statistics record.

    :return: Latest statistics record
    """
    from app.repositories import wer_statistics_repository

    stats = wer_statistics_repository.get_latest_statistics(db)

    if not stats:
        return {
            "status": "not_found",
            "message": "No WER statistics found. Please run recalculation first.",
            "statistics": None,
        }

    return {
        "status": "success",
        "statistics": {
            "id": stats.id,
            "total_evaluations": stats.total_evaluations,
            "avg_whisper_wer": stats.avg_whisper_wer,
            "avg_llm_wer": stats.avg_llm_wer,
            "avg_llm_rag_wer": stats.avg_llm_rag_wer,
            "avg_llm_hematology_wer": stats.avg_llm_hematology_wer,
            "avg_llm_agent_wer": stats.avg_llm_agent_wer,
            "count_whisper_wer": stats.count_whisper_wer,
            "count_llm_wer": stats.count_llm_wer,
            "count_llm_rag_wer": stats.count_llm_rag_wer,
            "count_llm_hematology_wer": stats.count_llm_hematology_wer,
            "count_llm_agent_wer": stats.count_llm_agent_wer,
            "created_at": stats.created_at.isoformat() if stats.created_at else None,
            "updated_at": stats.updated_at.isoformat() if stats.updated_at else None,
        },
    }


@router.get("/wer_statistics/all", summary="Get all WER statistics records (history)")
def get_all_wer_statistics(db: Session = Depends(get_db)):
    """
    Get all WER statistics records for history tracking.

    :return: List of all statistics records
    """
    from app.repositories import wer_statistics_repository

    all_stats = wer_statistics_repository.get_all_statistics(db)

    return {
        "status": "success",
        "count": len(all_stats),
        "statistics": [
            {
                "id": stats.id,
                "total_evaluations": stats.total_evaluations,
                "avg_whisper_wer": stats.avg_whisper_wer,
                "avg_llm_wer": stats.avg_llm_wer,
                "avg_llm_rag_wer": stats.avg_llm_rag_wer,
                "avg_llm_hematology_wer": stats.avg_llm_hematology_wer,
                "avg_llm_agent_wer": stats.avg_llm_agent_wer,
                "count_whisper_wer": stats.count_whisper_wer,
                "count_llm_wer": stats.count_llm_wer,
                "count_llm_rag_wer": stats.count_llm_rag_wer,
                "count_llm_hematology_wer": stats.count_llm_hematology_wer,
                "count_llm_agent_wer": stats.count_llm_agent_wer,
                "created_at": (
                    stats.created_at.isoformat() if stats.created_at else None
                ),
                "updated_at": (
                    stats.updated_at.isoformat() if stats.updated_at else None
                ),
            }
            for stats in all_stats
        ],
    }


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
