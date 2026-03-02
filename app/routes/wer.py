from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.models import get_db
from app.schemas.req_res.audio import WordErrorRateResponse
from app.repositories import (
    transcription_repository,
    llm_output_repository,
    evaluation_repository,
)
from app.services.wer import wer


router = APIRouter(tags=["wer"], prefix="/wer")


@router.post(
    "/whisper",
    summary="Calculate WER for Whisper transcriptions",
)
def calculate_whisper_wer(db: Session = Depends(get_db)):
    transcriptions = transcription_repository.get_all_transcriptions(db)

    updated_count = 0
    for transcription in transcriptions:
        llm_outputs = llm_output_repository.get_llm_outputs_by_transcription(
            db, transcription.id
        )

        for llm_output in llm_outputs:
            evaluation = evaluation_repository.get_evaluation_by_llm_output(
                db, llm_output.id
            )
            if evaluation and evaluation.ground_truth:
                whisper_wer_value = wer(evaluation.ground_truth, transcription.text)
                evaluation_repository.update_evaluation_wer(
                    db, evaluation.id, whisper_wer=whisper_wer_value
                )
                updated_count += 1

    return WordErrorRateResponse(
        status=f"Updated WER for {updated_count} evaluations successfully."
    )


@router.post(
    "/llm",
    summary="Calculate WER for LLM outputs",
)
def calculate_llm_wer(db: Session = Depends(get_db)):
    llm_outputs = llm_output_repository.get_llm_outputs_with_ground_truth(db)

    updated_count = 0
    for llm_output in llm_outputs:
        evaluation = evaluation_repository.get_evaluation_by_llm_output(
            db, llm_output.id
        )

        if evaluation and evaluation.ground_truth:
            gt = evaluation.ground_truth
            llm_wer_value = wer(gt, llm_output.text) if llm_output.text else None
            llm_rag_wer_value = (
                wer(gt, llm_output.text_with_rag) if llm_output.text_with_rag else None
            )
            llm_hematology_wer_value = (
                wer(gt, llm_output.text_with_hematology)
                if llm_output.text_with_hematology
                else None
            )
            llm_agent_wer_value = (
                wer(gt, llm_output.text_agent) if llm_output.text_agent else None
            )
            google_wer_value = (
                wer(gt, llm_output.text_with_google)
                if llm_output.text_with_google
                else None
            )
            aws_wer_value = (
                wer(gt, llm_output.text_with_aws) if llm_output.text_with_aws else None
            )
            dr_ai_wer_value = (
                wer(gt, llm_output.text_with_dr_ai) if llm_output.text_with_dr_ai else None
            )

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
    "/llm/model-comparison",
    summary="Compare average llm_wer across selected LLM models",
)
def get_llm_model_wer_comparison(db: Session = Depends(get_db)):
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
        avg_llm_wer = (
            float(row.avg_llm_wer)
            if row and row.avg_llm_wer is not None
            else None
        )
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


@router.post(
    "/statistics/recalculate", summary="Recalculate total WER statistics (API2)"
)
def recalculate_wer_statistics(db: Session = Depends(get_db)):
    from app.services.wer_statistics_calculator import recalculate_wer_statistics

    stats = recalculate_wer_statistics(db)
    return {
        "status": "success",
        "message": "WER statistics recalculated successfully",
        "statistics": stats,
    }


@router.get("/statistics/latest", summary="Get latest WER statistics")
def get_latest_wer_statistics(db: Session = Depends(get_db)):
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


@router.get("/statistics/all", summary="Get all WER statistics records (history)")
def get_all_wer_statistics(db: Session = Depends(get_db)):
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
