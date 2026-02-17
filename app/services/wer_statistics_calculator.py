"""
WER Statistics Calculator Service
Calculates aggregated WER statistics across all evaluations
"""

from sqlalchemy.orm import Session
from typing import Dict, Any
from sqlalchemy import func

from app.repositories import wer_statistics_repository
from app.models.analyze import Evaluation


def calculate_total_wer_statistics(db: Session) -> Dict[str, Any]:
    """
    Calculate total WER statistics from all evaluations.

    :param db: Database session
    :return: Dictionary with calculated statistics
    """
    # Get all evaluations with ground truth
    evaluations = db.query(Evaluation).filter(Evaluation.ground_truth.isnot(None)).all()

    total_evaluations = len(evaluations)

    # Calculate averages and counts for each WER type
    stats = {
        "total_evaluations": total_evaluations,
        "avg_whisper_wer": None,
        "avg_llm_wer": None,
        "avg_llm_rag_wer": None,
        "avg_llm_hematology_wer": None,
        "avg_llm_agent_wer": None,
        "avg_google_wer": None,
        "avg_aws_wer": None,
        "avg_dr_ai_wer": None,
        "count_whisper_wer": 0,
        "count_llm_wer": 0,
        "count_llm_rag_wer": 0,
        "count_llm_hematology_wer": 0,
        "count_llm_agent_wer": 0,
    }

    if total_evaluations == 0:
        return stats

    # Calculate averages using SQL aggregation for efficiency
    # Whisper WER
    whisper_result = (
        db.query(
            func.avg(Evaluation.whisper_wer).label("avg"),
            func.count(Evaluation.whisper_wer).label("count"),
        )
        .filter(Evaluation.whisper_wer.isnot(None))
        .first()
    )

    if whisper_result and whisper_result.avg is not None:
        stats["avg_whisper_wer"] = round(float(whisper_result.avg), 4)
        stats["count_whisper_wer"] = whisper_result.count

    # LLM WER
    llm_result = (
        db.query(
            func.avg(Evaluation.llm_wer).label("avg"),
            func.count(Evaluation.llm_wer).label("count"),
        )
        .filter(Evaluation.llm_wer.isnot(None))
        .first()
    )

    if llm_result and llm_result.avg is not None:
        stats["avg_llm_wer"] = round(float(llm_result.avg), 4)
        stats["count_llm_wer"] = llm_result.count

    # LLM RAG WER
    llm_rag_result = (
        db.query(
            func.avg(Evaluation.llm_rag_wer).label("avg"),
            func.count(Evaluation.llm_rag_wer).label("count"),
        )
        .filter(Evaluation.llm_rag_wer.isnot(None))
        .first()
    )

    if llm_rag_result and llm_rag_result.avg is not None:
        stats["avg_llm_rag_wer"] = round(float(llm_rag_result.avg), 4)
        stats["count_llm_rag_wer"] = llm_rag_result.count

    # LLM Hematology WER
    llm_hematology_result = (
        db.query(
            func.avg(Evaluation.llm_hematology_wer).label("avg"),
            func.count(Evaluation.llm_hematology_wer).label("count"),
        )
        .filter(Evaluation.llm_hematology_wer.isnot(None))
        .first()
    )

    if llm_hematology_result and llm_hematology_result.avg is not None:
        stats["avg_llm_hematology_wer"] = round(float(llm_hematology_result.avg), 4)
        stats["count_llm_hematology_wer"] = llm_hematology_result.count

    # LLM Agent WER
    llm_agent_result = (
        db.query(
            func.avg(Evaluation.llm_agent_wer).label("avg"),
            func.count(Evaluation.llm_agent_wer).label("count"),
        )
        .filter(Evaluation.llm_agent_wer.isnot(None))
        .first()
    )

    if llm_agent_result and llm_agent_result.avg is not None:
        stats["avg_llm_agent_wer"] = round(float(llm_agent_result.avg), 4)
        stats["count_llm_agent_wer"] = llm_agent_result.count

    # Google WER
    google_result = (
        db.query(
            func.avg(Evaluation.google_wer).label("avg"),
            func.count(Evaluation.google_wer).label("count"),
        )
        .filter(Evaluation.google_wer.isnot(None))
        .first()
    )

    if google_result and google_result.avg is not None:
        stats["avg_google_wer"] = round(float(google_result.avg), 4)

    # AWS WER
    aws_result = (
        db.query(
            func.avg(Evaluation.aws_wer).label("avg"),
            func.count(Evaluation.aws_wer).label("count"),
        )
        .filter(Evaluation.aws_wer.isnot(None))
        .first()
    )

    if aws_result and aws_result.avg is not None:
        stats["avg_aws_wer"] = round(float(aws_result.avg), 4)

    # Dr_AI WER
    dr_ai_result = (
        db.query(
            func.avg(Evaluation.dr_ai_wer).label("avg"),
            func.count(Evaluation.dr_ai_wer).label("count"),
        )
        .filter(Evaluation.dr_ai_wer.isnot(None))
        .first()
    )

    if dr_ai_result and dr_ai_result.avg is not None:
        stats["avg_dr_ai_wer"] = round(float(dr_ai_result.avg), 4)

    return stats


def update_wer_statistics(db: Session) -> Dict[str, Any]:
    """
    Calculate and update WER statistics in the database.
    This is called automatically when evaluations are updated (API1).

    :param db: Database session
    :return: Dictionary with updated statistics
    """
    # Calculate statistics
    stats = calculate_total_wer_statistics(db)

    # Get or create statistics record
    existing_stats = wer_statistics_repository.get_latest_statistics(db)

    if existing_stats:
        # Update existing record
        wer_statistics_repository.update_statistics(
            db=db, statistics_id=existing_stats.id, **stats
        )
        print(f"✅ Updated WER statistics (ID: {existing_stats.id})")
    else:
        # Create new record
        new_stats = wer_statistics_repository.create_statistics(db=db, **stats)
        print(f"✅ Created new WER statistics (ID: {new_stats.id})")

    return stats


def recalculate_wer_statistics(db: Session) -> Dict[str, Any]:
    """
    Recalculate WER statistics from scratch and create a new record.
    This is the manual API endpoint (API2).

    :param db: Database session
    :return: Dictionary with recalculated statistics
    """
    # Calculate statistics
    stats = calculate_total_wer_statistics(db)

    # Always create a new record (for history tracking)
    new_stats = wer_statistics_repository.create_statistics(db=db, **stats)
    print(f"✅ Recalculated and created new WER statistics (ID: {new_stats.id})")

    return stats
