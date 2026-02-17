from sqlalchemy.orm import Session
from typing import Optional
from app.models.analyze import WerStatistics


def get_latest_statistics(db: Session) -> Optional[WerStatistics]:
    """Get the latest WER statistics record"""
    return db.query(WerStatistics).order_by(WerStatistics.updated_at.desc()).first()


def create_statistics(
    db: Session,
    total_evaluations: int,
    avg_whisper_wer: Optional[float] = None,
    avg_llm_wer: Optional[float] = None,
    avg_llm_rag_wer: Optional[float] = None,
    avg_llm_hematology_wer: Optional[float] = None,
    avg_llm_agent_wer: Optional[float] = None,
    avg_google_wer: Optional[float] = None,
    avg_aws_wer: Optional[float] = None,
    avg_dr_ai_wer: Optional[float] = None,
    count_whisper_wer: int = 0,
    count_llm_wer: int = 0,
    count_llm_rag_wer: int = 0,
    count_llm_hematology_wer: int = 0,
    count_llm_agent_wer: int = 0,
) -> WerStatistics:
    """Create a new WER statistics record"""
    statistics = WerStatistics(
        total_evaluations=total_evaluations,
        avg_whisper_wer=avg_whisper_wer,
        avg_llm_wer=avg_llm_wer,
        avg_llm_rag_wer=avg_llm_rag_wer,
        avg_llm_hematology_wer=avg_llm_hematology_wer,
        avg_llm_agent_wer=avg_llm_agent_wer,
        avg_google_wer=avg_google_wer,
        avg_aws_wer=avg_aws_wer,
        avg_dr_ai_wer=avg_dr_ai_wer,
        count_whisper_wer=count_whisper_wer,
        count_llm_wer=count_llm_wer,
        count_llm_rag_wer=count_llm_rag_wer,
        count_llm_hematology_wer=count_llm_hematology_wer,
        count_llm_agent_wer=count_llm_agent_wer,
    )
    db.add(statistics)
    db.commit()
    db.refresh(statistics)
    return statistics


def update_statistics(
    db: Session,
    statistics_id: int,
    total_evaluations: Optional[int] = None,
    avg_whisper_wer: Optional[float] = None,
    avg_llm_wer: Optional[float] = None,
    avg_llm_rag_wer: Optional[float] = None,
    avg_llm_hematology_wer: Optional[float] = None,
    avg_llm_agent_wer: Optional[float] = None,
    avg_google_wer: Optional[float] = None,
    avg_aws_wer: Optional[float] = None,
    avg_dr_ai_wer: Optional[float] = None,
    count_whisper_wer: Optional[int] = None,
    count_llm_wer: Optional[int] = None,
    count_llm_rag_wer: Optional[int] = None,
    count_llm_hematology_wer: Optional[int] = None,
    count_llm_agent_wer: Optional[int] = None,
) -> Optional[WerStatistics]:
    """Update an existing WER statistics record"""
    statistics = db.query(WerStatistics).filter(WerStatistics.id == statistics_id).first()
    if statistics:
        if total_evaluations is not None:
            statistics.total_evaluations = total_evaluations
        if avg_whisper_wer is not None:
            statistics.avg_whisper_wer = avg_whisper_wer
        if avg_llm_wer is not None:
            statistics.avg_llm_wer = avg_llm_wer
        if avg_llm_rag_wer is not None:
            statistics.avg_llm_rag_wer = avg_llm_rag_wer
        if avg_llm_hematology_wer is not None:
            statistics.avg_llm_hematology_wer = avg_llm_hematology_wer
        if avg_llm_agent_wer is not None:
            statistics.avg_llm_agent_wer = avg_llm_agent_wer
        if avg_google_wer is not None:
            statistics.avg_google_wer = avg_google_wer
        if avg_aws_wer is not None:
            statistics.avg_aws_wer = avg_aws_wer
        if avg_dr_ai_wer is not None:
            statistics.avg_dr_ai_wer = avg_dr_ai_wer
        if count_whisper_wer is not None:
            statistics.count_whisper_wer = count_whisper_wer
        if count_llm_wer is not None:
            statistics.count_llm_wer = count_llm_wer
        if count_llm_rag_wer is not None:
            statistics.count_llm_rag_wer = count_llm_rag_wer
        if count_llm_hematology_wer is not None:
            statistics.count_llm_hematology_wer = count_llm_hematology_wer
        if count_llm_agent_wer is not None:
            statistics.count_llm_agent_wer = count_llm_agent_wer
        db.commit()
        db.refresh(statistics)
    return statistics


def get_all_statistics(db: Session) -> list:
    """Get all WER statistics records"""
    return db.query(WerStatistics).order_by(WerStatistics.updated_at.desc()).all()
