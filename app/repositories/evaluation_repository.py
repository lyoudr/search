from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.analyze import Evaluation, LLMOutput, LLMModel


def create_evaluation(
    db: Session,
    llm_output_id: int,
    ground_truth: Optional[str] = None,
    whisper_wer: Optional[float] = None,
    llm_wer: Optional[float] = None,
    llm_rag_wer: Optional[float] = None,
    llm_hematology_wer: Optional[float] = None,
    llm_agent_wer: Optional[float] = None,
    google_wer: Optional[float] = None,
    aws_wer: Optional[float] = None,
    dr_ai_wer: Optional[float] = None,
) -> Evaluation:
    """Create a new evaluation record"""
    evaluation = Evaluation(
        llm_output_id=llm_output_id,
        ground_truth=ground_truth,
        whisper_wer=whisper_wer,
        llm_wer=llm_wer,
        llm_rag_wer=llm_rag_wer,
        llm_hematology_wer=llm_hematology_wer,
        llm_agent_wer=llm_agent_wer,
        google_wer=google_wer,
        aws_wer=aws_wer,
        dr_ai_wer=dr_ai_wer,
    )
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)
    return evaluation


def get_evaluation_by_id(db: Session, evaluation_id: int) -> Optional[Evaluation]:
    """Get evaluation by ID"""
    return db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()


def get_evaluation_by_llm_output(db: Session, llm_output_id: int) -> Optional[Evaluation]:
    """Get evaluation for an LLM output"""
    return db.query(Evaluation).filter(Evaluation.llm_output_id == llm_output_id).first()


def update_evaluation_wer(
    db: Session,
    evaluation_id: int,
    whisper_wer: Optional[float] = None,
    llm_wer: Optional[float] = None,
    llm_rag_wer: Optional[float] = None,
    llm_hematology_wer: Optional[float] = None,
    llm_agent_wer: Optional[float] = None,
    google_wer: Optional[float] = None,
    aws_wer: Optional[float] = None,
    dr_ai_wer: Optional[float] = None,
    auto_update_statistics: bool = True,
):
    """
    Update WER values for an evaluation.
    
    :param db: Database session
    :param evaluation_id: Evaluation ID
    :param whisper_wer: Whisper WER value
    :param llm_wer: LLM WER value
    :param llm_rag_wer: LLM RAG WER value
    :param llm_hematology_wer: LLM Hematology WER value
    :param llm_agent_wer: LLM Agent WER value
    :param auto_update_statistics: Whether to automatically update total WER statistics (API1)
    """
    evaluation = db.query(Evaluation).filter(Evaluation.id == evaluation_id).first()
    if evaluation:
        if whisper_wer is not None:
            evaluation.whisper_wer = whisper_wer
        if llm_wer is not None:
            evaluation.llm_wer = llm_wer
        if llm_rag_wer is not None:
            evaluation.llm_rag_wer = llm_rag_wer
        if llm_hematology_wer is not None:
            evaluation.llm_hematology_wer = llm_hematology_wer
        if llm_agent_wer is not None:
            evaluation.llm_agent_wer = llm_agent_wer
        if google_wer is not None:
            evaluation.google_wer = google_wer
        if aws_wer is not None:
            evaluation.aws_wer = aws_wer
        if dr_ai_wer is not None:
            evaluation.dr_ai_wer = dr_ai_wer
        db.commit()
        db.refresh(evaluation)
        
        # API1: Automatically update total WER statistics after each update
        if auto_update_statistics:
            try:
                from app.services.wer_statistics_calculator import update_wer_statistics
                update_wer_statistics(db)
            except Exception as e:
                # Don't fail the update if statistics calculation fails
                print(f"⚠️  Failed to update WER statistics: {e}")
    
    return evaluation


def get_all_evaluations(db: Session) -> List[Evaluation]:
    """Get all evaluations"""
    return db.query(Evaluation).all()


def get_avg_llm_wer_by_model_names(
    db: Session, model_names: List[str]
):
    """
    Get average llm_wer grouped by model name for selected models only.
    """
    return (
        db.query(
            LLMModel.name.label("model_name"),
            func.avg(Evaluation.llm_wer).label("avg_llm_wer"),
            func.count(Evaluation.llm_wer).label("wer_count"),
        )
        .join(LLMOutput, LLMOutput.llm_model_id == LLMModel.id)
        .join(Evaluation, Evaluation.llm_output_id == LLMOutput.id)
        .filter(LLMModel.name.in_(model_names))
        .filter(Evaluation.llm_wer.isnot(None))
        .group_by(LLMModel.name)
        .all()
    )

