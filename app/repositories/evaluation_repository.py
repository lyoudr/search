from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.analyze import Evaluation


def create_evaluation(db: Session, llm_output_id: int, ground_truth: Optional[str] = None,
                      whisper_wer: Optional[float] = None, llm_wer: Optional[float] = None,
                      llm_rag_wer: Optional[float] = None,
                      llm_hematology_wer: Optional[float] = None,
                      llm_agent_wer: Optional[float] = None) -> Evaluation:
    """Create a new evaluation record"""
    evaluation = Evaluation(
        llm_output_id=llm_output_id,
        ground_truth=ground_truth,
        whisper_wer=whisper_wer,
        llm_wer=llm_wer,
        llm_rag_wer=llm_rag_wer,
        llm_hematology_wer=llm_hematology_wer,
        llm_agent_wer=llm_agent_wer
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
    llm_agent_wer: Optional[float] = None
):
    """Update WER values for an evaluation"""
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
        db.commit()
        db.refresh(evaluation)
    return evaluation


def get_all_evaluations(db: Session) -> List[Evaluation]:
    """Get all evaluations"""
    return db.query(Evaluation).all()

