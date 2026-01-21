from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.analyze import LLMOutput


def create_llm_output(db: Session, transcription_id: int, llm_model_id: int,
                      prompt_version: Optional[str], text: str) -> LLMOutput:
    """Create a new LLM output record"""
    llm_output = LLMOutput(
        transcription_id=transcription_id,
        llm_model_id=llm_model_id,
        prompt_version=prompt_version,
        text=text
    )
    db.add(llm_output)
    db.commit()
    db.refresh(llm_output)
    return llm_output


def get_llm_output_by_id(db: Session, llm_output_id: int) -> Optional[LLMOutput]:
    """Get LLM output by ID"""
    return db.query(LLMOutput).filter(LLMOutput.id == llm_output_id).first()


def get_llm_outputs_by_transcription(db: Session, transcription_id: int) -> List[LLMOutput]:
    """Get all LLM outputs for a transcription"""
    return db.query(LLMOutput).filter(LLMOutput.transcription_id == transcription_id).all()


def get_llm_outputs_with_ground_truth(db: Session) -> List[LLMOutput]:
    """Get LLM outputs that have ground truth for WER calculation"""
    from app.models.analyze import Evaluation
    
    return db.query(LLMOutput).join(
        Evaluation, LLMOutput.id == Evaluation.llm_output_id
    ).filter(
        Evaluation.ground_truth.isnot(None)
    ).all()


def get_all_llm_outputs(db: Session) -> List[LLMOutput]:
    """Get all LLM outputs"""
    return db.query(LLMOutput).all()

