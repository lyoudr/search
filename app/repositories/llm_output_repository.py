from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.analyze import LLMOutput

def create_llm_output(db: Session, transcription_id: int, llm_model_id: int,
                      prompt_version: Optional[str], text: Optional[str] = None,
                      text_with_rag: Optional[str] = None,
                      text_with_mts: Optional[str] = None) -> LLMOutput:
    """
    Create a new LLM output record.
    
    :param db: Database session
    :param transcription_id: Transcription ID
    :param llm_model_id: LLM model ID
    :param prompt_version: Prompt version
    :param text: Direct LLM correction (without RAG)
    :param text_with_rag: LLM correction with RAG (using medical documents)
    :param text_with_mts: LLM correction with MTSamples RAG
    """
    llm_output = LLMOutput(
        transcription_id=transcription_id,
        llm_model_id=llm_model_id,
        prompt_version=prompt_version,
        text=text,
        text_with_rag=text_with_rag,
        text_with_mts=text_with_mts
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


def get_llm_output_by_transcription_and_model(
    db: Session, 
    transcription_id: int, 
    llm_model_id: int
) -> Optional[LLMOutput]:
    """Get LLM output for a specific transcription and model"""
    return db.query(LLMOutput).filter(
        LLMOutput.transcription_id == transcription_id,
        LLMOutput.llm_model_id == llm_model_id
    ).first()


def update_llm_output(
    db: Session,
    llm_output_id: int,
    text: Optional[str] = None,
    text_with_rag: Optional[str] = None,
    text_with_mts: Optional[str] = None,
    prompt_version: Optional[str] = None
) -> LLMOutput:
    """Update an existing LLM output record"""
    llm_output = db.query(LLMOutput).filter(LLMOutput.id == llm_output_id).first()
    if not llm_output:
        raise ValueError(f"LLM output {llm_output_id} not found")
    
    if text is not None:
        llm_output.text = text
    if text_with_rag is not None:
        llm_output.text_with_rag = text_with_rag
    if text_with_mts is not None:
        llm_output.text_with_mts = text_with_mts
    if prompt_version is not None:
        llm_output.prompt_version = prompt_version
    
    db.commit()
    db.refresh(llm_output)
    return llm_output


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

