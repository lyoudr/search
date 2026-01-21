from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.analyze import Transcription, AudioFile


def create_transcription(db: Session, audio_file_id: int, engine: str, text: str) -> Transcription:
    """Create a new transcription record"""
    transcription = Transcription(
        audio_file_id=audio_file_id,
        engine=engine,
        text=text
    )
    db.add(transcription)
    db.commit()
    db.refresh(transcription)
    return transcription


def get_transcription_by_id(db: Session, transcription_id: int) -> Optional[Transcription]:
    """Get transcription by ID"""
    return db.query(Transcription).filter(Transcription.id == transcription_id).first()


def get_transcriptions_by_audio_file(db: Session, audio_file_id: int) -> List[Transcription]:
    """Get all transcriptions for an audio file"""
    return db.query(Transcription).filter(Transcription.audio_file_id == audio_file_id).all()


def get_all_transcriptions(db: Session) -> List[Transcription]:
    """Get all transcriptions"""
    return db.query(Transcription).all()


def get_transcriptions_with_ground_truth(db: Session, limit: int = 10) -> List[dict]:
    """Get transcriptions with ground truth for CoT examples"""
    from app.models.analyze import LLMOutput, Evaluation
    
    # Join Transcription -> LLMOutput -> Evaluation to get ground truth
    transcriptions = db.query(Transcription).join(
        LLMOutput, Transcription.id == LLMOutput.transcription_id
    ).join(
        Evaluation, LLMOutput.id == Evaluation.llm_output_id
    ).filter(
        Evaluation.ground_truth.isnot(None)
    ).limit(limit).all()
    
    examples = []
    for transcription in transcriptions:
        # Get the first LLM output with ground truth for this transcription
        llm_outputs = db.query(LLMOutput).join(
            Evaluation, LLMOutput.id == Evaluation.llm_output_id
        ).filter(
            LLMOutput.transcription_id == transcription.id,
            Evaluation.ground_truth.isnot(None)
        ).first()
        
        if llm_outputs and llm_outputs.evaluation:
            examples.append({
                "input": transcription.text,
                "reasoning": "1. 補上漏掉的標點符號\n2. 修正文法詞彙錯誤\n",
                "output": llm_outputs.evaluation.ground_truth
            })
    
    return examples[:limit]

