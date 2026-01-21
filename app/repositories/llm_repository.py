from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.analyze import LLMModel


def create_llm_model(db: Session, name: str, provider: Optional[str] = None, 
                     size: Optional[str] = None, quantization: Optional[str] = None) -> LLMModel:
    """Create a new LLM model record"""
    llm_model = LLMModel(
        name=name,
        provider=provider,
        size=size,
        quantization=quantization
    )
    db.add(llm_model)
    db.commit()
    db.refresh(llm_model)
    return llm_model


def get_llm_model_by_id(db: Session, llm_model_id: int) -> Optional[LLMModel]:
    """Get LLM model by ID"""
    return db.query(LLMModel).filter(LLMModel.id == llm_model_id).first()


def get_llm_model_by_name(db: Session, name: str) -> Optional[LLMModel]:
    """Get LLM model by name"""
    return db.query(LLMModel).filter(LLMModel.name == name).first()


def get_all_llm_models(db: Session) -> List[LLMModel]:
    """Get all LLM models"""
    return db.query(LLMModel).all()

