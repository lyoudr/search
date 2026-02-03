from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# LLMModel schemas
class LLMModelBase(BaseModel):
    name: str
    provider: Optional[str] = None
    size: Optional[str] = None
    quantization: Optional[str] = None


class LLMModelCreate(LLMModelBase):
    pass


class LLMModelResponse(LLMModelBase):
    id: int

    class Config:
        from_attributes = True


# AudioFile schemas
class AudioFileBase(BaseModel):
    file_path: str
    duration_sec: Optional[float] = None
    language: Optional[str] = None


class AudioFileCreate(AudioFileBase):
    pass


class AudioFileResponse(AudioFileBase):
    id: int

    class Config:
        from_attributes = True


# Transcription schemas
class TranscriptionBase(BaseModel):
    audio_file_id: int
    engine: Optional[str] = None
    text: str


class TranscriptionCreate(TranscriptionBase):
    pass


class TranscriptionResponse(TranscriptionBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# LLMOutput schemas
class LLMOutputBase(BaseModel):
    transcription_id: int
    llm_model_id: int
    prompt_version: Optional[str] = None
    text: Optional[str] = None  # Direct LLM correction (without RAG)
    text_with_rag: Optional[str] = None  # LLM correction with RAG (using medical documents)
    text_with_hematology: Optional[str] = None  # LLM correction with Hematology Dictionary RAG
    text_agent: Optional[str] = None  # LLM correction using agent-based approach


class LLMOutputCreate(LLMOutputBase):
    pass


class LLMOutputResponse(LLMOutputBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Evaluation schemas
class EvaluationBase(BaseModel):
    llm_output_id: int
    ground_truth: Optional[str] = None
    whisper_wer: Optional[float] = None
    llm_wer: Optional[float] = None  # WER for direct LLM correction (without RAG)
    llm_rag_wer: Optional[float] = None  # WER for LLM correction with RAG
    llm_hematology_wer: Optional[float] = None  # WER for LLM correction with Hematology Dictionary RAG
    llm_agent_wer: Optional[float] = None  # WER for LLM correction using agent-based approach


class EvaluationCreate(EvaluationBase):
    pass


class EvaluationResponse(EvaluationBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


# Legacy AnalyzeBase for backward compatibility (if needed)
class AnalyzeBase(BaseModel):
    file_path: str
    whisper_text: Optional[str] = None
    llm_text: Optional[str] = None
    ground_truth: Optional[str] = None
