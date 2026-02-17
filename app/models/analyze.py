from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, TIMESTAMP, func
from sqlalchemy.orm import relationship
from datetime import date

from . import Base


# ~ ✅ Table 1: llm_models
# Stores metadata about LLMs
class LLMModel(Base):
    __tablename__ = "llm_models"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    provider = Column(String(50))  # openai / qwen / llama
    size = Column(String(50))  # 7B / 1.5B
    quantization = Column(String(50))  # 4bit / 8bit / fp16

    def __repr__(self):
        return f"<LLMModel(name={self.name})"


# ~ ✅ Table 2: audio_files
# One row per audio file
class AudioFile(Base):
    __tablename__ = "audio_files"

    id = Column(Integer, primary_key=True)
    file_path = Column(String(255), nullable=False, unique=True)
    duration_sec = Column(Float)
    language = Column(String(10))  # zh / en / zh-TW

    def __repr__(self):
        return f"<AudioFile(path={self.file_path})>"


# ~ ✅ Table 3: transcriptions
# Stores whisper output
class Transcription(Base):
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True)
    audio_file_id = Column(Integer, ForeignKey("audio_files.id"), nullable=False)

    engine = Column(String(50))  # whisper-large-v3
    text = Column(Text, nullable=False)

    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    audio_file = relationship("AudioFile", backref="transcriptions")


# ~ ✅ Table 4: llm_outputs
# Stores LLM-corrected text
class LLMOutput(Base):
    __tablename__ = "llm_outputs"

    id = Column(Integer, primary_key=True)
    transcription_id = Column(Integer, ForeignKey("transcriptions.id"), nullable=False)
    llm_model_id = Column(Integer, ForeignKey("llm_models.id"), nullable=False)

    prompt_version = Column(String(50))  # v1 / v2 / medical_v3
    text = Column(Text, nullable=True)  # Direct LLM correction (without RAG)
    text_with_rag = Column(
        Text, nullable=True
    )  # LLM correction with RAG (using medical documents)
    text_with_hematology = Column(
        Text, nullable=True
    )  # LLM correction with Hematology Dictionary RAG
    text_agent = Column(
        Text, nullable=True
    )  # LLM correction using agent-based approach

    text_with_google = Column(Text, nullable=True)  # Cloud service tools
    text_with_aws = Column(Text, nullable=True)
    text_with_dr_ai = Column(Text, nullable=True)

    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    transcription = relationship("Transcription", backref="llm_outputs")
    llm_model = relationship("LLMModel")


# ~ ✅ Table 5: evaluations
# Stores metrics (WER, CER, etc.)
class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True)
    llm_output_id = Column(Integer, ForeignKey("llm_outputs.id"), nullable=False)

    ground_truth = Column(Text, nullable=True)

    whisper_wer = Column(Float)
    llm_wer = Column(Float)  # WER for direct LLM correction (without RAG)
    llm_rag_wer = Column(Float)  # WER for LLM correction with RAG
    llm_hematology_wer = Column(
        Float
    )  # WER for LLM correction with Hematology Dictionary RAG
    llm_agent_wer = Column(Float)  # WER for LLM correction using agent-based approach
    google_wer = Column(Float)  # WER for Google
    aws_wer = Column(Float)  # WER for AWS
    dr_ai_wer = Column(Float)

    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    llm_output = relationship("LLMOutput", backref="evaluation")


# ~ ✅ Table 6: wer_statistics
# Stores aggregated WER statistics across all evaluations
class WerStatistics(Base):
    __tablename__ = "wer_statistics"

    id = Column(Integer, primary_key=True)

    # Total counts
    total_evaluations = Column(
        Integer, default=0
    )  # Total number of evaluations with ground truth

    # Average WER values
    avg_whisper_wer = Column(Float)  # Average whisper_wer
    avg_llm_wer = Column(Float)  # Average llm_wer
    avg_llm_rag_wer = Column(Float)  # Average llm_rag_wer
    avg_llm_hematology_wer = Column(Float)  # Average llm_hematology_wer
    avg_llm_agent_wer = Column(Float)  # Average llm_agent_wer
    avg_google_wer = Column(Float)  # Average google_wer
    avg_aws_wer = Column(Float)  # Average aws_wer
    avg_dr_ai_wer = Column(Float)  # Average dr_ai_wer

    # Counts for each method (how many evaluations have this WER calculated)
    count_whisper_wer = Column(
        Integer, default=0
    )  # Count of evaluations with whisper_wer
    count_llm_wer = Column(Integer, default=0)  # Count of evaluations with llm_wer
    count_llm_rag_wer = Column(
        Integer, default=0
    )  # Count of evaluations with llm_rag_wer
    count_llm_hematology_wer = Column(
        Integer, default=0
    )  # Count of evaluations with llm_hematology_wer
    count_llm_agent_wer = Column(
        Integer, default=0
    )  # Count of evaluations with llm_agent_wer

    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
