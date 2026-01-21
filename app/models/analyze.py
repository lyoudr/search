from sqlalchemy import (
    Column, 
    Integer, 
    String, 
    Text, 
    Float, 
    ForeignKey, 
    TIMESTAMP,
    func
)
from sqlalchemy.orm import relationship

from . import Base 

# ~ ✅ Table 1: llm_models
# Stores metadata about LLMs
class LLMModel(Base):
    __tablename__ = "llm_models"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    provider = Column(String(50))          # openai / qwen / llama
    size = Column(String(50))              # 7B / 1.5B
    quantization = Column(String(50))      # 4bit / 8bit / fp16

    def __repr__(self):
        return f"<LLMModel(name={self.name})"


# ~ ✅ Table 2: audio_files
# One row per audio file   
class AudioFile(Base):
    __tablename__ = "audio_files"

    id = Column(Integer, primary_key=True)
    file_path = Column(String(255), nullable=False, unique=True)
    duration_sec = Column(Float)
    language = Column(String(10))          # zh / en / zh-TW

    def __repr__(self):
        return f"<AudioFile(path={self.file_path})>"


# ~ ✅ Table 3: transcriptions
# Stores whisper output
class Transcription(Base):
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True)
    audio_file_id = Column(Integer, ForeignKey("audio_files.id"), nullable=False)

    engine = Column(String(50))             # whisper-large-v3
    text = Column(Text, nullable=False)

    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    audio_file = relationship("AudioFile", backref="transcriptions")

# ~ ✅ Table 4: llm_outputs
# Stores LLM-corrected text
class LLMOutput(Base):
    __tablename__ = "llm_outputs"

    id = Column(Integer, primary_key=True)
    transcription_id = Column(Integer, ForeignKey("transcriptions.id"), nullable=False)
    llm_model_id = Column(Integer, ForeignKey("llm_models.id"), nullable=False)

    prompt_version = Column(String(50))     # v1 / v2 / medical_v3
    text = Column(Text, nullable=False)

    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

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
    llm_wer = Column(Float)

    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now())

    llm_output = relationship("LLMOutput", backref="evaluation")