from sqlalchemy import Column, Integer, String, Text, Float
from . import Base 

class Analyze(Base):
    __tablename__ = 'analyze'

    id = Column(Integer, primary_key=True, autoincrement=True)
    file_path = Column(String(255), nullable=False, unique=True)
    whisper_text = Column(Text, nullable=True)
    llm_text = Column(Text, nullable=True)
    ground_truth = Column(Text, nullable=True)
    whisper_wer = Column(Float, nullable=True) # Word Error Rate for Whisper output
    llm_wer = Column(Float, nullable=True) # Word Error Rate for LLM output

    def __repr__(self):
        return f"<Analyze(id={self.id}, file_path={self.file_path})>"