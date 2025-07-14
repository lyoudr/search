from pydantic import BaseModel 
from typing import Optional 

class AnalyzeBase(BaseModel):
    file_path: str 
    whisper_text: Optional[str] = None 
    llm_text: Optional[str] = None
    ground_truth: Optional[str] = None