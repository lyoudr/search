from pydantic import BaseModel

class MedicalQuestionRequest(BaseModel):
    question: str