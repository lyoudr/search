from pydantic import BaseModel 

class ParseRequest(BaseModel):
    input_dir: str
    output_dir: str 

class ParseResponse(BaseModel):
    status: str


class AnalyzeRequest(BaseModel):
    input_dir: str

class WordErrorRateResponse(BaseModel):
    status: str 

class LLMCorrectResponse(BaseModel):
    status: str 
    