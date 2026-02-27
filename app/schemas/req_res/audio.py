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


class GoogleTranscribeRequest(BaseModel):
    """Optional body for Google Speech-to-Text. Audio comes from LLM output's transcription audio file."""

    language_code: str = "zh-TW"
    sample_rate_hertz: int = 16000


class AwsTranscribeRequest(BaseModel):
    """Optional body for Amazon Transcribe. Audio is uploaded to S3 then transcribed."""

    language_code: str = "zh-TW"
    media_format: str = "wav"
