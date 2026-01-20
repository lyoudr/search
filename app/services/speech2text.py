import os
from sqlalchemy.orm import Session
from openai import OpenAI
from opencc import OpenCC

from app.schemas.analyze import AnalyzeBase
from app.repositories.audio_repository import (
    create_analyze_record,
    check_file_exist,
)

client = OpenAI()
cc = OpenCC('s2t')

def speech_to_text(audio_path: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", # * whisper-1
            file=audio_file,
            language="zh"  # adjust as needed
        )
        simplified_text = transcript.text
        traditional_text = cc.convert(simplified_text)
        print(f"Transcription for {audio_path}: {traditional_text}")
        return traditional_text

def whisper_to_text(db: Session, input_dir: str):
    """
    Converts all audio files in the input directory to text and creates Analyze records.
    """
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith((".wav", ".mp3", ".m4a")):  # Add formats as needed
                file_path = os.path.join(root, file_name)
                try:
                    existed = check_file_exist(db, file_path)
                    if existed:
                        continue
                    whisper_text = speech_to_text(file_path)

                    analyze_payload = AnalyzeBase(
                        file_path=file_path,
                        whisper_text=whisper_text,
                        llm_text=None,
                        ground_truth=None  # Optional, update if you have it
                    )

                    record = create_analyze_record(db=db, payload=analyze_payload)
                    print(f"✅ Created record for {file_path} with ID {record.id}")
                except Exception as e:
                    print(f"❌ Failed to process {file_path}: {e}")