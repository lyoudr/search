import os
from sqlalchemy.orm import Session
from openai import OpenAI
from opencc import OpenCC

from app.repositories import (
    audio_file_repository,
    transcription_repository
)

client = OpenAI()
cc = OpenCC('s2t')

def speech_to_text(audio_path: str, engine: str = "whisper-1") -> str:
    """Convert audio file to text using Whisper"""
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=engine,  # whisper-1
            file=audio_file,
            language="zh"  # adjust as needed
        )
        simplified_text = transcript.text
        traditional_text = cc.convert(simplified_text)
        print(f"Transcription for {audio_path}: {traditional_text}")
        return traditional_text


def whisper_to_text(db: Session, input_dir: str, engine: str = "whisper-1"):
    """
    Converts all audio files in the input directory to text and creates AudioFile and Transcription records.
    """
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith((".wav", ".mp3", ".m4a")):  # Add formats as needed
                file_path = os.path.join(root, file_name)
                try:
                    # Check if audio file already exists
                    existing_audio_file = audio_file_repository.get_audio_file_by_path(db, file_path)
                    if existing_audio_file:
                        # Check if transcription already exists for this audio file
                        existing_transcriptions = transcription_repository.get_transcriptions_by_audio_file(
                            db, existing_audio_file.id
                        )
                        if existing_transcriptions:
                            print(f"⏭️  Skipping {file_path} - transcription already exists")
                            continue
                        audio_file_id = existing_audio_file.id
                    else:
                        # Create new audio file record
                        audio_file = audio_file_repository.create_audio_file(
                            db=db,
                            file_path=file_path,
                            language="zh"
                        )
                        audio_file_id = audio_file.id
                        print(f"✅ Created audio file record for {file_path} with ID {audio_file_id}")

                    # Transcribe audio
                    whisper_text = speech_to_text(file_path, engine)

                    # Create transcription record
                    transcription = transcription_repository.create_transcription(
                        db=db,
                        audio_file_id=audio_file_id,
                        engine=engine,
                        text=whisper_text
                    )
                    print(f"✅ Created transcription for {file_path} with ID {transcription.id}")
                except Exception as e:
                    print(f"❌ Failed to process {file_path}: {e}")