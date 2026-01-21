from sqlalchemy.orm import Session
from typing import List, Optional
from app.models.analyze import AudioFile


def create_audio_file(db: Session, file_path: str, duration_sec: Optional[float] = None,
                      language: Optional[str] = None) -> AudioFile:
    """Create a new audio file record"""
    audio_file = AudioFile(
        file_path=file_path,
        duration_sec=duration_sec,
        language=language
    )
    db.add(audio_file)
    db.commit()
    db.refresh(audio_file)
    return audio_file


def get_audio_file_by_id(db: Session, audio_file_id: int) -> Optional[AudioFile]:
    """Get audio file by ID"""
    return db.query(AudioFile).filter(AudioFile.id == audio_file_id).first()


def get_audio_file_by_path(db: Session, file_path: str) -> Optional[AudioFile]:
    """Get audio file by file path"""
    return db.query(AudioFile).filter(AudioFile.file_path == file_path).first()


def get_all_audio_files(db: Session) -> List[AudioFile]:
    """Get all audio files"""
    return db.query(AudioFile).all()

