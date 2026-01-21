import os
from sqlalchemy.orm import Session
from pathlib import Path
from typing import List, Optional

from app.repositories import audio_file_repository
from app.config.settings import get_settings

settings = get_settings()

# Supported audio file extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'}


def get_audio_duration(file_path: str) -> Optional[float]:
    """
    Get audio file duration in seconds.
    Requires ffmpeg or similar tools. Returns None if not available.
    """
    try:
        import subprocess
        import json
        
        # Use ffprobe to get duration
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            file_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return duration
    except (subprocess.CalledProcessError, FileNotFoundError, KeyError, ValueError) as e:
        print(f"⚠️  Could not get duration for {file_path}: {e}")
        return None


def detect_language_from_filename(file_path: str) -> Optional[str]:
    """
    Try to detect language from filename.
    Returns language code (e.g., 'zh', 'en', 'zh-TW') or None.
    """
    filename = Path(file_path).stem.lower()
    
    # Simple heuristic - can be enhanced
    if 'zh' in filename or 'chinese' in filename:
        if 'tw' in filename or 'trad' in filename:
            return 'zh-TW'
        return 'zh'
    elif 'en' in filename or 'english' in filename:
        return 'en'
    
    return None


def scan_audio_files_in_directory(directory: str, recursive: bool = True) -> List[str]:
    """
    Scan directory for audio files and return their full paths.
    
    :param directory: Directory to scan
    :param recursive: Whether to scan subdirectories
    :return: List of audio file paths
    """
    audio_files = []
    
    if not os.path.exists(directory):
        print(f"❌ Directory does not exist: {directory}")
        return audio_files
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                file_ext = Path(file_path).suffix.lower()
                if file_ext in AUDIO_EXTENSIONS:
                    audio_files.append(file_path)
    else:
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                file_ext = Path(file_path).suffix.lower()
                if file_ext in AUDIO_EXTENSIONS:
                    audio_files.append(file_path)
    
    return audio_files


def import_audio_files_from_splited(db: Session, splited_dir: Optional[str] = None, 
                                    get_duration: bool = True) -> dict:
    """
    Import all audio files from splited directory into audio_files table.
    
    :param db: Database session
    :param splited_dir: Path to splited directory (relative to SOURCE_DIR if not absolute)
    :param get_duration: Whether to get audio file duration (requires ffmpeg)
    :return: Dictionary with import statistics
    """
    # Default to splited folder in SOURCE_DIR
    if splited_dir is None:
        splited_dir = "splited"
    
    # Construct full path
    if os.path.isabs(splited_dir):
        full_path = splited_dir
    else:
        full_path = os.path.join(settings.SOURCE_DIR, splited_dir)
    
    print(f"📁 Scanning directory: {full_path}")
    
    # Scan for audio files
    audio_files = scan_audio_files_in_directory(full_path, recursive=True)
    print(f"🎵 Found {len(audio_files)} audio file(s)")
    
    stats = {
        "total_found": len(audio_files),
        "created": 0,
        "skipped": 0,
        "errors": 0,
        "errors_list": []
    }
    
    # Import each file
    for file_path in audio_files:
        try:
            # Check if file already exists in database
            existing_file = audio_file_repository.get_audio_file_by_path(db, file_path)
            if existing_file:
                print(f"⏭️  Skipping {file_path} - already in database (ID: {existing_file.id})")
                stats["skipped"] += 1
                continue
            
            # Get audio file metadata
            duration_sec = None
            if get_duration:
                duration_sec = get_audio_duration(file_path)
            
            language = detect_language_from_filename(file_path)
            
            # Create audio file record
            audio_file = audio_file_repository.create_audio_file(
                db=db,
                file_path=file_path,
                duration_sec=duration_sec,
                language=language
            )
            
            print(f"✅ Created audio file record (ID: {audio_file.id}) - {file_path}")
            if duration_sec:
                print(f"   Duration: {duration_sec:.2f}s")
            if language:
                print(f"   Language: {language}")
            
            stats["created"] += 1
            
        except Exception as e:
            error_msg = f"❌ Failed to import {file_path}: {e}"
            print(error_msg)
            stats["errors"] += 1
            stats["errors_list"].append({"file": file_path, "error": str(e)})
    
    print(f"\n📊 Import Summary:")
    print(f"   Total found: {stats['total_found']}")
    print(f"   Created: {stats['created']}")
    print(f"   Skipped: {stats['skipped']}")
    print(f"   Errors: {stats['errors']}")
    
    return stats

