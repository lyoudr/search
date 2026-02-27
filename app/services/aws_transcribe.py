"""
Amazon Transcribe integration.
Uploads audio to S3, runs a transcription job, and returns the transcript
for storing in LLMOutput.text_with_aws.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import boto3
import urllib.request


def transcribe_audio(
    audio_path: str,
    language_code: str = "zh-TW",
    media_format: str = "wav",
    region_name: str = "us-east-1",
    bucket: Optional[str] = None,
) -> str:
    """
    Transcribe audio using Amazon Transcribe (async job: upload to S3, start job, poll, return transcript).

    :param audio_path: Path to audio file (WAV, MP3, etc. per media_format).
    :param language_code: Language code (e.g. "zh-TW", "en-US").
    :param media_format: Media format: wav, mp3, mp4, flac, ogg, amr, webm, m4a.
    :param region_name: AWS region for Transcribe and S3.
    :param bucket: S3 bucket name (required; audio is uploaded here for Transcribe to read).
    :return: Full transcript text.
    :raises FileNotFoundError: If audio_path does not exist.
    :raises ValueError: If bucket is missing or job fails.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not bucket:
        raise ValueError("AWS_S3_BUCKET is required for Amazon Transcribe")

    s3 = boto3.client("s3", region_name=region_name)
    transcribe = boto3.client("transcribe", region_name=region_name)

    job_name = f"transcribe-{uuid.uuid4().hex[:12]}"
    upload_key = f"transcribe-input/{job_name}.{media_format}"

    s3.upload_file(str(path), bucket, upload_key)

    media_uri = f"s3://{bucket}/{upload_key}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat=media_format,
        LanguageCode=language_code,
    )

    while True:
        job = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = job["TranscriptionJob"]["TranscriptionJobStatus"]
        if status == "COMPLETED":
            break
        if status == "FAILED":
            failure = job["TranscriptionJob"].get("FailureReason", "Unknown")
            raise ValueError(f"Transcription job failed: {failure}")
        time.sleep(2)

    transcript_file_uri = job["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    if transcript_file_uri.startswith("s3://"):
        parsed = urlparse(transcript_file_uri)
        bucket_name = parsed.netloc
        transcript_key = parsed.path.lstrip("/")
        obj = s3.get_object(Bucket=bucket_name, Key=transcript_key)
        data = json.load(obj["Body"])
    else:
        with urllib.request.urlopen(transcript_file_uri) as resp:
            data = json.load(resp)

    try:
        transcript = data["results"]["transcripts"][0]["transcript"]
    except (KeyError, IndexError):
        transcript = data.get("results", {}).get("transcripts", [{}])[0].get("transcript", "")

    # Optional: delete input from S3 and the job to avoid clutter
    try:
        s3.delete_object(Bucket=bucket, Key=upload_key)
    except Exception:
        pass
    try:
        transcribe.delete_transcription_job(TranscriptionJobName=job_name)
    except Exception:
        pass

    return (transcript or "").strip()
