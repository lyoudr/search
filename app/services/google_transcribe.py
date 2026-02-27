"""
Google Cloud Speech-to-Text integration.
Transcribes audio and returns the transcript for storing in LLMOutput.text_with_google.
"""

from google.cloud import speech


def transcribe_audio(
    audio_path: str,
    language_code: str = "zh-TW",
    sample_rate_hertz: int = 16000,
    encoding: speech.RecognitionConfig.AudioEncoding = speech.RecognitionConfig.AudioEncoding.LINEAR16,
) -> str:
    """
    Transcribe audio file using Google Cloud Speech-to-Text.

    :param audio_path: Path to WAV file (LINEAR16, mono).
    :param language_code: Language code (e.g. "zh-TW", "en-US").
    :param sample_rate_hertz: Sample rate of the WAV; must match the file (default 16000).
    :param encoding: Audio encoding (default LINEAR16).
    :return: Concatenated transcript text from all results.
    :raises FileNotFoundError: If audio_path does not exist.
    :raises Exception: On API or credential errors.
    """

    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=encoding,
        sample_rate_hertz=sample_rate_hertz,
        language_code=language_code,
    )
    response = client.recognize(config=config, audio=audio)

    parts = []
    for result in response.results:
        if result.alternatives:
            parts.append(result.alternatives[0].transcript)
    return " ".join(parts).strip()
