import os
import subprocess
import wave

import webrtcvad
from openai import OpenAI
from opencc import OpenCC
from pydub import AudioSegment, effects
from sqlalchemy.orm import Session

from app.ai_agents.agents.whisper_agent import WhisperAgent
from app.repositories import audio_file_repository, transcription_repository


INPUT_FORMATS = [".mp3", ".m4a", ".acc", ".flac", ".wav"]

client = OpenAI()
cc = OpenCC("s2t")


def convert_to_wav(input_dir: str, output_dir: str):
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}.wav")
        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            output_path,
        ]
        try:
            print(f"Converting {input_path} -> {output_path}...")
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"X failed to convert {filename}: {e}")


def read_wave(path):
    audio = AudioSegment.from_wav(path).set_channels(1).set_frame_rate(16000)
    audio = effects.normalize(audio)
    raw_audio = audio.raw_data
    return raw_audio, 16000


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * frame_duration_ms / 1000) * 2
    for i in range(0, len(audio), n):
        yield audio[i : i + n]


def vad_collector(audio, sample_rate, aggressiveness=2):
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(10, audio, sample_rate))
    voiced_frames = []
    frame_size = int(sample_rate * 10 / 1000) * 2
    for f in frames:
        if len(f) < frame_size:
            continue
        if vad.is_speech(f, sample_rate):
            voiced_frames.append(f)
    return voiced_frames


def save_chunks(audio_data, sample_rate, out_dir, base_name):
    chunk_length = 30 * 16000 * 2
    os.makedirs(out_dir, exist_ok=True)

    for i in range(0, len(audio_data), chunk_length):
        chunk = audio_data[i : i + chunk_length]
        if len(chunk) < 10000:
            continue
        out_path = os.path.join(out_dir, f"{base_name}_part{i // chunk_length + 1}_zh.wav")
        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(chunk)


def process_audio(path, out_dir):
    audio_data, sr = read_wave(path)
    voiced_data = b"".join(vad_collector(audio_data, sr))
    base_name = os.path.splitext(os.path.basename(path))[0]
    save_chunks(voiced_data, sr, out_dir, base_name)


def split_audio(input_dir: str, output_dir: str):
    for fname in os.listdir(input_dir):
        if fname.endswith(".wav"):
            print(f"Processing {fname}...")
            process_audio(os.path.join(input_dir, fname), output_dir)


def speech_to_text(audio_path: str, engine: str = "whisper-1") -> str:
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=engine,
            file=audio_file,
            language="zh",
        )
        simplified_text = transcript.text
        traditional_text = cc.convert(simplified_text)
        print(f"Transcription for {audio_path}: {traditional_text}")
        return traditional_text


def whisper_to_text(
    db: Session,
    input_dir: str,
    engine: str = "whisper-1",
    extraction_model: str = "gpt-5.2",
):
    whisper_agent = WhisperAgent(extraction_model=extraction_model)

    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith((".wav", ".mp3", ".m4a")):
                file_path = os.path.join(root, file_name)
                try:
                    existing_audio_file = audio_file_repository.get_audio_file_by_path(
                        db, file_path
                    )
                    if existing_audio_file:
                        existing_transcriptions = (
                            transcription_repository.get_transcriptions_by_audio_file(
                                db, existing_audio_file.id
                            )
                        )
                        if existing_transcriptions:
                            print(f"⏭️  Skipping {file_path} - transcription already exists")
                            continue
                        audio_file_id = existing_audio_file.id
                    else:
                        audio_file = audio_file_repository.create_audio_file(
                            db=db,
                            file_path=file_path,
                            language="zh",
                        )
                        audio_file_id = audio_file.id
                        print(
                            f"✅ Created audio file record for {file_path} "
                            f"with ID {audio_file_id}"
                        )

                    whisper_text = speech_to_text(file_path, engine)
                    transcription = transcription_repository.create_transcription(
                        db=db,
                        audio_file_id=audio_file_id,
                        engine=engine,
                        text=whisper_text,
                    )
                    print(f"✅ Created transcription for {file_path} with ID {transcription.id}")

                    try:
                        result = whisper_agent.process_transcription(
                            transcription_id=transcription.id,
                            audio_file_id=audio_file_id,
                            engine=engine,
                            text=whisper_text,
                        )
                        print(
                            f"✅ WhisperAgent done for transcription {transcription.id}: "
                            f"terms={result['term_count']}, chunks={result['chunk_count']}, "
                            f"term_vectors={result['term_vectors_upserted']}, "
                            f"chunk_vectors={result['chunk_vectors_upserted']}"
                        )
                    except Exception as e:
                        print(
                            f"⚠️  WhisperAgent post-process failed for transcription "
                            f"{transcription.id}: {e}"
                        )
                except Exception as e:
                    print(f"❌ Failed to process {file_path}: {e}")
