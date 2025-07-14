import os
import webrtcvad
import wave
from pydub import AudioSegment, effects

# speaker = os.getenv("SPEAKER", "personal")

# input_dir = f"parse/{speaker}"
# output_dir = f"splited/{speaker}"

def read_wave(path):
    audio = AudioSegment.from_wav(path).set_channels(1).set_frame_rate(16000)
    # Normalize volume
    audio = effects.normalize(audio)
    raw_audio = audio.raw_data
    return raw_audio, 16000

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * frame_duration_ms / 1000) * 2  # 16-bit audio
    for i in range(0, len(audio), n):
        yield audio[i:i + n]

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
    chunk_length = 30 * 16000 * 2  # 30 seconds of 16kHz, 16-bit audio
    os.makedirs(out_dir, exist_ok=True)

    for i in range(0, len(audio_data), chunk_length):
        chunk = audio_data[i:i+chunk_length]
        if len(chunk) < 10000:  # skip very small chunks (probably silence)
            continue
        out_path = os.path.join(out_dir, f"{base_name}_part{i//chunk_length + 1}.wav")
        with wave.open(out_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(chunk)

def process_audio(path, out_dir):
    audio_data, sr = read_wave(path)
    voiced_data = b''.join(vad_collector(audio_data, sr))
    base_name = os.path.splitext(os.path.basename(path))[0]
    save_chunks(voiced_data, sr, out_dir, base_name)

def split_audio(input_dir: str, output_dir: str):
    for fname in os.listdir(input_dir):
        if fname.endswith(".wav"):
            print(f"Processing {fname}...")
            process_audio(os.path.join(input_dir, fname), output_dir)

