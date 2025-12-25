import os 
import subprocess

# Accepted autio input formats
input_formats = [".mp3", ".m4a", ".acc", ".flac", ".wav"]

def convert_to_wav(input_dir: str, output_dir: str):
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        name, _ = os.path.splitext(filename)
        ouput_path = os.path.join(output_dir, f"{name}.wav")

        # FFmpeg command: convert to 16kHz, mono WAV (ideal for speech recognition)
        command = [
            "ffmpeg",
            "-i", input_path,
            "-ar", "16000",       # Set audio sample rate to 16kHz
            "-ac", "1",           # Set number of audio channels to 1 (mono)
            "-c:a", "pcm_s16le",  # Use PCM signed 16-bit little-endian format
            ouput_path
        ]
        try:
            print(f"Converting {input_path} -> {ouput_path}...")
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"X failed to convert {filename}: {e}")