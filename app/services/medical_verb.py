import torch 
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE 
from gtts import gTTS

# Step 1. Store correct pronunciations in FAISS
def save_goolge_translate(text):
    tts = gTTS(text, lang="en")
    tts.save(f"{text}.mp3")

# Step 2. Store pronunciation file in Vector Database
# Load model and processor
bundle = WAV2VEC2_BASE
model = bundle.get_model()

def get_audio_embedding(path):
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)
    
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)
        # Use the mean over time (first layer)
        embedding = features[0].mean(dim=1).squeeze()
        return embedding.numpy()

# Step 3: Store correct pronunciations in FAISS 
import faiss 
import numpy as np 

words = ["dicloxacillin", "segment", "Augmentin"]
paths = [
    "app/sources/reference/audio/dicloxacillin.mp3", 
    "app/sources/reference/audio/segment.mp3", 
    "app/sources/reference/audio/augmentin.mp3"
]


# Get embeddings
vectors = np.stack([get_audio_embedding(p) for p in paths])

# Create FAISS index
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)


# # Assume user_audio_path is your voice input
# query_vec = get_audio_embedding("user_voice.wav")
# D, I = index.search(np.expand_dims(query_vec, axis=0), k=1)

# # Print matched word
# print("Matched word:", words[I[0][0]])

