from google.cloud import storage 
import os 

# client = storage.Client()
# bucket_name = "train-0342"
# bucket = client.bucket(bucket_name)

# # Create local folders if they don't exist
# os.makedirs("./data/audio", exist_ok=True)
# os.makedirs("./data/text", exist_ok=True)

# # Download audio and text files
# for blob in bucket.list_blobs(prefix="wav"):
#     if blob.name.endswith(".wav"):
#         print("blob.name is ->", blob.name)
#         blob.download_to_filename(f"./data/audio/{os.path.basename(blob.name)}")

# for blob in bucket.list_blobs(prefix="text"):
#     if blob.name.endswith(".txt"):
#         blob.download_to_filename(f"./data/text/{os.path.basename(blob.name)}")

# ------------------------------------------------------------------------------
# import os
# import json

# audio_dir = "./data/audio"
# text_dir = "./data/text"
# output_file = "train.jsonl"

# with open(output_file, "w", encoding="utf-8") as out_f:
#     for audio_file in os.listdir(audio_dir):
#         if audio_file.endswith(".wav"):
#             basename = os.path.splitext(audio_file)[0]  # e.g., "record_1"
#             audio_path = os.path.join(audio_dir, audio_file)
#             text_path = os.path.join(text_dir, f"{basename}.txt")
            
#             if os.path.exists(text_path):
#                 with open(text_path, "r", encoding="utf-8") as f:
#                     transcript = f.read().strip()

#                 json_line = {
#                     "audio_filepath": audio_path,
#                     "text": transcript
#                 }
#                 out_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
#             else:
#                 print(f"Warning: No matching text for {audio_file}")

# -------------------------------- Start Training ------------------------------
# """ 
# Below are the step-by-step instructions to fine-tune Whisper using Hugging Face's transformers and datasets
# """

# import torch 
# from datasets import load_dataset, Audio 
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments

# # Load dataset from JSONL
# dataset = load_dataset("json", data_files={"train": "train.jsonl"}, split="train")
# dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

# # Load processor & model 
# model_name = "openai/whisper-small"
# processor = WhisperProcessor.from_pretrained(model_name)
# model = WhisperForConditionalGeneration.from_pretrained(model_name)

# # Preprocess data
# def preprocess(example):
#     audio = example["audio_filepath"]
#     input_features = processor.feature_extractor(
#         audio["array"], 
#         sampling_rate=16000,
#         return_tensors="pt",
#         padding=True
#     ).input_features[0] # converts the raw waveform (audio["array"]) into input features expected by Whisper.
#     labels = processor.tokenizer(
#         example["text"],
#         padding = "max_length",
#         truncation=True,
#         return_tensors="pt"
#     ).input_ids # tokenizes the references transcript (example["text"]) into integer tooken IDs that the model will try to predict.
#     return {"input_features": input_features, "labels": labels} # final format for Whisper model training (features → tokens).

# dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# # Training arguments
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./whisper-finetune",
#     per_device_train_batch_size=4,        # How many samples per GPU/CPU
#     gradient_accumulation_steps=2,        # Accumulate gradients to simulate larger batches 
#     learning_rate=1e-5,                   # Initial learning rate for optimizer
#     warmup_steps=1,                       # Gradually increase learning rate for first 100 steps
#     max_steps=1,                          # Total number of training steps
#     save_steps=1,                         # Save model every 500 steps
#     logging_steps=50,                     # Log metrics every 50 steps
#     fp16=torch.cuda.is_available(),       # Use half-precision on GPU (faster training, less memory)
#     report_to="none",                     # Disable reporting to tools like WandB
# )


# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     tokenizer=processor.feature_extractor,
# )

# # Start training
# trainer.train()

# ------------------------------ Upload model to Hugging Face -------------------------
# from huggingface_hub import HfApi, HfFolder, snapshot_download, create_repo, upload_folder 
# from transformers import WhisperForConditionalGeneration, WhisperProcessor

# repo_name = "whisper-finetuned-ann"
# repo_id = f"Ann5432/whisper"

# api = HfApi()
# api.create_repo(
#     token="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", 
#     repo_id=repo_id, 
#     private=False
# )


# # Push model and processor
# upload_folder(
#     folder_path="./whisper-finetune",
#     repo_id=repo_id,
#     repo_type="model"
# )

# print(f"✅ Uploaded to https://huggingface.co/{repo_id}")

# ----------------------------------- Call Model API -------------------------- 
import requests 

API_URL = "https://router.huggingface.co/fal-ai/fal-ai/whisper"
headers = {"Authorization": "Bearer xxx"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read() 
        response = requests.post(API_URL, headers={"Content-Type": "audio/flac", **headers}, data=data)
        return response.json()

filename = "/mnt/c/Users/EF11405237/Code/search/data/audio/record_1.wav"
output = query(filename)
print("text is ->", output)
