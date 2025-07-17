from airflow import DAG 
from airflow.operators.python import PythonOperator
from datetime import datetime
import os
import json
import torch
from google.cloud import storage
from datasets import load_dataset, Audio
from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
from huggingface_hub import HfApi, upload_folder

BUCKET_NAME = "train-0342"
MODEL_NAME = "openai/whisper-small"
LOCAL_AUDIO_DIR = "/tmp/audio"
LOCAL_TEXT_DIR = "/tmp/text"
JSONL_PATH = "/tmp/train.jsonl"
OUTPUT_DIR = "/tmp/whisper-finetune"
REPO_ID = ""  # Change this as needed
HF_TOKEN = ""  # Hugging Face Token


# Download audio and text files
def download_from_gcs():
    client = storage.Client()
    bucket = client.bucket()
    os.makedirs(LOCAL_AUDIO_DIR, exist_ok=True)
    os.makedirs(LOCAL_TEXT_DIR, exist_ok=True)

    # Download audio
    for blob in bucket.list_blobs(prefix="wav"):
        if blob.name.endswith(".wav"):
            local_path = os.path.join(LOCAL_AUDIO_DIR, os.path.basename(blob.name))
            blob.download_to_filename(local_path)

    # Download transcripts
    for blob in bucket.list_blobs(prefix="text"):
        if blob.name.endswith(".txt"):
            local_path = os.path.join(LOCAL_TEXT_DIR, os.path.basename(blob.name))
            blob.download_to_filename(local_path)


def generate_jsonl():
    with open(JSONL_PATH, "w", encoding="utf-8") as out_f:
        for audio_file in os.listdir(LOCAL_AUDIO_DIR):
            if audio_file.endswith(".wav"):
                basename = os.path.splitext(audio_file)[0]
                audio_path = os.path.join(LOCAL_AUDIO_DIR, audio_file)
                text_path = os.path.join(LOCAL_TEXT_DIR, f"{basename}.txt")
                if os.path.exists(text_path):
                    with open(text_path, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()
                    json_line = {
                        "audio_filepath": audio_path,
                        "text": transcript
                    }
                    out_f.write(json.dumps(json_line, ensure_ascii=False) + "\n")

def fine_tune_whisper():
    dataset = load_dataset("json", data_files={"train": JSONL_PATH}, split="train")
    dataset = dataset.cast_column("audio_filepath", Audio(sampling_rate=16000))

    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

    def preprocess(example):
        audio = example["audio_filepath"]
        input_features = processor.feature_extractor(
            audio["array"], sampling_rate=16000, return_tensors="pt", padding=True
        ).input_features[0]
        labels = processor.tokenizer(
            example["text"], padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids[0]
        return {"input_features": input_features, "labels": labels}

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=1,
        max_steps=1,
        save_steps=1,
        logging_steps=1,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

# ------------------------------ Upload model to Hugging Face -------------------------
def upload_to_huggingface():
    api = HfApi()
    if REPO_ID not in [repo.repo_id for repo in api.list_repos(token=HF_TOKEN)]:
        api.create_repo(token=HF_TOKEN, repo_id=REPO_ID, private=False)

    upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        token=HF_TOKEN
    )
    print(f"✅ Uploaded to https://huggingface.co/{REPO_ID}")

with DAG("whisper_fine_tune", schedule_interval="0 3 * * 1", catchup=False, default_args=default_args) as dag:
    download = PythonOperator(task_id="download_data", python_callable=download_from_gcs)
    preprocess = PythonOperator(task_id="generate_jsonl", python_callable=generate_jsonl)
    train = PythonOperator(task_id="train_model", python_callable=fine_tune_whisper)
    upload = PythonOperator(task_id="upload_model", python_callable=upload_to_huggingface)

    download >> preprocess >> train >> upload