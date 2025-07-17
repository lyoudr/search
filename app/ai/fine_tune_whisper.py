import os
import pandas as pd
from datasets import Dataset, Audio, load_dataset, Features, Value
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import shutil
from tqdm.auto import tqdm # 導入 tqdm 以顯示進度條

# --- PEFT 相關引入 ---
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model, prepare_model_for_kbit_training
# ---

# --- 0. 環境設定與裝置偵測 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置: {device}")
if device == "cuda":
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print(f"PyTorch CUDA 可用: {torch.cuda.is_available()}")
    print(f"GPU 數量: {torch.cuda.device_count()}")
    print(f"當前 GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# --- 1. 設定您想配置的參數 ---
ORIGINAL_TRAIN_CSV_PATH = "/home/pd/wang/sumdata/train_0617.csv"
ORIGINAL_EVAL_CSV_PATH = "/home/pd/wang/sumdata/eval.csv"

TRAIN_PARQUET_PATH = "/home/pd/wang/智海/parquet/train.parquet"
EVAL_PARQUET_PATH = "/home/pd/wang/智海/parquet/eval.parquet"

AUDIO_COLUMN_NAME = "wav" # 指定音檔的路徑
TEXT_COLUMN_NAME = "label" # 實際該音檔的文字

MODEL_NAME = "openai/whisper-large-v3"
OUTPUT_DIR = "./whisper-v3-ft-500"

# --- 測試模式開關 ---
TEST_MODE = False
TEST_MODE_SAMPLES = 10
# ==============================

# --- 調整後的參數 ---
MAX_STEPS = 500
LEARNING_RATE = 1e-3
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
EVAL_STEPS = 100
SAVE_STEPS = 100
WARMUP_STEPS = 50
LOGGING_STEPS = 25

FP16_ENABLED = False
BF16_ENABLED = True
GRADIENT_CHECKPOINTING_ENABLED = False 
PUSH_TO_HUB = False
GENERATION_NUM_BEAMS = 2 

BATCH_PROCESS_SIZE = 1500
DATALOADER_NUM_WORKERS = 1
print(f"Dataloader workers: {DATALOADER_NUM_WORKERS}")


# --- LoRA 相關參數 ---
LORA_R = 32
LORA_ALPHA = 64 
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
# ==============================

# --- 2. 資料轉換與載入 ---
def convert_csv_to_parquet(csv_path: str, parquet_path: str, is_text_content_directly: bool = False):
    """
    將 CSV 檔案轉換為 Parquet 檔案。
    
    Args:
        csv_path (str): 原始 CSV 檔案的路徑。
        parquet_path (str): 輸出 Parquet 檔案的路徑。
        is_text_content_directly (bool): 如果為 True，表示 CSV 中的 TEXT_COLUMN_NAME 
                                         已經是逐字稿文本內容，無需再從 .txt 檔案讀取。
                                         如果為 False (預設)，則表示是 .txt 檔案路徑。
    """
    if not csv_path:
        print(f"警告: 未提供 CSV 路徑 ('{csv_path}')，跳過轉換為 Parquet。")
        return False

    if os.path.exists(parquet_path):
        print(f"'{parquet_path}' 已存在，強制刪除舊檔案以重新生成。")
        try:
            os.remove(parquet_path)
        except OSError as e:
            print(f"錯誤: 無法刪除舊的 Parquet 檔案 '{parquet_path}'。請檢查權限或檔案是否被佔用。詳情: {e}")
            return False

    print(f"正在從 '{csv_path}' 轉換為 '{parquet_path}'...")
    try:
        df = pd.read_csv(csv_path)
        required_columns = [AUDIO_COLUMN_NAME, TEXT_COLUMN_NAME]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"原始 CSV 檔案 '{csv_path}' 缺少必要的欄位: '{col}'。請檢查您的 CSV 檔案。")
        
        processed_records = []
        
        print(f"正在處理 {len(df)} 條數據...")
        for index, row in tqdm(df.iterrows(), total=len(df), desc=f"處理 {os.path.basename(csv_path)} 數據"):
            wav_path = row[AUDIO_COLUMN_NAME]
            label_value = str(row[TEXT_COLUMN_NAME]).strip() # 確保是字串並去除空白

            actual_text = ""
            if is_text_content_directly:
                # 如果標誌為 True，直接使用 label_value 作為文本內容
                actual_text = label_value
            else:
                # 否則，嘗試將 label_value 作為 .txt 檔案路徑讀取
                txt_file_path = label_value
                if txt_file_path.endswith('.txt') and os.path.exists(txt_file_path):
                    try:
                        with open(txt_file_path, 'r', encoding='utf-8') as f:
                            actual_text = f.read().strip()
                    except Exception as e:
                        print(f"警告: 讀取文本檔案 '{txt_file_path}' 時發生錯誤: {e}，該條數據的文本將為空。")
                        actual_text = ""
                else:
                    # 如果不是 .txt 結尾或檔案不存在，發出警告
                    print(f"警告: '{txt_file_path}' 不是有效的 .txt 檔案路徑或檔案不存在，該條數據的文本將為空。")
                    actual_text = ""
            
            processed_records.append({
                AUDIO_COLUMN_NAME: wav_path,
                TEXT_COLUMN_NAME: actual_text
            })
        
        df_processed = pd.DataFrame(processed_records)
        df_processed.to_parquet(parquet_path, index=False)
        print(f"'{csv_path}' 已成功轉換為 '{parquet_path}'。")
        print(f"Parquet 檔案包含 '{df_processed.columns.tolist()}' 欄位。")
        return True
    except FileNotFoundError:
        print(f"錯誤: 原始 CSV 檔案 '{csv_path}' 不存在，無法進行轉換。")
        return False
    except Exception as e:
        print(f"轉換 '{csv_path}' 到 Parquet 時發生錯誤: {e}")
        return False

# 訓練集：假設 label 欄位是 .txt 檔案路徑
train_parquet_generated = convert_csv_to_parquet(ORIGINAL_TRAIN_CSV_PATH, TRAIN_PARQUET_PATH, is_text_content_directly=False)
if not train_parquet_generated or not os.path.exists(TRAIN_PARQUET_PATH):
    print(f"致命錯誤: 訓練 Parquet 檔案 '{TRAIN_PARQUET_PATH}' 未能成功生成或不存在。程式終止。")
    exit()

eval_parquet_generated = False
if ORIGINAL_EVAL_CSV_PATH and os.path.exists(ORIGINAL_EVAL_CSV_PATH):
    # 評估集：明確指定 label 欄位就是逐字稿文本內容
    eval_parquet_generated = convert_csv_to_parquet(ORIGINAL_EVAL_CSV_PATH, EVAL_PARQUET_PATH, is_text_content_directly=True)
    if not eval_parquet_generated or not os.path.exists(EVAL_PARQUET_PATH):
        print(f"警告: 評估 Parquet 檔案 '{EVAL_PARQUET_PATH}' 未能成功生成或不存在。將不進行評估。")
else:
    print(f"警告: 原始評估 CSV 檔案 '{ORIGINAL_EVAL_CSV_PATH}' 不存在或路徑無效。將不嘗試生成評估 Parquet。")

# 2.1 資料收集器 (Data Collator)
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any # Whisper Processor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt") 
        # This is a data collator, will transfer Hugging Face Trainer, 
        # let every step do "batch data" and "padding data"

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        # 確保只返回模型需要的 input_features 和 labels
        return {"input_features": batch["input_features"], "labels": batch["labels"]}


# 2.2 評估指標
metric = evaluate.load("cer") # Character Error Rate
# Measures errors at the character level
# Used commonly in OCR and fine-grained ASR evaluation 

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"cer": cer}

# 2.3 載入訓練資料集
print(f"載入訓練資料集從: {TRAIN_PARQUET_PATH}")
try:
    train_features = Features({
        AUDIO_COLUMN_NAME: Audio(sampling_rate=16000),
        TEXT_COLUMN_NAME: Value("string"), # 這裡保持 Value("string")，因為現在 Parquet 中存的是實際文本了
    })
    train_dataset = load_dataset(
        "parquet",
        data_files=TRAIN_PARQUET_PATH,
        features=train_features,
        split="train"
    )
    if len(train_dataset) == 0:
        raise ValueError(f"訓練資料集 '{TRAIN_PARQUET_PATH}' 載入後為空。請檢查 Parquet 檔案內容。")

    if TEST_MODE:
        original_train_size = len(train_dataset)
        train_dataset = train_dataset.select(range(min(TEST_MODE_SAMPLES, original_train_size)))
        print(f"測試模式已啟用: 訓練資料集已從 {original_train_size} 筆截斷至 {len(train_dataset)} 筆。")

    train_dataset = train_dataset.rename_columns({AUDIO_COLUMN_NAME: "audio", TEXT_COLUMN_NAME: "text"})
    if "audio" not in train_dataset.column_names or "text" not in train_dataset.column_names:
        raise ValueError("訓練資料集在重命名後缺少 'audio' 或 'text' 欄位。請檢查 AUDIO_COLUMN_NAME 和 TEXT_COLUMN_NAME 設定。")

    print(f"訓練資料集載入成功。大小: {len(train_dataset)}")
except FileNotFoundError:
    print(f"致命錯誤: 找不到訓練 Parquet 檔案 '{TRAIN_PARQUET_PATH}'。請確認檔案路徑是否正確。程式終止。")
    exit()
except Exception as e:
    print(f"致命錯誤: 載入訓練 Parquet 檔案時發生錯誤。請確認檔案格式和欄位名稱是否正確。錯誤詳情: {e}。程式終止。")
    exit()

# 2.4 載入評估資料集 (如果提供)
eval_dataset = None
if eval_parquet_generated and os.path.exists(EVAL_PARQUET_PATH):
    print(f"載入評估資料集從: {EVAL_PARQUET_PATH}")
    try:
        eval_features = Features({
            AUDIO_COLUMN_NAME: Audio(sampling_rate=16000),
            TEXT_COLUMN_NAME: Value("string"), # 這裡保持 Value("string")
        })
        eval_dataset = load_dataset(
            "parquet",
            data_files=EVAL_PARQUET_PATH,
            features=eval_features,
            split="train"
        )
        if len(eval_dataset) == 0:
            print(f"警告: 評估資料集 '{EVAL_PARQUET_PATH}' 載入後為空。將不進行評估。")
            eval_dataset = None
        else:
            if TEST_MODE:
                original_eval_size = len(eval_dataset)
                eval_dataset = eval_dataset.select(range(min(TEST_MODE_SAMPLES, original_eval_size)))
                print(f"測試模式已啟用: 評估資料集已從 {original_eval_size} 筆截斷至 {len(eval_dataset)} 筆。")

            eval_dataset = eval_dataset.rename_columns({AUDIO_COLUMN_NAME: "audio", TEXT_COLUMN_NAME: "text"})
            if "audio" not in eval_dataset.column_names or "text" not in eval_dataset.column_names:
                raise ValueError("評估資料集在重命名後缺少 'audio' 或 'text' 欄位。請檢查 AUDIO_COLUMN_NAME 和 TEXT_COLUMN_NAME 設定。")
            print(f"評估資料集載入成功。大小: {len(eval_dataset)}")
    except Exception as e:
        print(f"警告: 載入評估 Parquet 檔案 '{EVAL_PARQUET_PATH}' 時發生錯誤。將不進行評估。錯誤詳情: {e}")
        eval_dataset = None
else:
    print(f"警告: 評估 Parquet 檔案 '{EVAL_PARQUET_PATH}' 不存在或未成功生成，將不進行評估。")

# --- 3. 模型初始化與資料處理 ---
feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language="zh", task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="zh", task="transcribe")

def prepare_dataset_batch(batch):
    audio_arrays = [audio["array"] for audio in batch["audio"]]
    
    batch["input_features"] = feature_extractor(
        audio_arrays,
        sampling_rate=16000
    ).input_features

    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

print("進行單條數據預處理測試...")
try:
    if len(train_dataset) > 0:
        sample_train_data = train_dataset[0]
        processed_sample = prepare_dataset_batch({"audio": [sample_train_data["audio"]], "text": [sample_train_data["text"]]})
        assert "input_features" in processed_sample, "單條訓練數據處理後缺少 'input_features'。"
        assert "labels" in processed_sample, "單條訓練數據處理後缺少 'labels'。"
        print("單條訓練數據預處理測試成功。")
    else:
        print("訓練資料集為空，跳過單條數據預處理測試。")
    
    if eval_dataset and len(eval_dataset) > 0:
        sample_eval_data = eval_dataset[0]
        processed_sample_eval = prepare_dataset_batch({"audio": [sample_eval_data["audio"]], "text": [sample_eval_data["text"]]})
        assert "input_features" in processed_sample_eval, "單條評估數據處理後缺少 'input_features'。"
        assert "labels" in processed_sample_eval, "單條評估數據處理後缺少 'labels'。"
        print("單條評估數據預處理測試成功。")
    elif eval_dataset:
        print("評估資料集載入後為空，跳過單條評估數據預處理測試。")
    else:
        print("沒有評估資料集，跳過單條評估數據預處理測試。")

except Exception as e:
    print(f"致命錯誤: 單條數據預處理測試失敗。這通常表示音訊檔案損壞、路徑錯誤、或文本編碼問題。錯誤詳情: {e}。程式終止。")
    exit()

print("開始批次處理訓練資料集 (map 操作)...")
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))

train_dataset = train_dataset.map(
    prepare_dataset_batch,
    remove_columns=[col for col in train_dataset.column_names if col not in ["input_features", "labels"]],
    batched=True,
    batch_size=BATCH_PROCESS_SIZE,
    num_proc=DATALOADER_NUM_WORKERS 
)
print("開始過濾訓練資料集 (filter 操作)...")
train_dataset = train_dataset.filter(lambda example: example["input_features"] is not None and example["labels"] is not None)
print(f"訓練資料集批次處理和過濾完成。最終大小: {len(train_dataset)}")

if eval_dataset: 
    print("開始批次處理評估資料集 (map 操作)...")
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))

    eval_dataset = eval_dataset.map(
        prepare_dataset_batch,
        remove_columns=[col for col in eval_dataset.column_names if col not in ["input_features", "labels"]],
        batched=True,
        batch_size=BATCH_PROCESS_SIZE,
        num_proc=DATALOADER_NUM_WORKERS 
    )
    print("開始過濾評估資料集 (filter 操作)...")
    eval_dataset = eval_dataset.filter(lambda example: example["input_features"] is not None and example["labels"] is not None)
    print(f"評估資料集批次處理和過濾完成。最終大小: {len(eval_dataset)}")
else:
    print("沒有評估資料集，跳過評估資料集批次處理。")

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 引入 LoRA 相關的設定和模型處理 ---
print(f"載入預訓練模型: {MODEL_NAME}")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.bfloat16 if BF16_ENABLED else torch.float16 if FP16_ENABLED else torch.float32,
    device_map={"": device},
)

model.config.use_cache = False
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# 緊急修復 1: 禁用 no_speech_detection 相關的內部邏輯 (保留作為預防措施)
model.config.no_speech_threshold = None
model.config.no_speech_tokens = None

model = prepare_model_for_kbit_training(model) 

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# --- 4. 定義訓練參數與啟動訓練 ---
print(f"--- 策略設定前的 eval_dataset 最終狀態: {eval_dataset is not None} ---")
print(f"原始 eval_dataset: {eval_dataset}")
if eval_dataset is not None:
    print(f"eval_dataset 類型: {type(eval_dataset)}")
    print(f"eval_dataset 樣本數: {len(eval_dataset)}")


if eval_dataset is not None and len(eval_dataset) > 0:
    print("偵測到 eval_dataset 存在且非空。啟用評估和最佳模型載入策略。")
    current_save_strategy = "steps"
    current_evaluation_strategy = "steps"
    current_load_best_model_at_end = True
    current_metric_for_best_model = "cer"
    current_greater_is_better = False
    current_do_eval = True 
    current_eval_steps = EVAL_STEPS
    current_per_device_eval_batch_size = PER_DEVICE_EVAL_BATCH_SIZE
else:
    print("偵測到 eval_dataset 不存在或為空。禁用評估和最佳模型載入策略。")
    current_save_strategy = "no"
    current_evaluation_strategy = "no"
    current_load_best_model_at_end = False
    current_metric_for_best_model = None
    current_greater_is_better = None
    current_do_eval = False
    current_eval_steps = None
    current_per_device_eval_batch_size = None

print(f"最終設定: current_save_strategy={current_save_strategy}, current_evaluation_strategy={current_evaluation_strategy}, current_load_best_model_at_end={current_load_best_model_at_end}, current_do_eval={current_do_eval}, current_eval_steps={current_eval_steps}, current_per_device_eval_batch_size={current_per_device_eval_batch_size}")


training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    fp16=FP16_ENABLED,
    bf16=BF16_ENABLED,
    
    label_names=["labels"], 
    
    eval_strategy=current_evaluation_strategy, 
    save_strategy=current_save_strategy,
    
    eval_steps=current_eval_steps,
    save_steps=SAVE_STEPS,
    load_best_model_at_end=current_load_best_model_at_end,
    metric_for_best_model=current_metric_for_best_model,
    greater_is_better=current_greater_is_better,
    #no_repeat_ngram_size=3, 
    gradient_checkpointing=GRADIENT_CHECKPOINTING_ENABLED,
    dataloader_num_workers=DATALOADER_NUM_WORKERS,

    predict_with_generate=True,
    generation_num_beams=GENERATION_NUM_BEAMS, 

    logging_steps=LOGGING_STEPS,
    report_to=["tensorboard"],
    push_to_hub=PUSH_TO_HUB,

    per_device_eval_batch_size=current_per_device_eval_batch_size,
    overwrite_output_dir=True, 
    
    do_eval=current_do_eval,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer, 
)

print("開始 Fine-tuning...")
try:
    trainer.train()
    print(f"Fine-tuning 完成！模型已儲存到 '{OUTPUT_DIR}'")

    final_adapter_path = os.path.join(OUTPUT_DIR, "final_lora_adapter")
    trainer.model.save_pretrained(final_adapter_path)
    processor.save_pretrained(final_adapter_path)
    print(f"最終 LoRA 適配器和處理器已儲存到 '{final_adapter_path}'。")

except Exception as e:
    print(f"\n訓練過程中發生錯誤: {e}")
    print("請檢查您的資料路徑、原始 CSV 或 Parquet 檔案格式、欄位名稱以及硬體資源是否足夠。")
    print("特別是檢查 `AUDIO_COLUMN_NAME` 和 `TEXT_COLUMN_NAME` 是否與您的 CSV/Parquet 檔案中的實際欄位名稱完全匹配。")
    print("並確認 `load_dataset` 中的 `features` 定義與您的 Parquet 檔案內容一致。")

## 5. 清理暫存檔案以釋放空間

print("\n--- 開始清理暫存檔案與快取 ---")

# --- 5.1 清理 datasets 庫的快取 ---
try:
    hf_cache_home = os.path.expanduser(os.path.join("~", ".cache", "huggingface"))
    datasets_cache_dir = os.path.join(hf_cache_home, "datasets")

    if os.path.exists(datasets_cache_dir):
        print(f"發現 Hugging Face Datasets 快取目錄: {datasets_cache_dir}")
        print("警告: 建議手動檢查此目錄並刪除您不需要的特定數據集快取。")
        parquet_cache_path = os.path.join(datasets_cache_dir, "parquet")
        if os.path.exists(parquet_cache_path):
            print(f'正在嘗試刪除 datasets \'parquet\' 快取目錄: {parquet_cache_path}')
            shutil.rmtree(parquet_cache_path, ignore_errors=True)
            print("Datasets 'parquet' 快取目錄清理完成。")
        else:
            print(f"Datasets 'parquet' 快取目錄 '{parquet_cache_path}' 不存在，無需清理。")
    else:
        print("未發現 Hugging Face Datasets 快取目錄。")

except Exception as e:
    print(f"清理 Hugging Face Datasets 快取時發生錯誤: {e}")

# --- 5.2 清理訓練輸出目錄中的舊檢查點 (保留最終模型) ---
print(f"\n正在清理訓練輸出目錄 '{OUTPUT_DIR}' 中的舊檢查點...")

final_adapter_path = os.path.join(OUTPUT_DIR, "final_lora_adapter")

if os.path.exists(OUTPUT_DIR):
    for item in os.listdir(OUTPUT_DIR):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(item_path) and not item_path == final_adapter_path:
            if item.startswith("checkpoint-") or item == "runs":
                print(f"   - 刪除舊檢查點/日誌目錄: {item_path}")
                try:
                    shutil.rmtree(item_path)
                except OSError as e:
                    print(f"     警告: 無法刪除 '{item_path}'。請檢查權限或檔案是否被佔用。詳情: {e}")
        elif os.path.isfile(item_path) and (item.endswith(".json") or item.endswith(".txt")):
            pass
    print("訓練輸出目錄清理完成。")
else:
    print(f"訓練輸出目錄 '{OUTPUT_DIR}' 不存在，無需清理。")

print("--- 暫存檔案與快取清理結束 ---")