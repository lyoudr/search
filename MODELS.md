# Model Management System

This document explains how to use the model management system to add and use different LLM models.

## Overview

The model management system supports:
- **OpenAI API models**: GPT-4o, GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- **Hugging Face models**: Qwen2.5, Qwen2, LLaMA 3 (with optional quantization)
- **Easy model registration**: Add new models via configuration file

## Adding New Models

### Method 1: Edit `app/config/models.py`

Add your model configuration to the `MODEL_REGISTRY` dictionary:

```python
"your-model-name": ModelConfig(
    name="your-model-name",
    display_name="Your Model Display Name",
    provider="huggingface",  # or "openai", "anthropic", "local"
    model_type="local",  # or "api" for cloud models
    hf_model_id="username/model-name",  # For Hugging Face models
    size="7B",
    quantization="fp16",  # or "4bit", "8bit"
    max_context_length=32768,
    torch_dtype="float16",
    device_map="auto",
    description="Description of your model"
)
```

### Method 2: Register via API (Temporary)

```bash
POST /models/register
{
  "name": "my-model",
  "display_name": "My Model",
  "provider": "huggingface",
  "model_type": "local",
  "hf_model_id": "username/model-name",
  ...
}
```

## Using Models

### List Available Models

```bash
GET /models/
```

### Get Model Details

```bash
GET /models/{model_name}
```

### Download Hugging Face Models

Models are automatically downloaded when first used, but you can pre-download them:

```bash
POST /models/{model_name}/download
```

Example:
```bash
POST /models/qwen2.5-7b-instruct/download
```

### Use Models in Code

```python
from app.services.model_manager import model_manager

# Generate text
text = model_manager.generate_text(
    model_name="gpt-4o",
    prompt="Your prompt here",
    max_length=512,
    temperature=0.3
)
```

## Pre-configured Models

### OpenAI Models (API)
- `gpt-4o` - Latest GPT-4 optimized model
- `gpt-4` - GPT-4
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-3.5-turbo` - GPT-3.5 Turbo

### Qwen Models (Hugging Face)
- `qwen2.5-7b-instruct` - Qwen2.5 7B Instruct
- `qwen2.5-14b-instruct` - Qwen2.5 14B Instruct
- `qwen2.5-32b-instruct` - Qwen2.5 32B Instruct
- `qwen2.5-7b-instruct-4bit` - Qwen2.5 7B with 4-bit quantization
- `qwen2-7b-instruct` - Qwen2 7B Instruct (legacy)

### LLaMA Models (Hugging Face)
- `llama-3-8b-instruct` - LLaMA 3 8B Instruct
- `llama-3-70b-instruct` - LLaMA 3 70B Instruct
- `llama-3-8b-instruct-4bit` - LLaMA 3 8B with 4-bit quantization

## Requirements

### For OpenAI Models
- Set `OPENAI_API_KEY` in your `.env` file

### For Hugging Face Models
- Install required packages:
  ```bash
  pip install transformers torch accelerate
  ```

- For quantization (4-bit/8-bit):
  ```bash
  pip install bitsandbytes
  ```

- Optional: Set Hugging Face token for private models:
  ```bash
  export HF_TOKEN=your_token_here
  # or
  huggingface-cli login
  ```

## Model Loading

Models are cached in memory after first load. To reload:

```bash
POST /models/{model_name}/load?force_reload=true
```

To unload a model and free memory:

```bash
POST /models/{model_name}/unload
```

## Example: Adding Qwen3 Model

1. Edit `app/config/models.py`:

```python
"qwen3-7b-instruct": ModelConfig(
    name="qwen3-7b-instruct",
    display_name="Qwen3 7B Instruct",
    provider="huggingface",
    model_type="local",
    hf_model_id="Qwen/Qwen3-7B-Instruct",  # Update when available
    size="7B",
    quantization="fp16",
    max_context_length=32768,
    torch_dtype="float16",
    device_map="auto",
    description="Qwen3 7B Instruct model"
)
```

2. Use it:

```python
from app.services.llm import correct_whisper_text

corrected = correct_whisper_text(
    whisper_text="your text",
    model_name="qwen3-7b-instruct"
)
```

## API Endpoints

- `GET /models/` - List all available models
- `GET /models/{model_name}` - Get model details
- `POST /models/{model_name}/download` - Download Hugging Face model
- `POST /models/{model_name}/load` - Load model into memory
- `POST /models/{model_name}/unload` - Unload model from memory
- `GET /models/provider/{provider}` - List models by provider
- `GET /models/db/list` - List models in database
- `POST /models/register` - Register new model (temporary)

## Notes

- Hugging Face models are downloaded to `~/.cache/huggingface/` by default
- Large models may require significant GPU memory
- Use quantization (4-bit/8-bit) to reduce memory usage
- API models don't require downloading or loading

