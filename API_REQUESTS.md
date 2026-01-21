# API Request JSON Body Examples

This document provides examples of JSON request bodies for all POST endpoints in the models API.

## Table of Contents
1. [Register Model](#register-model)
2. [Download Model](#download-model)
3. [Load Model](#load-model)
4. [Generate Text](#generate-text)

---

## Register Model

**Endpoint:** `POST /models/register`

**Description:** Register a new model configuration (temporary, in-memory only)

### Example 1: Hugging Face Model (Qwen3)

```json
{
  "name": "qwen3-7b-instruct",
  "display_name": "Qwen3 7B Instruct",
  "provider": "huggingface",
  "model_type": "local",
  "hf_model_id": "Qwen/Qwen3-7B-Instruct",
  "size": "7B",
  "quantization": "fp16",
  "max_context_length": 32768,
  "torch_dtype": "float16",
  "device_map": "auto",
  "load_in_4bit": false,
  "load_in_8bit": false,
  "temperature": 0.3,
  "description": "Qwen3 7B Instruct model"
}
```

### Example 2: Hugging Face Model with 4-bit Quantization

```json
{
  "name": "qwen3-7b-instruct-4bit",
  "display_name": "Qwen3 7B Instruct (4-bit)",
  "provider": "huggingface",
  "model_type": "local",
  "hf_model_id": "Qwen/Qwen3-7B-Instruct",
  "size": "7B",
  "quantization": "4bit",
  "max_context_length": 32768,
  "device_map": "auto",
  "load_in_4bit": true,
  "load_in_8bit": false,
  "temperature": 0.3,
  "description": "Qwen3 7B Instruct with 4-bit quantization"
}
```

### Example 3: OpenAI API Model

```json
{
  "name": "gpt-4o-mini",
  "display_name": "GPT-4o Mini",
  "provider": "openai",
  "model_type": "api",
  "api_model_name": "gpt-4o-mini",
  "size": "Unknown",
  "max_context_length": 128000,
  "temperature": 0.3,
  "description": "OpenAI GPT-4o Mini"
}
```

### Example 4: Minimal Required Fields

```json
{
  "name": "my-model",
  "display_name": "My Model",
  "provider": "huggingface",
  "model_type": "local",
  "hf_model_id": "username/model-name"
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | ✅ Yes | Model identifier (e.g., "qwen3-7b-instruct") |
| `display_name` | string | ✅ Yes | Human-readable model name |
| `provider` | string | ✅ Yes | One of: "openai", "huggingface", "anthropic", "local" |
| `model_type` | string | ✅ Yes | "api" for cloud models, "local" for downloaded models |
| `api_model_name` | string | Optional | API model name (required for API models) |
| `hf_model_id` | string | Optional | Hugging Face model ID (required for HF models) |
| `hf_revision` | string | Optional | Specific Hugging Face revision/branch |
| `size` | string | Optional | Model size (e.g., "7B", "14B") |
| `quantization` | string | Optional | "4bit", "8bit", "fp16", "fp32" |
| `max_context_length` | integer | Optional | Maximum context window in tokens |
| `torch_dtype` | string | Optional | "float16", "bfloat16", "float32" |
| `device_map` | string | Optional | Device mapping (default: "auto") |
| `load_in_4bit` | boolean | Optional | Use 4-bit quantization (default: false) |
| `load_in_8bit` | boolean | Optional | Use 8-bit quantization (default: false) |
| `temperature` | float | Optional | Default temperature (default: 0.3) |
| `max_tokens` | integer | Optional | Max tokens for generation |
| `description` | string | Optional | Model description |

---

## Download Model

**Endpoint:** `POST /models/{model_name}/download`

**Description:** Download a Hugging Face model to local cache

### Example 1: Default Cache Directory

```json
{}
```

Or simply send an empty body (no JSON needed).

### Example 2: Custom Cache Directory

```json
{
  "cache_dir": "/path/to/custom/cache"
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `cache_dir` | string | Optional | Custom cache directory path |

**Note:** This endpoint only works for Hugging Face models. API models (like OpenAI) don't need downloading.

---

## Load Model

**Endpoint:** `POST /models/{model_name}/load`

**Description:** Load a model into memory

### Example 1: Normal Load

```json
{}
```

Or simply send an empty body (no JSON needed).

### Example 2: Force Reload

```json
{
  "force_reload": true
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `force_reload` | boolean | Optional | Force reload even if already loaded (default: false) |

**Note:** For API models, this is a no-op (no loading needed).

---

## Generate Text

**Endpoint:** `POST /generate/{model_name}`

**Description:** Generate text using a specified model

### Example 1: Basic Generation

```json
{
  "prompt": "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n1. 不補上標點符號\n2. 只修正詞彙錯誤\n\n原文：某某先生為45歲男性病人\n修正：",
  "max_length": 512,
  "temperature": 0.3
}
```

### Example 2: With Max Tokens

```json
{
  "prompt": "你好，請介紹一下自己",
  "max_length": 200,
  "temperature": 0.7,
  "max_tokens": 100
}
```

### Example 3: Simple Prompt

```json
{
  "prompt": "What is machine learning?",
  "max_length": 300
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | ✅ Yes | Input prompt text |
| `max_length` | integer | Optional | Maximum generation length (default: 512) |
| `temperature` | float | Optional | Sampling temperature (uses model default if None) |
| `max_tokens` | integer | Optional | Maximum tokens to generate |

---

## cURL Examples

### Register a Model

```bash
curl -X POST "http://localhost:8000/models/register" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "qwen3-7b-instruct",
    "display_name": "Qwen3 7B Instruct",
    "provider": "huggingface",
    "model_type": "local",
    "hf_model_id": "Qwen/Qwen3-7B-Instruct",
    "size": "7B",
    "quantization": "fp16"
  }'
```

### Download a Model

```bash
curl -X POST "http://localhost:8000/models/qwen2.5-7b-instruct/download" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Load a Model

```bash
curl -X POST "http://localhost:8000/models/qwen2.5-7b-instruct/load" \
  -H "Content-Type: application/json" \
  -d '{
    "force_reload": false
  }'
```

### Generate Text

```bash
curl -X POST "http://localhost:8000/generate/gpt-4o" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "你好，請介紹一下自己",
    "max_length": 200,
    "temperature": 0.7
  }'
```

---

## Python Examples

### Using requests library

```python
import requests

# Register a model
response = requests.post(
    "http://localhost:8000/models/register",
    json={
        "name": "qwen3-7b-instruct",
        "display_name": "Qwen3 7B Instruct",
        "provider": "huggingface",
        "model_type": "local",
        "hf_model_id": "Qwen/Qwen3-7B-Instruct",
        "size": "7B",
        "quantization": "fp16"
    }
)
print(response.json())

# Generate text
response = requests.post(
    "http://localhost:8000/generate/gpt-4o",
    json={
        "prompt": "你好，請介紹一下自己",
        "max_length": 200,
        "temperature": 0.7
    }
)
print(response.json())
```

---

## Notes

1. **Empty JSON Body**: For endpoints with optional request bodies, you can send `{}` or omit the body entirely.

2. **Content-Type Header**: Always include `Content-Type: application/json` header for POST requests with JSON bodies.

3. **Model Names**: Use the exact model name from the registry. Check available models with `GET /models/`.

4. **Validation**: All requests are validated using Pydantic schemas. Invalid requests will return 400 Bad Request with error details.

5. **Error Responses**: All endpoints return detailed error messages in case of failure.

