"""
Routes for managing LLM models
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional

from app.models import get_db
from app.config.models import (
    ModelConfig,
    get_model_config,
    list_available_models,
    register_model,
    get_models_by_provider
)
from app.services.model_manager import model_manager
from app.repositories import llm_repository
from app.schemas.req_res.models import (
    ModelRegisterRequest,
    ModelDownloadRequest,
    ModelLoadRequest,
    ModelResponse,
)

router = APIRouter(tags=["models"], prefix="/models")


@router.get("/", summary="List all available models")
def list_models():
    """List all models registered in the system"""
    models = list_available_models()
    return {
        "total": len(models),
        "models": [
            {
                "name": name,
                "display_name": config.display_name,
                "provider": config.provider,
                "model_type": config.model_type,
                "size": config.size,
                "quantization": config.quantization,
                "description": config.description
            }
            for name, config in models.items()
        ]
    }


@router.get("/{model_name}", summary="Get model details")
def get_model_details(model_name: str):
    """Get detailed information about a specific model"""
    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    return {
        "name": config.name,
        "display_name": config.display_name,
        "provider": config.provider,
        "model_type": config.model_type,
        "api_model_name": config.api_model_name,
        "hf_model_id": config.hf_model_id,
        "size": config.size,
        "quantization": config.quantization,
        "max_context_length": config.max_context_length,
        "description": config.description
    }


@router.post("/{model_name}/download", summary="Download a Hugging Face model", response_model=ModelResponse)
def download_model(model_name: str, request: Optional[ModelDownloadRequest] = None):
    """
    Download a Hugging Face model to local cache.
    Only works for Hugging Face models (not API models).
    
    **Request Body (optional):**
    ```json
    {
        "cache_dir": "/path/to/cache"  // Optional: custom cache directory
    }
    ```
    """
    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    if config.provider != "huggingface":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' is not a Hugging Face model. Download is only available for Hugging Face models."
        )
    
    cache_dir = request.cache_dir if request else None
    success = model_manager.download_hf_model(config, cache_dir)
    if success:
        return ModelResponse(
            status="success",
            message=f"Model {model_name} downloaded successfully"
        )
    else:
        raise HTTPException(status_code=500, detail=f"Failed to download model {model_name}")


@router.post("/{model_name}/load", summary="Load a model into memory", response_model=ModelResponse)
def load_model(model_name: str, request: Optional[ModelLoadRequest] = None):
    """
    Load a model into memory.
    For API models, this is a no-op.
    For local models, loads from cache or downloads if needed.
    
    **Request Body (optional):**
    ```json
    {
        "force_reload": false  // Optional: force reload even if already loaded
    }
    ```
    """
    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    force_reload = request.force_reload if request else False
    try:
        model, tokenizer, pipeline = model_manager.load_model(model_name, force_reload)
        if config.model_type == "api":
            return ModelResponse(
                status="success",
                message=f"Model {model_name} is an API model and doesn't need loading",
                model_type="api"
            )
        else:
            return ModelResponse(
                status="success",
                message=f"Model {model_name} loaded successfully",
                model_type="local"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/{model_name}/unload", summary="Unload a model from memory")
def unload_model(model_name: str):
    """Unload a model from memory to free up resources"""
    config = get_model_config(model_name)
    if not config:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    model_manager.unload_model(model_name)
    return {"status": "success", "message": f"Model {model_name} unloaded"}


@router.get("/provider/{provider}", summary="List models by provider")
def list_models_by_provider(provider: str):
    """List all models for a specific provider"""
    models = get_models_by_provider(provider)
    if not models:
        raise HTTPException(status_code=404, detail=f"No models found for provider '{provider}'")
    
    return {
        "provider": provider,
        "total": len(models),
        "models": [
            {
                "name": name,
                "display_name": config.display_name,
                "size": config.size,
                "quantization": config.quantization,
                "description": config.description
            }
            for name, config in models.items()
        ]
    }


@router.get("/db/list", summary="List models registered in database")
def list_db_models(db: Session = Depends(get_db)):
    """List all models that are registered in the database"""
    models = llm_repository.get_all_llm_models(db)
    return {
        "total": len(models),
        "models": [
            {
                "id": model.id,
                "name": model.name,
                "provider": model.provider,
                "size": model.size,
                "quantization": model.quantization
            }
            for model in models
        ]
    }


@router.post("/register", summary="Register a new model configuration", response_model=ModelResponse)
def register_new_model(request: ModelRegisterRequest):
    """
    Register a new model configuration.
    Note: This is a temporary registration (in-memory only).
    For permanent registration, add the model to app/config/models.py
    
    **Request Body Example:**
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
        "description": "Qwen3 7B Instruct model"
    }
    ```
    
    **Minimal Example (OpenAI API model):**
    ```json
    {
        "name": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "provider": "openai",
        "model_type": "api",
        "api_model_name": "gpt-4o-mini",
        "description": "OpenAI GPT-4o Mini"
    }
    ```
    """
    try:
        model_config = ModelConfig(**request.dict())
        register_model(model_config)
        return ModelResponse(
            status="success",
            message=f"Model {model_config.name} registered successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to register model: {str(e)}")

