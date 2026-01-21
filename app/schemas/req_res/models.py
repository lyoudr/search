from pydantic import BaseModel, Field
from typing import Optional, Literal


class ModelRegisterRequest(BaseModel):
    """Request schema for registering a new model"""
    name: str = Field(..., description="Model identifier (e.g., 'qwen3-7b-instruct')")
    display_name: str = Field(..., description="Human-readable model name")
    provider: Literal["openai", "huggingface", "anthropic", "local"] = Field(
        ..., description="Model provider"
    )
    model_type: Literal["api", "local"] = Field(
        ..., description="Model type: 'api' for cloud models, 'local' for downloaded models"
    )
    
    # For API models (OpenAI, Anthropic)
    api_model_name: Optional[str] = Field(
        None, description="API model name (e.g., 'gpt-4o' for OpenAI)"
    )
    
    # For Hugging Face models
    hf_model_id: Optional[str] = Field(
        None, description="Hugging Face model ID (e.g., 'Qwen/Qwen2.5-7B-Instruct')"
    )
    hf_revision: Optional[str] = Field(
        None, description="Specific Hugging Face revision/branch"
    )
    hf_trust_remote_code: bool = Field(
        False, description="Trust remote code from Hugging Face"
    )
    
    # Model metadata
    size: Optional[str] = Field(None, description="Model size (e.g., '7B', '8B', '1.5B')")
    quantization: Optional[str] = Field(
        None, description="Quantization type: '4bit', '8bit', 'fp16', 'fp32'"
    )
    max_context_length: Optional[int] = Field(
        None, description="Maximum context window in tokens"
    )
    
    # Model capabilities
    supports_chat: bool = Field(True, description="Supports chat format")
    supports_completion: bool = Field(True, description="Supports completion format")
    
    # Download/loading settings (for local models)
    device_map: str = Field("auto", description="Device mapping for local models")
    torch_dtype: Optional[str] = Field(
        None, description="Torch dtype: 'float16', 'bfloat16', 'float32'"
    )
    load_in_4bit: bool = Field(False, description="Use 4-bit quantization")
    load_in_8bit: bool = Field(False, description="Use 8-bit quantization")
    
    # API settings
    temperature: float = Field(0.3, description="Default temperature")
    max_tokens: Optional[int] = Field(None, description="Max tokens for generation")
    
    # Description
    description: Optional[str] = Field(None, description="Model description")


class ModelDownloadRequest(BaseModel):
    """Request schema for downloading a model"""
    cache_dir: Optional[str] = Field(
        None, description="Custom cache directory path (optional)"
    )


class ModelLoadRequest(BaseModel):
    """Request schema for loading a model"""
    force_reload: bool = Field(
        False, description="Force reload even if model is already loaded"
    )


class ModelGenerateRequest(BaseModel):
    """Request schema for generating text with a model"""
    prompt: str = Field(..., description="Input prompt")
    max_length: int = Field(512, description="Maximum generation length")
    temperature: Optional[float] = Field(
        None, description="Sampling temperature (uses model default if None)"
    )
    max_tokens: Optional[int] = Field(
        None, description="Maximum tokens to generate"
    )


class ModelResponse(BaseModel):
    """Response schema for model operations"""
    status: str
    message: str
    model_type: Optional[str] = None


class ModelListResponse(BaseModel):
    """Response schema for listing models"""
    total: int
    models: list[dict]


class ModelDetailResponse(BaseModel):
    """Response schema for model details"""
    name: str
    display_name: str
    provider: str
    model_type: str
    api_model_name: Optional[str] = None
    hf_model_id: Optional[str] = None
    size: Optional[str] = None
    quantization: Optional[str] = None
    max_context_length: Optional[int] = None
    description: Optional[str] = None

