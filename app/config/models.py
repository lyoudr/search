"""
Model configuration file for different LLM models.
Add new models here to make them available in the system.
"""
from typing import Dict, Optional, Literal
from dataclasses import dataclass

ModelProvider = Literal["openai", "huggingface", "anthropic", "local"]
ModelType = Literal["api", "local"]


@dataclass
class ModelConfig:
    """Configuration for an LLM model"""
    name: str  # Model identifier (e.g., "gpt-4o", "qwen2.5-7b-instruct")
    display_name: str  # Human-readable name
    provider: ModelProvider  # openai, huggingface, anthropic, local
    model_type: ModelType  # api (cloud) or local (downloaded)
    
    # For API models (OpenAI, Anthropic)
    api_model_name: Optional[str] = None  # API model name (e.g., "gpt-4o" for OpenAI)
    
    # For Hugging Face models
    hf_model_id: Optional[str] = None  # Hugging Face model ID (e.g., "Qwen/Qwen2.5-7B-Instruct")
    hf_revision: Optional[str] = None  # Specific revision/branch
    hf_trust_remote_code: bool = False  # Trust remote code from HF
    
    # Model metadata
    size: Optional[str] = None  # e.g., "7B", "8B", "1.5B"
    quantization: Optional[str] = None  # e.g., "4bit", "8bit", "fp16", "fp32"
    max_context_length: Optional[int] = None  # Maximum context window

    """
    # ~ Size
    # * LLM (like GPT-4, Qwen, LLaMA) have billions of parameters
    # * Model        Parameters        FP32 RAM
    # * 7B           7 billion         ~14-16GB
    # * 13B          13 billion.       ~26-30GB
    
    # ~ Quantization
    # & Quantization is the process of reducing the precision of the numbers used to store
    # & a model's weights to make the model smaller, faster and less memory-intensive, often with minimal loss of accuracy
    Type	Description	Pros	                                        Cons
    ------------------------------------------------------------------------------------------
    fp32	Full 32-bit floating point	Max precision	                Huge memory & slow
    fp16	Half precision (16-bit float)	Faster, smaller	            Slight precision loss
    8bit	8-bit integer	Smaller, fits more models on GPU            Some accuracy loss
    4bit	4-bit integer	Extremely small, can run 7B+ on 6–8GB GPU	Accuracy can drop more
    
    # ~ Max context length (or context window)
    max_context_length is the maximum number of tokens the model can “look at” in one input.
    """
    
    # Model capabilities
    supports_chat: bool = True  # Supports chat format
    supports_completion: bool = True  # Supports completion format
    
    # Download/loading settings
    device_map: str = "auto"  # Device mapping for local models
    torch_dtype: Optional[str] = None  # e.g., "float16", "bfloat16", "float32"
    load_in_4bit: bool = False  # Use 4-bit quantization
    load_in_8bit: bool = False  # Use 8-bit quantization
    
    # API settings
    temperature: float = 0.3  # Default temperature
    max_tokens: Optional[int] = None  # Max tokens for generation
    
    # Description
    description: Optional[str] = None


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # ========== OpenAI Models ==========
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        display_name="GPT-4o",
        provider="openai",
        model_type="api",
        api_model_name="gpt-4o",
        size="Unknown",
        max_context_length=128000,
        description="OpenAI's latest GPT-4 optimized model"
    ),
    
    "gpt-4": ModelConfig(
        name="gpt-4",
        display_name="GPT-4",
        provider="openai",
        model_type="api",
        api_model_name="gpt-4",
        size="Unknown",
        max_context_length=8192,
        description="OpenAI GPT-4"
    ),
    
    "gpt-4-turbo": ModelConfig(
        name="gpt-4-turbo",
        display_name="GPT-4 Turbo",
        provider="openai",
        model_type="api",
        api_model_name="gpt-4-turbo",
        size="Unknown",
        max_context_length=128000,
        description="OpenAI GPT-4 Turbo"
    ),
    
    "gpt-3.5-turbo": ModelConfig(
        name="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        provider="openai",
        model_type="api",
        api_model_name="gpt-3.5-turbo",
        size="Unknown",
        max_context_length=16385,
        description="OpenAI GPT-3.5 Turbo"
    ),
    
    # ========== Qwen Models (Hugging Face) ==========
    "qwen2.5-7b-instruct": ModelConfig(
        name="qwen2.5-7b-instruct",
        display_name="Qwen2.5 7B Instruct",
        provider="huggingface",
        model_type="local",
        hf_model_id="Qwen/Qwen2.5-7B-Instruct",
        size="7B",
        quantization="4bit",             # use 4-bit quantization for speed
        load_in_4bit=True,               # enable 4-bit loading
        max_context_length=8192,         # reduce to a practical context length
        torch_dtype="float16",           # keep FP16 for GPU
        device_map="auto",               # auto-select GPU
        description="Qwen2.5 7B Instruct model from Hugging Face, optimized for fast text correction"
    ),
    
    "qwen2.5-14b-instruct": ModelConfig(
        name="qwen2.5-14b-instruct",
        display_name="Qwen2.5 14B Instruct",
        provider="huggingface",
        model_type="local",
        hf_model_id="Qwen/Qwen2.5-14B-Instruct",
        size="14B",
        quantization="fp16",
        max_context_length=32768,
        torch_dtype="float16",
        device_map="auto",
        description="Qwen2.5 14B Instruct model"
    ),
    
    "qwen2.5-32b-instruct": ModelConfig(
        name="qwen2.5-32b-instruct",
        display_name="Qwen2.5 32B Instruct",
        provider="huggingface",
        model_type="local",
        hf_model_id="Qwen/Qwen2.5-32B-Instruct",
        size="32B",
        quantization="fp16",
        max_context_length=32768,
        torch_dtype="float16",
        device_map="auto",
        description="Qwen2.5 32B Instruct model"
    ),
    
    "qwen2.5-7b-instruct-4bit": ModelConfig(
        name="qwen2.5-7b-instruct-4bit",
        display_name="Qwen2.5 7B Instruct (4-bit)",
        provider="huggingface",
        model_type="local",
        hf_model_id="Qwen/Qwen2.5-7B-Instruct",
        size="7B",
        quantization="4bit",
        max_context_length=32768,
        load_in_4bit=True,
        device_map="auto",
        description="Qwen2.5 7B Instruct with 4-bit quantization"
    ),
    
    "qwen2-7b-instruct": ModelConfig(
        name="qwen2-7b-instruct",
        display_name="Qwen2 7B Instruct",
        provider="huggingface",
        model_type="local",
        hf_model_id="Qwen/Qwen2-7B-Instruct",
        size="7B",
        quantization="fp16",
        max_context_length=32768,
        torch_dtype="float16",
        device_map="auto",
        description="Qwen2 7B Instruct model"
    ),
    
    # ========== LLaMA Models (Hugging Face) ==========
    "llama-3-8b-instruct": ModelConfig(
        name="llama-3-8b-instruct",
        display_name="LLaMA 3 8B Instruct",
        provider="huggingface",
        model_type="local",
        hf_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        size="8B",
        quantization="fp16",
        max_context_length=8192,
        torch_dtype="float16",
        device_map="auto",
        description="Meta LLaMA 3 8B Instruct model"
    ),
    
    "llama-3-70b-instruct": ModelConfig(
        name="llama-3-70b-instruct",
        display_name="LLaMA 3 70B Instruct",
        provider="huggingface",
        model_type="local",
        hf_model_id="meta-llama/Meta-Llama-3-70B-Instruct",
        size="70B",
        quantization="fp16",
        max_context_length=8192,
        torch_dtype="float16",
        device_map="auto",
        description="Meta LLaMA 3 70B Instruct model"
    ),
    
    "llama-3-8b-instruct-4bit": ModelConfig(
        name="llama-3-8b-instruct-4bit",
        display_name="LLaMA 3 8B Instruct (4-bit)",
        provider="huggingface",
        model_type="local",
        hf_model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        size="8B",
        quantization="4bit",
        max_context_length=8192,
        load_in_4bit=True,
        device_map="auto",
        description="Meta LLaMA 3 8B Instruct with 4-bit quantization"
    ),
    
    # ========== Legacy Models (for backward compatibility) ==========
    "qwen2": ModelConfig(
        name="qwen2",
        display_name="Qwen2 7B Instruct (Legacy)",
        provider="huggingface",
        model_type="local",
        hf_model_id="Qwen/Qwen2-7B-Instruct",
        size="7B",
        quantization="fp16",
        max_context_length=32768,
        torch_dtype="float16",
        device_map="auto",
        description="Legacy Qwen2 model name"
    ),
    
    "llama3": ModelConfig(
        name="llama3",
        display_name="LLaMA 3 8B (Legacy)",
        provider="huggingface",
        model_type="local",
        hf_model_id="meta-llama/Meta-Llama-3-8B",
        size="8B",
        quantization="fp16",
        max_context_length=8192,
        torch_dtype="float16",
        device_map="auto",
        description="Legacy LLaMA 3 model name"
    ),
}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get model configuration by name"""
    return MODEL_REGISTRY.get(model_name)


def list_available_models() -> Dict[str, ModelConfig]:
    """List all available models"""
    return MODEL_REGISTRY.copy()


def register_model(config: ModelConfig) -> None:
    """Register a new model configuration"""
    MODEL_REGISTRY[config.name] = config


def get_models_by_provider(provider: ModelProvider) -> Dict[str, ModelConfig]:
    """Get all models for a specific provider"""
    return {name: config for name, config in MODEL_REGISTRY.items() 
            if config.provider == provider}

