"""
Model Manager Service
Handles model loading, downloading, and inference for different LLM providers.
"""
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Optional import for quantization
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None

from app.config.models import ModelConfig, get_model_config
from app.repositories import llm_repository
from app.config.settings import get_settings

settings = get_settings()

# Global cache for loaded models
_loaded_models: Dict[str, Any] = {}
_loaded_tokenizers: Dict[str, Any] = {}
_loaded_pipelines: Dict[str, Any] = {}


class ModelManager:
    """Manages LLM models for different providers"""
    
    def __init__(self):
        # OpenAI client for OpenAI API models (gpt-4o, gpt-4, etc.)
        # self.openai_client = None
        # if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
        #     self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Hugging Face router client for HF models accessed via OpenAI-compatible API
        # Uncomment if you want to use Hugging Face router for some models
        self.openai_client = None
        if hasattr(settings, 'HF_TOKEN') and settings.HF_TOKEN:
            self.openai_client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=settings.HF_TOKEN
            )
        
    
    def ensure_model_in_db(self, db: Session, model_name: str) -> int:
        """
        Ensure model exists in database, create if not exists.
        Returns model ID.
        """
        config = get_model_config(model_name)
        if not config:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # Check if model exists in DB
        llm_model = llm_repository.get_llm_model_by_name(db, model_name)
        if llm_model:
            return llm_model.id
        
        # Create new model record
        llm_model = llm_repository.create_llm_model(
            db=db,
            name=model_name,
            provider=config.provider,
            size=config.size,
            quantization=config.quantization
        )
        return llm_model.id
    
    def download_hf_model(self, model_config: ModelConfig, cache_dir: Optional[str] = None) -> bool:
        """
        Download Hugging Face model to local cache.
        
        :param model_config: Model configuration
        :param cache_dir: Optional custom cache directory
        :return: True if successful
        """
        if model_config.provider != "huggingface" or not model_config.hf_model_id:
            raise ValueError(f"Model {model_config.name} is not a Hugging Face model")
        
        try:
            print(f"📥 Downloading model: {model_config.hf_model_id}")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.hf_model_id,
                revision=model_config.hf_revision,
                cache_dir=cache_dir,
                trust_remote_code=model_config.hf_trust_remote_code
            )
            
            # Prepare model loading kwargs
            model_kwargs = {
                "revision": model_config.hf_revision,
                "cache_dir": cache_dir,
                "trust_remote_code": model_config.hf_trust_remote_code,
                "device_map": model_config.device_map,
            }
            
            # Add quantization config if needed
            if model_config.load_in_4bit:
                if not BITSANDBYTES_AVAILABLE:
                    raise ImportError("bitsandbytes is required for 4-bit quantization. Install it with: pip install bitsandbytes")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype="float16",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = bnb_config
            elif model_config.load_in_8bit:
                if not BITSANDBYTES_AVAILABLE:
                    raise ImportError("bitsandbytes is required for 8-bit quantization. Install it with: pip install bitsandbytes")
                model_kwargs["load_in_8bit"] = True
            
            # Add torch dtype
            if model_config.torch_dtype:
                import torch
                dtype_map = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }
                model_kwargs["torch_dtype"] = dtype_map.get(model_config.torch_dtype, torch.float16)
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(
                model_config.hf_model_id,
                **model_kwargs
            )
            
            print(f"✅ Successfully downloaded model: {model_config.hf_model_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download model {model_config.hf_model_id}: {e}")
            return False
    
    def load_model(self, model_name: str, force_reload: bool = False):
        """
        Load a model into memory. For API models, this is a no-op.
        For local models, loads from Hugging Face cache or downloads if needed.
        
        :param model_name: Model identifier
        :param force_reload: Force reload even if already loaded
        :return: Loaded model, tokenizer, and pipeline (or None for API models)
        """
        config = get_model_config(model_name)
        if not config:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # API models don't need loading
        if config.model_type == "api":
            return None, None, None
        
        # Check cache
        if not force_reload and model_name in _loaded_pipelines:
            return (
                _loaded_models.get(model_name),
                _loaded_tokenizers.get(model_name),
                _loaded_pipelines.get(model_name)
            )
        
        # Load Hugging Face model
        if config.provider == "huggingface" and config.hf_model_id:
            try:
                print(f"🔄 Loading model: {model_name}")
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    config.hf_model_id,
                    revision=config.hf_revision,
                    trust_remote_code=config.hf_trust_remote_code
                )
                
                # Prepare model loading kwargs
                model_kwargs = {
                    "revision": config.hf_revision,
                    "trust_remote_code": config.hf_trust_remote_code,
                    "device_map": config.device_map,
                }
                
                # Add quantization config if needed
                if config.load_in_4bit:
                    print("load in 4bit")
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype="float16",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = bnb_config
                elif config.load_in_8bit:
                    model_kwargs["load_in_8bit"] = True
                
                # Add torch dtype
                if config.torch_dtype:
                    import torch
                    dtype_map = {
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "float32": torch.float32,
                    }
                    model_kwargs["torch_dtype"] = dtype_map.get(config.torch_dtype, torch.float16)
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    config.hf_model_id,
                    **model_kwargs
                )
                
                # Create pipeline
                generator = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map=config.device_map
                )
                
                # Cache loaded models
                _loaded_models[model_name] = model
                _loaded_tokenizers[model_name] = tokenizer
                _loaded_pipelines[model_name] = generator
                
                print(f"✅ Successfully loaded model: {model_name}")
                return model, tokenizer, generator
                
            except Exception as e:
                print(f"❌ Failed to load model {model_name}: {e}")
                raise
        
        raise ValueError(f"Unsupported model type for {model_name}")
    
    def generate_text(
        self,
        model_name: str,
        prompt: str,
        max_length: int = 512,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the specified model.
        
        :param model_name: Model identifier
        :param prompt: Input prompt
        :param max_length: Maximum generation length
        :param temperature: Sampling temperature (uses model default if None)
        :param kwargs: Additional generation parameters
        :return: Generated text
        """
        config = get_model_config(model_name)
        if not config:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        # OpenAI API models
        if config.provider == "openai" and config.model_type == "api":
            if not self.openai_client:
                raise ValueError("OpenAI API key not configured")
            
            temp = temperature if temperature is not None else config.temperature
            max_tokens = kwargs.get("max_tokens", config.max_tokens or max_length)
            
            response = self.openai_client.chat.completions.create(
                model=config.api_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=max_tokens,
                **{k: v for k, v in kwargs.items() if k != "max_tokens"}
            )
            return response.choices[0].message.content.strip()
        
        # Hugging Face local models
        elif config.provider == "huggingface" and config.model_type == "local":
            model, tokenizer, generator = self.load_model(model_name)
            
            if not generator:
                raise ValueError(f"Failed to load model {model_name}")
            
            temp = temperature if temperature is not None else config.temperature
            
            # Generate text
            result = generator(
                prompt,
                max_length=max_length,
                temperature=temp,
                do_sample=temp > 0,
                **kwargs
            )
            
            # Extract generated text (remove prompt)
            generated_text = result[0]["generated_text"]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
        
        else:
            raise ValueError(f"Unsupported model provider/type: {config.provider}/{config.model_type}")
    
    def unload_model(self, model_name: str):
        """Unload a model from memory to free up resources"""
        if model_name in _loaded_models:
            del _loaded_models[model_name]
        if model_name in _loaded_tokenizers:
            del _loaded_tokenizers[model_name]
        if model_name in _loaded_pipelines:
            del _loaded_pipelines[model_name]
        print(f"🗑️  Unloaded model: {model_name}")


# Global model manager instance
model_manager = ModelManager()

