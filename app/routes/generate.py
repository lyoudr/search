"""
Routes for text generation with LLM models
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.models import get_db
from app.services.model_manager import model_manager
from app.schemas.req_res.models import ModelGenerateRequest

router = APIRouter(tags=["generate"], prefix="/generate")


@router.post("/{model_name}", summary="Generate text using a model")
def generate_text(model_name: str, request: ModelGenerateRequest):
    """
    Generate text using the specified model.
    
    **Request Body:**
    ```json
    {
        "prompt": "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確。\n\n規則：\n1. 不補上任何標點符號\n2. 只修正詞彙錯誤\n3. 不新增或刪除內容\n4. 不輸出任何解釋\n\n請只輸出修正後的完整文字內容。\n\n原文：\n[...]",
        "max_length": 512,
        "temperature": 0.3,
        "max_tokens": 256
    }
    ```
    
    **Example Request:**
    ```bash
    POST /generate/qwen2.5-7b-instruct
    {
        "prompt": "你好，請介紹一下自己",
        "max_length": 200,
        "temperature": 0.7
    }
    ```
    """
    try:
        generated_text = model_manager.generate_text(
            model_name=model_name,
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        return {
            "status": "success",
            "model": model_name,
            "generated_text": generated_text,
            "prompt": request.prompt
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate text: {str(e)}")

