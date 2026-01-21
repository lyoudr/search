from sqlalchemy.orm import Session

from app.repositories import (
    transcription_repository,
    llm_output_repository
)
from app.services.model_manager import model_manager


def correct_whisper_text(whisper_text: str, model_name: str = "gpt-4") -> str:
    """
    Modify Whisper transcribed text (不標點符號)
    Supports any model registered in the model registry (GPT-4o, Qwen2.5, LLaMA 3, etc.)

    :param whisper_text: Whisper transcribed text
    :param model_name: Model identifier from model registry (e.g., "gpt-4o", "qwen2.5-7b-instruct", "llama-3-8b-instruct")
    :return: 修正過後的文字
    """
    
    prompt = (
        "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
        "1. 不補上標點符號\n"
        "2. 只修正詞彙錯誤\n\n"
        f"原文：{whisper_text}\n"
        f"修正："
    )
    
    # Use model manager to generate text
    try:
        corrected_text = model_manager.generate_text(
            model_name=model_name,
            prompt=prompt,
            max_length=512,
            temperature=0.3
        )
        return corrected_text
    except Exception as e:
        raise ValueError(f"Failed to generate text with model {model_name}: {e}")


def batch_correct_whisper_text(db: Session, llm_model_name: str = "gpt-4", 
                                         prompt_version: str = "v1", limit: int = 10):
    """
    Batch correct Whisper transcriptions using any registered LLM model and create LLMOutput records.
    
    :param db: Database session
    :param llm_model_name: Name of the LLM model to use (default: "gpt-4")
                          Can be any model from the registry: "gpt-4o", "qwen2.5-7b-instruct", "llama-3-8b-instruct", etc.
    :param prompt_version: Version of the prompt used (default: "v1")
    :param limit: Maximum number of transcriptions to process
    """
    # Ensure model exists in database
    llm_model_id = model_manager.ensure_model_in_db(db, llm_model_name)
    
    # Get transcriptions that don't have LLM outputs yet
    transcriptions = transcription_repository.get_all_transcriptions(db)
    processed_count = 0
    
    for transcription in transcriptions[:limit]:
        # Check if LLM output already exists for this transcription and model
        existing_outputs = llm_output_repository.get_llm_outputs_by_transcription(
            db, transcription.id
        )
        # Check if output exists for this specific model
        if any(output.llm_model_id == llm_model_id for output in existing_outputs):
            print(f"⏭️  Skipping transcription ID {transcription.id} - LLM output already exists for model {llm_model_name}")
            continue
        
        try:
            # Correct the transcription text
            corrected_text = correct_whisper_text(transcription.text, model_name=llm_model_name)
            
            # Create LLM output record
            llm_output = llm_output_repository.create_llm_output(
                db=db,
                transcription_id=transcription.id,
                llm_model_id=llm_model_id,
                prompt_version=prompt_version,
                text=corrected_text
            )
            print(f"✅ Corrected transcription ID {transcription.id} -> LLM output ID {llm_output.id}: {corrected_text[:50]}...")
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed to correct transcription ID {transcription.id}: {e}")
    
    print(f"✅ Processed {processed_count} transcriptions")

