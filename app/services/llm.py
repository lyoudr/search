from sqlalchemy.orm import Session
from typing import Optional

from app.repositories import (
    transcription_repository,
    llm_output_repository
)
from app.services.model_manager import model_manager
from app.services.medical_document_retriever import MedicalDocumentRetriever


def correct_whisper_text(
    whisper_text: str,
    model_name: str = "gpt-4",
    use_rag: bool = False,
    transcription_id: Optional[int] = None,
    top_k_queries: int = 2,
    top_k_documents: int = 3
) -> str:
    """
    Modify Whisper transcribed text (不標點符號).
    
    Two modes:
    1. RAG + LLM correction: Uses query_index to find medical terms, then queries medical_documents for context
    2. Direct LLM correction: Uses LLM directly without RAG
    
    Supports any model registered in the model registry (GPT-4o, Qwen2.5, LLaMA 3, etc.)

    :param whisper_text: Whisper transcribed text
    :param model_name: Model identifier from model registry (e.g., "gpt-4o", "qwen2.5-7b-instruct", "llama-3-8b-instruct")
    :param use_rag: Whether to use RAG (Retrieval-Augmented Generation). If False, uses direct LLM correction.
    :param transcription_id: Transcription ID to find queries in query_index (required if use_rag=True)
    :param top_k_queries: Number of queries (terms) to retrieve from query_index (only used if use_rag=True)
    :param top_k_documents: Number of documents to retrieve per query from medical-documents (only used if use_rag=True)
    :return: 修正過後的文字
    """
    
    # Mode 1: RAG + LLM correction
    if use_rag:
        if not transcription_id:
            raise ValueError("transcription_id is required when use_rag=True")
        
        # Retrieve relevant medical documents using queries from query_index
        relevant_documents = []
        try:
            retriever = MedicalDocumentRetriever()
            relevant_documents = retriever.retrieve_documents_for_correction(
                transcription_id=transcription_id,
                transcription_text=whisper_text,
                top_k_queries=top_k_queries,
                top_k_documents=top_k_documents
            )
        except Exception as e:
            print(f"⚠️  Failed to retrieve documents for RAG: {e}")
        
        # Build prompt with RAG context
        prompt_parts = [
            "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
            "1. 不補上標點符號\n"
            "2. 只修正詞彙錯誤\n"
        ]
        
        # Add medical documents context (ground truth from medical-documents)
        if relevant_documents:
            documents_context = "\n\n".join([
                f"參考資料 {i+1}：\n{text}" 
                for i, text in enumerate(relevant_documents)
            ])
            prompt_parts.append(
                f"\n3. 請參考以下醫療文件資料（ground truth），只修改英文專有術語錯字部分，中文語意若有錯字也請修改：\n\n"
                f"{documents_context}\n\n"
            )
        
        prompt_parts.append(f"原文：{whisper_text}\n修正：")
        prompt = "".join(prompt_parts)
    
    # Mode 2: Direct LLM correction (no RAG)
    else:
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
            temperature=0.1
        )
        return corrected_text
    except Exception as e:
        raise ValueError(f"Failed to generate text with model {model_name}: {e}")


def batch_correct_whisper_text(
    db: Session,
    llm_model_name: str = "gpt-4",
    prompt_version: str = "v1",
    limit: int = 10,
    use_rag: bool = False,
    top_k_queries: int = 3,
    top_k_documents: int = 5
):
    """
    Batch correct Whisper transcriptions using any registered LLM model and create LLMOutput records.
    
    Two modes:
    1. RAG + LLM: Uses query_index and medical-documents for enhanced correction
    2. Direct LLM: Uses LLM directly without RAG
    
    :param db: Database session
    :param llm_model_name: Name of the LLM model to use (default: "gpt-4")
                          Can be any model from the registry: "gpt-4o", "qwen2.5-7b-instruct", "llama-3-8b-instruct", etc.
    :param prompt_version: Version of the prompt used (default: "v1")
    :param limit: Maximum number of transcriptions to process
    :param use_rag: Whether to use RAG (default: False - direct LLM correction)
    :param top_k_queries: Number of queries to retrieve from query_index (only if use_rag=True)
    :param top_k_documents: Number of documents per query from medical-documents (only if use_rag=True)
    """
    # Ensure model exists in database
    llm_model_id = model_manager.ensure_model_in_db(db, llm_model_name)
    
    # Get transcriptions that don't have LLM outputs yet
    transcriptions = transcription_repository.get_all_transcriptions(db)
    processed_count = 0
    
    for transcription in transcriptions[:limit]:
        # Check if LLM output already exists for this transcription and model
        existing_output = llm_output_repository.get_llm_output_by_transcription_and_model(
            db, transcription.id, llm_model_id
        )
        
        # Check if the specific field (text or text_with_rag) already exists
        if use_rag:
            # If using RAG, check if text_with_rag already exists
            if existing_output and existing_output.text_with_rag:
                print(f"⏭️  Skipping transcription ID {transcription.id} - RAG output already exists for model {llm_model_name}")
                continue
        else:
            # If not using RAG, check if text already exists
            if existing_output and existing_output.text:
                print(f"⏭️  Skipping transcription ID {transcription.id} - Direct LLM output already exists for model {llm_model_name}")
                continue
        
        try:
            # Correct the transcription text (with or without RAG)
            corrected_text = correct_whisper_text(
                transcription.text,
                model_name=llm_model_name,
                use_rag=use_rag,
                transcription_id=transcription.id,
                top_k_queries=top_k_queries,
                top_k_documents=top_k_documents
            )
            
            # Create or update LLM output record
            if existing_output:
                # Update existing record with the new field
                if use_rag:
                    llm_output = llm_output_repository.update_llm_output(
                        db=db,
                        llm_output_id=existing_output.id,
                        text_with_rag=corrected_text,
                        prompt_version=prompt_version
                    )
                else:
                    llm_output = llm_output_repository.update_llm_output(
                        db=db,
                        llm_output_id=existing_output.id,
                        text=corrected_text,
                        prompt_version=prompt_version
                    )
                print(f"✅ Updated transcription ID {transcription.id} -> LLM output ID {llm_output.id}: {corrected_text[:50]}...")
            else:
                # Create new record
                if use_rag:
                    llm_output = llm_output_repository.create_llm_output(
                        db=db,
                        transcription_id=transcription.id,
                        llm_model_id=llm_model_id,
                        prompt_version=prompt_version,
                        text=None,
                        text_with_rag=corrected_text
                    )
                else:
                    llm_output = llm_output_repository.create_llm_output(
                        db=db,
                        transcription_id=transcription.id,
                        llm_model_id=llm_model_id,
                        prompt_version=prompt_version,
                        text=corrected_text,
                        text_with_rag=None
                    )
                print(f"✅ Corrected transcription ID {transcription.id} -> LLM output ID {llm_output.id}: {corrected_text[:50]}...")
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed to correct transcription ID {transcription.id}: {e}")
    
    print(f"✅ Processed {processed_count} transcriptions")

