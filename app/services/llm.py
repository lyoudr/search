from sqlalchemy.orm import Session
from typing import Optional

from app.repositories import (
    transcription_repository,
    llm_output_repository
)
from app.services.model_manager import model_manager
from app.services.medical_document_retriever import MedicalDocumentRetriever
from app.services.mtsamples_retriever import MTSamplesRetriever


def correct_whisper_text(
    whisper_text: str,
    model_name: str = "gpt-4",
    use_rag: bool = False,
    transcription_id: Optional[int] = None,
    top_k_queries: int = 2,
    top_k_documents: int = 3
) -> str:
    """
    Modify Whisper transcribed text (不標點符號) using LLM.
    
    Two modes:
    1. use_rag=False: Direct LLM correction (without RAG) - stores in 'text' field
    2. use_rag=True: LLM correction with RAG (using medical documents) - stores in 'text_with_rag' field
    
    Supports any model registered in the model registry (GPT-4o, Qwen2.5, LLaMA 3, etc.)

    :param whisper_text: Whisper transcribed text
    :param model_name: Model identifier from model registry (e.g., "gpt-4o", "qwen2.5-7b-instruct", "llama-3-8b-instruct")
    :param use_rag: Whether to use RAG (Retrieval-Augmented Generation) with medical documents
    :param transcription_id: Transcription ID (required if use_rag=True)
    :param top_k_queries: Number of queries (terms) to retrieve from query_index (only if use_rag=True)
    :param top_k_documents: Number of documents to retrieve per query from medical-documents (only if use_rag=True)
    :return: 修正過後的文字
    """
    
    # Base prompt for LLM correction
    base_prompt = (
        "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
        "1. 不補上標點符號\n"
        "2. 只修正詞彙錯誤\n\n"
    )
    
    # Mode 1: With RAG (use medical documents)
    if use_rag:
        if not transcription_id:
            raise ValueError("transcription_id is required when use_rag=True")
        
        try:
            # Retrieve relevant medical documents using MedicalDocumentRetriever
            retriever = MedicalDocumentRetriever()
            documents = retriever.retrieve_documents_for_correction(
                transcription_id=transcription_id,
                transcription_text=whisper_text,
                top_k_queries=top_k_queries,
                top_k_documents=top_k_documents
            )
            
            if documents:
                # Build prompt with retrieved documents as context
                context = "\n\n".join([f"參考文檔 {i+1}：{doc}" for i, doc in enumerate(documents)])
                prompt = (
                    f"{base_prompt}"
                    f"以下是一些醫療文檔作為參考：\n{context}\n\n"
        f"原文：{whisper_text}\n"
        f"修正："
    )
            else:
                # No documents found, fallback to direct LLM
                print(f"⚠️  No documents found for transcription {transcription_id}, using direct LLM correction")
                prompt = f"{base_prompt}原文：{whisper_text}\n修正："
            
            corrected_text = model_manager.generate_text(
                model_name=model_name,
                prompt=prompt,
                max_length=512,
                temperature=0.1
            )
            return corrected_text
            
        except Exception as e:
            print(f"⚠️  RAG correction failed: {e}, falling back to direct LLM")
            # Fallback to direct LLM
            prompt = f"{base_prompt}原文：{whisper_text}\n修正："
            corrected_text = model_manager.generate_text(
                model_name=model_name,
                prompt=prompt,
                max_length=512,
                temperature=0.1
            )
            return corrected_text
    
    # Mode 2: Direct LLM correction (no RAG)
    prompt = f"{base_prompt}原文：{whisper_text}\n修正："
    
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
    Batch correct Whisper transcriptions using LLM and create LLMOutput records.
    
    Two modes:
    1. use_rag=False: Direct LLM correction (without RAG) - stores in 'text' field
    2. use_rag=True: LLM correction with RAG (using medical documents) - stores in 'text_with_rag' field
    
    :param db: Database session
    :param llm_model_name: Name of the LLM model to use (default: "gpt-4")
                          Can be any model from the registry: "gpt-4o", "qwen2.5-7b-instruct", "llama-3-8b-instruct", etc.
    :param prompt_version: Version of the prompt used (default: "v1")
    :param limit: Maximum number of transcriptions to process
    :param use_rag: Whether to use RAG (Retrieval-Augmented Generation) with medical documents (default: False)
    :param top_k_queries: Number of queries (terms) to retrieve from query_index (only if use_rag=True)
    :param top_k_documents: Number of documents to retrieve per query from medical-documents (only if use_rag=True)
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
        
        # Check if the specific field already exists based on use_rag
        field_name = "text_with_rag" if use_rag else "text"
        
        if existing_output:
            existing_value = getattr(existing_output, field_name, None)
            if existing_value:
                mode_str = "RAG" if use_rag else "without RAG"
                print(f"⏭️  Skipping transcription ID {transcription.id} - {mode_str} output already exists for model {llm_model_name}")
                continue
        
        try:
            # Correct the transcription text
            corrected_text = correct_whisper_text(
                transcription.text,
                model_name=llm_model_name,
                use_rag=use_rag,
                transcription_id=transcription.id,
                top_k_queries=top_k_queries,
                top_k_documents=top_k_documents
            )
            
            # Prepare update/create parameters based on use_rag
            update_params = {
                "prompt_version": prompt_version
            }
            create_params = {
                "transcription_id": transcription.id,
                "llm_model_id": llm_model_id,
                "prompt_version": prompt_version
            }
            
            if use_rag:
                update_params["text_with_rag"] = corrected_text
                create_params["text"] = None
                create_params["text_with_rag"] = corrected_text
            else:
                update_params["text"] = corrected_text
                create_params["text"] = corrected_text
                create_params["text_with_rag"] = None
            
            # Create or update LLM output record
            if existing_output:
                llm_output = llm_output_repository.update_llm_output(
                    db=db,
                    llm_output_id=existing_output.id,
                    **update_params
                )
                mode_str = "RAG" if use_rag else "without RAG"
                print(f"✅ Updated transcription ID {transcription.id} -> LLM output ID {llm_output.id} ({mode_str}): {corrected_text[:50]}...")
            else:
                llm_output = llm_output_repository.create_llm_output(
                    db=db,
                    **create_params
                )
                mode_str = "RAG" if use_rag else "without RAG"
                print(f"✅ Corrected transcription ID {transcription.id} -> LLM output ID {llm_output.id} ({mode_str}): {corrected_text[:50]}...")
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed to correct transcription ID {transcription.id}: {e}")
    
    print(f"✅ Processed {processed_count} transcriptions")


def correct_whisper_text_with_mts(
    whisper_text: str,
    model_name: str = "gpt-4",
    transcription_id: Optional[int] = None,
    top_k_queries: int = 2,
    medical_specialty: str = "Hematology - Oncology",
    top_k: int = 5
) -> str:
    """
    Modify Whisper transcribed text (不標點符號) using LLM with MTSamples RAG.
    
    Process:
    1. Use transcription_id to query query_index and get stored medical terms (keywords)
    2. Use these keywords to search mtsamples index (filtered by medical_specialty)
    3. Use retrieved mtsamples keywords as context for LLM correction
    
    :param whisper_text: Whisper transcribed text
    :param model_name: Model identifier from model registry
    :param transcription_id: Transcription ID (required to get keywords from query_index)
    :param top_k_queries: Number of queries (terms) to retrieve from query_index
    :param medical_specialty: Medical specialty filter (default: "Hematology - Oncology")
    :param top_k: Number of MTSamples transcriptions to retrieve per query
    :return: 修正過後的文字
    """
    if not transcription_id:
        raise ValueError("transcription_id is required when using MTSamples RAG")
    
    # Base prompt for LLM correction
    base_prompt = (
        "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
        "1. 不補上標點符號\n"
        "2. 只修正詞彙錯誤\n\n"
    )
    
    try:
        # Retrieve relevant MTSamples keywords using medical terms from query_index
        retriever = MTSamplesRetriever()
        mtsamples_keywords = retriever.retrieve_transcriptions_for_correction(
            transcription_id=transcription_id,
            transcription_text=whisper_text,
            top_k_queries=top_k_queries,
            medical_specialty=medical_specialty,
            top_k=top_k
        )
        
        if mtsamples_keywords:
            # Build prompt with retrieved MTSamples keywords as context
            context = "\n\n".join([f"參考範例 {i+1}：{keywords}" for i, keywords in enumerate(mtsamples_keywords)])
            prompt = (
                f"{base_prompt}"
                f"以下是一些醫療轉錄範例關鍵詞（{medical_specialty}專科）作為參考：\n{context}\n\n"
                f"原文：{whisper_text}\n"
                f"修正："
            )
        else:
            # No MTSamples found, fallback to direct LLM
            print(f"⚠️  No MTSamples found for transcription {transcription_id}, using direct LLM correction")
            prompt = f"{base_prompt}原文：{whisper_text}\n修正："
        
        corrected_text = model_manager.generate_text(
            model_name=model_name,
            prompt=prompt,
            max_length=512,
            temperature=0.1
        )
        return corrected_text
        
    except Exception as e:
        print(f"⚠️  MTSamples RAG correction failed: {e}, falling back to direct LLM")
        # Fallback to direct LLM
        prompt = f"{base_prompt}原文：{whisper_text}\n修正："
        corrected_text = model_manager.generate_text(
            model_name=model_name,
            prompt=prompt,
            max_length=512,
            temperature=0.1
        )
        return corrected_text


def batch_correct_whisper_text_with_mts(
    db: Session,
    llm_model_name: str = "gpt-4",
    prompt_version: str = "v1",
    limit: int = 10,
    top_k_queries: int = 2,
    medical_specialty: str = "Hematology - Oncology",
    top_k: int = 5
):
    """
    Batch correct Whisper transcriptions using LLM with MTSamples RAG and create LLMOutput records.
    
    Process:
    1. Use transcription_id to query query_index and get stored medical terms (keywords)
    2. Use these keywords to search mtsamples index (filtered by medical_specialty)
    3. Store results in 'text_with_mts' field.
    
    :param db: Database session
    :param llm_model_name: Name of the LLM model to use (default: "gpt-4")
    :param prompt_version: Version of the prompt used (default: "v1")
    :param limit: Maximum number of transcriptions to process
    :param top_k_queries: Number of queries (terms) to retrieve from query_index
    :param medical_specialty: Medical specialty filter for MTSamples (default: "Hematology - Oncology")
    :param top_k: Number of MTSamples transcriptions to retrieve per query
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
        
        # Check if text_with_mts already exists
        if existing_output:
            if existing_output.text_with_mts:
                print(f"⏭️  Skipping transcription ID {transcription.id} - MTSamples output already exists for model {llm_model_name}")
                continue
        
        try:
            # Correct the transcription text with MTSamples RAG
            corrected_text = correct_whisper_text_with_mts(
                transcription.text,
                model_name=llm_model_name,
                transcription_id=transcription.id,
                top_k_queries=top_k_queries,
                medical_specialty=medical_specialty,
                top_k=top_k
            )
            
            # Prepare update/create parameters
            update_params = {
                "prompt_version": prompt_version,
                "text_with_mts": corrected_text
            }
            create_params = {
                "transcription_id": transcription.id,
                "llm_model_id": llm_model_id,
                "prompt_version": prompt_version,
                "text": None,
                "text_with_rag": None,
                "text_with_mts": corrected_text
            }
            
            # Create or update LLM output record
            if existing_output:
                llm_output = llm_output_repository.update_llm_output(
                    db=db,
                    llm_output_id=existing_output.id,
                    **update_params
                )
                print(f"✅ Updated transcription ID {transcription.id} -> LLM output ID {llm_output.id} (MTSamples): {corrected_text[:50]}...")
            else:
                llm_output = llm_output_repository.create_llm_output(
                    db=db,
                    **create_params
                )
                print(f"✅ Corrected transcription ID {transcription.id} -> LLM output ID {llm_output.id} (MTSamples): {corrected_text[:50]}...")
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed to correct transcription ID {transcription.id}: {e}")
    
    print(f"✅ Processed {processed_count} transcriptions with MTSamples RAG")

