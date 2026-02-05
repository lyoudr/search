from sqlalchemy.orm import Session
from typing import Optional

from app.repositories import transcription_repository, llm_output_repository
from app.services.model_manager import model_manager
from app.services.medical_document_retriever import MedicalDocumentRetriever
from app.services.hematology_retriever import HematologyRetriever


def correct_whisper_text(
    whisper_text: str,
    model_name: str = "gpt-4",
    use_rag: bool = False,
    transcription_id: Optional[int] = None,
    top_k_queries: int = 2,
    top_k_documents: int = 3,
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
                top_k_documents=top_k_documents,
            )

            if documents:
                # Build prompt with retrieved documents as context
                context = "\n\n".join(
                    [f"參考文檔 {i+1}：{doc}" for i, doc in enumerate(documents)]
                )
                prompt = (
                    f"{base_prompt}"
                    f"以下是一些醫療文檔作為參考：\n{context}\n\n"
                    f"原文：{whisper_text}\n"
                    f"修正："
                )
            else:
                # No documents found, fallback to direct LLM
                print(
                    f"⚠️  No documents found for transcription {transcription_id}, using direct LLM correction"
                )
                prompt = f"{base_prompt}原文：{whisper_text}\n修正："

            corrected_text = model_manager.generate_text(
                model_name=model_name, prompt=prompt, max_length=512, temperature=0.1
            )
            return corrected_text

        except Exception as e:
            print(f"⚠️  RAG correction failed: {e}, falling back to direct LLM")
            # Fallback to direct LLM
            prompt = f"{base_prompt}原文：{whisper_text}\n修正："
            corrected_text = model_manager.generate_text(
                model_name=model_name, prompt=prompt, max_length=512, temperature=0.1
            )
            return corrected_text

    # Mode 2: Direct LLM correction (no RAG)
    prompt = f"{base_prompt}原文：{whisper_text}\n修正："

    try:
        corrected_text = model_manager.generate_text(
            model_name=model_name, prompt=prompt, max_length=512, temperature=0.1
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
    top_k_documents: int = 5,
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
        existing_output = (
            llm_output_repository.get_llm_output_by_transcription_and_model(
                db, transcription.id, llm_model_id
            )
        )

        # Check if the specific field already exists based on use_rag
        field_name = "text_with_rag" if use_rag else "text"

        if existing_output:
            existing_value = getattr(existing_output, field_name, None)
            if existing_value:
                mode_str = "RAG" if use_rag else "without RAG"
                print(
                    f"⏭️  Skipping transcription ID {transcription.id} - {mode_str} output already exists for model {llm_model_name}"
                )
                continue

        try:
            # Correct the transcription text
            corrected_text = correct_whisper_text(
                transcription.text,
                model_name=llm_model_name,
                use_rag=use_rag,
                transcription_id=transcription.id,
                top_k_queries=top_k_queries,
                top_k_documents=top_k_documents,
            )

            # Prepare update/create parameters based on use_rag
            update_params = {"prompt_version": prompt_version}
            create_params = {
                "transcription_id": transcription.id,
                "llm_model_id": llm_model_id,
                "prompt_version": prompt_version,
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
                    db=db, llm_output_id=existing_output.id, **update_params
                )
                mode_str = "RAG" if use_rag else "without RAG"
                print(
                    f"✅ Updated transcription ID {transcription.id} -> LLM output ID {llm_output.id} ({mode_str}): {corrected_text[:50]}..."
                )
            else:
                llm_output = llm_output_repository.create_llm_output(
                    db=db, **create_params
                )
                mode_str = "RAG" if use_rag else "without RAG"
                print(
                    f"✅ Corrected transcription ID {transcription.id} -> LLM output ID {llm_output.id} ({mode_str}): {corrected_text[:50]}..."
                )
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed to correct transcription ID {transcription.id}: {e}")

    print(f"✅ Processed {processed_count} transcriptions")


def correct_whisper_text_with_hematology(
    whisper_text: str,
    model_name: str = "gpt-4",
    transcription_id: Optional[int] = None,
    top_k_queries: int = 2,
    top_k: int = 5,
) -> str:
    """
    Modify Whisper transcribed text (不標點符號) using LLM with Hematology Dictionary RAG.

    Process:
    1. Use transcription_id to query query_index and get stored medical terms (keywords)
    2. Use these keywords to search hematology dictionary index
    3. Use retrieved hematology dictionary entries as context for LLM correction

    :param whisper_text: Whisper transcribed text
    :param model_name: Model identifier from model registry
    :param transcription_id: Transcription ID (required to get keywords from query_index)
    :param top_k_queries: Number of queries (terms) to retrieve from query_index
    :param top_k: Number of hematology dictionary entries to retrieve per query
    :return: 修正過後的文字
    """
    if not transcription_id:
        raise ValueError(
            "transcription_id is required when using Hematology Dictionary RAG"
        )

    # Base prompt for LLM correction
    base_prompt = (
        "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
        "1. 不補上標點符號\n"
        "2. 只修正詞彙錯誤\n\n"
    )

    try:
        # Retrieve relevant hematology dictionary entries using medical terms from query_index
        retriever = HematologyRetriever()
        hematology_entries = retriever.retrieve_entries_for_correction(
            transcription_id=transcription_id,
            transcription_text=whisper_text,
            top_k_queries=top_k_queries,
            top_k=top_k,
        )

        if hematology_entries:
            # Build prompt with retrieved hematology dictionary entries as context
            context = "\n\n".join(
                [
                    f"參考範例 {i+1}：{entry}"
                    for i, entry in enumerate(hematology_entries)
                ]
            )
            prompt = (
                f"{base_prompt}"
                f"以下是一些血液學醫學詞典範例作為參考：\n{context}\n\n"
                f"原文：{whisper_text}\n"
                f"修正："
            )
        else:
            # No hematology entries found, fallback to direct LLM
            print(
                f"⚠️  No hematology dictionary entries found for transcription {transcription_id}, using direct LLM correction"
            )
            prompt = f"{base_prompt}原文：{whisper_text}\n修正："

        corrected_text = model_manager.generate_text(
            model_name=model_name, prompt=prompt, max_length=512, temperature=0.1
        )
        return corrected_text

    except Exception as e:
        print(
            f"⚠️  Hematology Dictionary RAG correction failed: {e}, falling back to direct LLM"
        )
        # Fallback to direct LLM
        prompt = f"{base_prompt}原文：{whisper_text}\n修正："
        corrected_text = model_manager.generate_text(
            model_name=model_name, prompt=prompt, max_length=512, temperature=0.1
        )
        return corrected_text


def batch_correct_whisper_text_with_hematology(
    db: Session,
    llm_model_name: str = "gpt-4",
    prompt_version: str = "v1",
    limit: int = 10,
    top_k_queries: int = 2,
    top_k: int = 5,
):
    """
    Batch correct Whisper transcriptions using LLM with Hematology Dictionary RAG and create LLMOutput records.

    Process:
    1. Use transcription_id to query query_index and get stored medical terms (keywords)
    2. Use these keywords to search hematology dictionary index
    3. Store results in 'text_with_hematology' field.

    :param db: Database session
    :param llm_model_name: Name of the LLM model to use (default: "gpt-4")
    :param prompt_version: Version of the prompt used (default: "v1")
    :param limit: Maximum number of transcriptions to process
    :param top_k_queries: Number of queries (terms) to retrieve from query_index
    :param top_k: Number of hematology dictionary entries to retrieve per query
    """
    # Ensure model exists in database
    llm_model_id = model_manager.ensure_model_in_db(db, llm_model_name)

    # Get transcriptions that don't have LLM outputs yet
    transcriptions = transcription_repository.get_all_transcriptions(db)
    processed_count = 0

    for transcription in transcriptions[:limit]:
        # Check if LLM output already exists for this transcription and model
        existing_output = (
            llm_output_repository.get_llm_output_by_transcription_and_model(
                db, transcription.id, llm_model_id
            )
        )

        # Check if text_with_hematology already exists
        if existing_output:
            if existing_output.text_with_hematology:
                print(
                    f"⏭️  Skipping transcription ID {transcription.id} - Hematology Dictionary output already exists for model {llm_model_name}"
                )
                continue

        try:
            # Correct the transcription text with Hematology Dictionary RAG
            corrected_text = correct_whisper_text_with_hematology(
                transcription.text,
                model_name=llm_model_name,
                transcription_id=transcription.id,
                top_k_queries=top_k_queries,
                top_k=top_k,
            )

            # Prepare update/create parameters
            update_params = {
                "prompt_version": prompt_version,
                "text_with_hematology": corrected_text,
            }
            create_params = {
                "transcription_id": transcription.id,
                "llm_model_id": llm_model_id,
                "prompt_version": prompt_version,
                "text": None,
                "text_with_rag": None,
                "text_with_hematology": corrected_text,
            }

            # Create or update LLM output record
            if existing_output:
                llm_output = llm_output_repository.update_llm_output(
                    db=db, llm_output_id=existing_output.id, **update_params
                )
                print(
                    f"✅ Updated transcription ID {transcription.id} -> LLM output ID {llm_output.id} (Hematology): {corrected_text[:50]}..."
                )
            else:
                llm_output = llm_output_repository.create_llm_output(
                    db=db, **create_params
                )
                print(
                    f"✅ Corrected transcription ID {transcription.id} -> LLM output ID {llm_output.id} (Hematology): {corrected_text[:50]}..."
                )
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed to correct transcription ID {transcription.id}: {e}")

    print(
        f"✅ Processed {processed_count} transcriptions with Hematology Dictionary RAG"
    )


def correct_whisper_text_with_agent(
    whisper_text: str,
    transcription_id: int,
    model_name: str = "gpt-4",
    initial_strategy: Optional[str] = None,
    max_iterations: int = 3,
    **kwargs,
) -> str:
    """
    Correct Whisper transcribed text using agent-based approach.

    The agent dynamically selects and combines tools (direct LLM, medical RAG,
    hematology RAG, combined RAG) to achieve the best correction quality.

    :param whisper_text: Whisper transcribed text
    :param transcription_id: Transcription ID (required for RAG tools)
    :param model_name: Model identifier from model registry
    :param initial_strategy: Initial strategy to try (None for auto-select)
    :param max_iterations: Maximum number of correction attempts
    :param kwargs: Additional parameters for tools
    :return: Corrected text
    """
    from app.services.transcription_agent import TranscriptionAgent

    agent = TranscriptionAgent(max_iterations=max_iterations)

    result = agent.correct_transcription(
        whisper_text=whisper_text,
        transcription_id=transcription_id,
        model_name=model_name,
        initial_strategy=initial_strategy,
        **kwargs,
    )

    return result.get("corrected_text", whisper_text)


def batch_correct_whisper_text_with_agent(
    db: Session,
    llm_model_name: str = "gpt-4",
    prompt_version: str = "v1",
    limit: int = 10,
    max_iterations: int = 3,
    initial_strategy: Optional[str] = None,
):
    """
    Batch correct Whisper transcriptions using agent-based approach.

    The agent dynamically selects the best correction strategy for each transcription.
    Results are stored in 'text_agent' field.

    :param db: Database session
    :param llm_model_name: Name of the LLM model to use (default: "gpt-4")
    :param prompt_version: Version of the prompt used (default: "v1")
    :param limit: Maximum number of transcriptions to process
    :param max_iterations: Maximum number of correction attempts per transcription
    :param initial_strategy: Initial strategy to try (None for auto-select)
    """
    from app.services.transcription_agent import TranscriptionAgent

    # Ensure model exists in database
    llm_model_id = model_manager.ensure_model_in_db(db, llm_model_name)

    # Get transcriptions
    transcriptions = transcription_repository.get_all_transcriptions(db)
    processed_count = 0

    agent = TranscriptionAgent(max_iterations=max_iterations)

    for transcription in transcriptions[:limit]:
        # Check if LLM output already exists for this transcription and model
        existing_output = (
            llm_output_repository.get_llm_output_by_transcription_and_model(
                db, transcription.id, llm_model_id
            )
        )

        # Check if text_agent already exists
        if existing_output:
            if existing_output.text_agent:
                print(
                    f"⏭️  Skipping transcription ID {transcription.id} - Agent output already exists for model {llm_model_name}"
                )
                continue

        try:
            # Correct the transcription text using agent
            result = agent.correct_transcription(
                whisper_text=transcription.text,
                transcription_id=transcription.id,
                model_name=llm_model_name,
                initial_strategy=initial_strategy,
            )

            corrected_text = result.get("corrected_text", transcription.text)
            method_used = result.get("method", "unknown")
            quality = result.get("quality", {})

            # Prepare update/create parameters
            update_params = {
                "prompt_version": prompt_version,
                "text_agent": corrected_text,
            }
            create_params = {
                "transcription_id": transcription.id,
                "llm_model_id": llm_model_id,
                "prompt_version": prompt_version,
                "text": None,
                "text_with_rag": None,
                "text_with_hematology": None,
                "text_agent": corrected_text,
            }

            # Create or update LLM output record
            if existing_output:
                llm_output = llm_output_repository.update_llm_output(
                    db=db, llm_output_id=existing_output.id, **update_params
                )
                print(
                    f"✅ Updated transcription ID {transcription.id} -> LLM output ID {llm_output.id} (Agent, method: {method_used}, quality: {quality.get('confidence', 'unknown')}): {corrected_text[:50]}..."
                )
            else:
                llm_output = llm_output_repository.create_llm_output(
                    db=db, **create_params
                )
                print(
                    f"✅ Corrected transcription ID {transcription.id} -> LLM output ID {llm_output.id} (Agent, method: {method_used}, quality: {quality.get('confidence', 'unknown')}): {corrected_text[:50]}..."
                )
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed to correct transcription ID {transcription.id}: {e}")

    print(f"✅ Processed {processed_count} transcriptions with Agent")
