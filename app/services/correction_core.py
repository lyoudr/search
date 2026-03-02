from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy.orm import Session

from app.ai_agents.agents.evaluation_agent import EvaluationAgent
from app.ai_agents.tools.shared_vector_tools import ChunkTool, EmbeddingTool, PineconeQueryTool
from app.repositories import llm_output_repository, transcription_repository
from app.services.hematology_services import HematologyRetriever
from app.services.medical_documents_services import MedicalDocumentRetriever
from app.services.model_manager import model_manager


BASE_CORRECTION_PROMPT = (
    "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確。\n\n"
    "規則：\n"
    "1. 不補上任何標點符號\n"
    "2. 只修正詞彙錯誤\n"
    "3. 不新增或刪除內容\n"
    "4. 不輸出任何解釋\n\n"
    "請只輸出修正後的完整文字內容。\n\n"
)


def build_correction_prompt(
    whisper_text: str,
    context_header: Optional[str] = None,
    context_lines: Optional[Iterable[str]] = None,
) -> str:
    prompt_parts: List[str] = [BASE_CORRECTION_PROMPT]
    lines = list(context_lines or [])
    if context_header and lines:
        prompt_parts.append(f"{context_header}\n")
        prompt_parts.append("\n\n".join(lines))
        prompt_parts.append("\n\n")
    prompt_parts.append(f"原文：\n[{whisper_text}]")
    return "".join(prompt_parts)


def build_numbered_lines(items: Iterable[str], prefix: str) -> List[str]:
    return [f"{prefix} {idx + 1}：{item}" for idx, item in enumerate(items)]


def generate_correction_text(
    model_name: str,
    prompt: str,
    max_length: int = 512,
    temperature: float = 0.1,
) -> str:
    return model_manager.generate_text(
        model_name=model_name,
        prompt=prompt,
        max_length=max_length,
        temperature=temperature,
    )


def extract_unique_metadata_terms(
    records: Iterable[Dict[str, Any]], term_key: str = "term"
) -> List[str]:
    terms: List[str] = []
    for record in records:
        term = record.get("metadata", {}).get(term_key, "")
        if term and term not in terms:
            terms.append(term)
    return terms


def run_llm_agent_correction(
    whisper_text: str,
    model_name: str,
    strategy: str,
    evaluation_agent: Any,
    chunk_tool: Any,
    embedding_tool: Any,
    vocab_query_tool: Any,
    medical_retriever: Any = None,
    hematology_retriever: Any = None,
    transcription_id: Optional[int] = None,
    top_k_queries: int = 2,
    top_k_documents: int = 3,
    top_k_hematology: int = 5,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    contexts: List[str] = []

    def _compact_context(raw_contexts: List[str], limit: int = 20) -> List[str]:
        compacted: List[str] = []
        for context in raw_contexts[:limit]:
            chunked = chunk_tool.execute(
                text=context,
                metadata={"type": "llm_context"},
            )["chunks"]
            if chunked:
                compacted.append(chunked[0]["text"])
        return compacted

    if strategy == "medical_document_rag":
        if not transcription_id:
            raise ValueError("transcription_id is required for medical_document_rag")
        if medical_retriever is None:
            raise ValueError("medical_retriever is required for medical_document_rag")
        docs = medical_retriever.retrieve_documents_for_correction(
            transcription_id=transcription_id,
            transcription_text=whisper_text,
            top_k_queries=top_k_queries,
            top_k_documents=top_k_documents,
        )
        metadata["documents_retrieved"] = len(docs)
        contexts = build_numbered_lines(_compact_context(docs), "參考文檔")
        context_header = "以下是一些醫療文檔作為參考："
    elif strategy == "hematology_rag":
        if not transcription_id:
            raise ValueError("transcription_id is required for hematology_rag")
        if hematology_retriever is None:
            raise ValueError("hematology_retriever is required for hematology_rag")
        entries = hematology_retriever.retrieve_entries_for_correction(
            transcription_id=transcription_id,
            transcription_text=whisper_text,
            top_k_queries=top_k_queries,
            top_k=top_k_hematology,
        )
        metadata["entries_retrieved"] = len(entries)
        contexts = build_numbered_lines(_compact_context(entries), "參考範例")
        context_header = "以下是一些血液學醫學詞典範例作為參考："
    else:
        context_header = None

    try:
        query_vector = embedding_tool.embed_text(whisper_text)
        query_result = vocab_query_tool.execute(
            query_vector=query_vector, top_k=5, include_metadata=True
        )
        vocab_hints = extract_unique_metadata_terms(query_result["matches"])
    except Exception:
        vocab_hints = []

    if vocab_hints:
        metadata["vocab_terms_retrieved"] = len(vocab_hints)
        contexts.extend(build_numbered_lines(vocab_hints[:5], "詞彙提示"))
        if context_header is None:
            context_header = "以下是一些血液學醫學詞彙作為參考："

    if contexts and context_header:
        prompt = build_correction_prompt(
            whisper_text=whisper_text,
            context_header=context_header,
            context_lines=contexts,
        )
    else:
        prompt = build_correction_prompt(whisper_text=whisper_text)

    corrected_text = generate_correction_text(model_name=model_name, prompt=prompt)
    quality = evaluation_agent.evaluate(
        original_text=whisper_text,
        corrected_text=corrected_text,
        method=strategy,
        metadata=metadata,
        model_name=model_name,
    )

    return {
        "corrected_text": corrected_text,
        "quality": quality,
        "metadata": metadata,
    }


def _build_llm_correction_runtime() -> Dict[str, Any]:
    return {
        "medical_retriever": MedicalDocumentRetriever(),
        "hematology_retriever": HematologyRetriever(),
        "chunk_tool": ChunkTool(chunk_size=256, chunk_overlap=40),
        "embedding_tool": EmbeddingTool(model_name="openai"),
        "vocab_query_tool": PineconeQueryTool(index_name="hematology-vocab"),
        "evaluation_agent": EvaluationAgent(),
    }


def correct_whisper_text(
    whisper_text: str,
    model_name: str = "gpt-5.2",
    use_rag: bool = False,
    transcription_id: Optional[int] = None,
    top_k_queries: int = 2,
    top_k_documents: int = 3,
) -> str:
    runtime = _build_llm_correction_runtime()

    if use_rag:
        if not transcription_id:
            raise ValueError("transcription_id is required when use_rag=True")
        try:
            result = run_llm_agent_correction(
                whisper_text=whisper_text,
                model_name=model_name,
                strategy="medical_document_rag",
                evaluation_agent=runtime["evaluation_agent"],
                chunk_tool=runtime["chunk_tool"],
                embedding_tool=runtime["embedding_tool"],
                vocab_query_tool=runtime["vocab_query_tool"],
                medical_retriever=runtime["medical_retriever"],
                hematology_retriever=runtime["hematology_retriever"],
                transcription_id=transcription_id,
                top_k_queries=top_k_queries,
                top_k_documents=top_k_documents,
            )
            return result["corrected_text"]
        except Exception as e:
            print(f"⚠️  RAG correction failed: {e}, falling back to direct LLM")
            result = run_llm_agent_correction(
                whisper_text=whisper_text,
                model_name=model_name,
                strategy="direct_llm",
                evaluation_agent=runtime["evaluation_agent"],
                chunk_tool=runtime["chunk_tool"],
                embedding_tool=runtime["embedding_tool"],
                vocab_query_tool=runtime["vocab_query_tool"],
                medical_retriever=runtime["medical_retriever"],
                hematology_retriever=runtime["hematology_retriever"],
            )
            return result["corrected_text"]

    try:
        result = run_llm_agent_correction(
            whisper_text=whisper_text,
            model_name=model_name,
            strategy="direct_llm",
            evaluation_agent=runtime["evaluation_agent"],
            chunk_tool=runtime["chunk_tool"],
            embedding_tool=runtime["embedding_tool"],
            vocab_query_tool=runtime["vocab_query_tool"],
            medical_retriever=runtime["medical_retriever"],
            hematology_retriever=runtime["hematology_retriever"],
        )
        return result["corrected_text"]
    except Exception as e:
        raise ValueError(f"Failed to generate text with model {model_name}: {e}")


def correct_whisper_text_with_hematology(
    whisper_text: str,
    model_name: str = "gpt-5.2",
    transcription_id: Optional[int] = None,
    top_k_queries: int = 2,
    top_k: int = 5,
) -> str:
    if not transcription_id:
        raise ValueError(
            "transcription_id is required when using Hematology Dictionary RAG"
        )

    runtime = _build_llm_correction_runtime()

    try:
        result = run_llm_agent_correction(
            whisper_text=whisper_text,
            model_name=model_name,
            strategy="hematology_rag",
            evaluation_agent=runtime["evaluation_agent"],
            chunk_tool=runtime["chunk_tool"],
            embedding_tool=runtime["embedding_tool"],
            vocab_query_tool=runtime["vocab_query_tool"],
            medical_retriever=runtime["medical_retriever"],
            hematology_retriever=runtime["hematology_retriever"],
            transcription_id=transcription_id,
            top_k_queries=top_k_queries,
            top_k_hematology=top_k,
        )
        return result["corrected_text"]
    except Exception as e:
        print(
            f"⚠️  Hematology Dictionary RAG correction failed: {e}, falling back to direct LLM"
        )
        result = run_llm_agent_correction(
            whisper_text=whisper_text,
            model_name=model_name,
            strategy="direct_llm",
            evaluation_agent=runtime["evaluation_agent"],
            chunk_tool=runtime["chunk_tool"],
            embedding_tool=runtime["embedding_tool"],
            vocab_query_tool=runtime["vocab_query_tool"],
            medical_retriever=runtime["medical_retriever"],
            hematology_retriever=runtime["hematology_retriever"],
        )
        return result["corrected_text"]


def batch_correct_whisper_text(
    db: Session,
    llm_model_name: str = "gpt-5.2",
    prompt_version: str = "v1",
    limit: int = 10,
    use_rag: bool = False,
    top_k_queries: int = 3,
    top_k_documents: int = 5,
) -> None:
    from app.ai_agents.tools.correction_tools import BatchCorrectWhisperTextTool

    result = BatchCorrectWhisperTextTool().execute(
        db=db,
        llm_model_name=llm_model_name,
        prompt_version=prompt_version,
        limit=limit,
        use_rag=use_rag,
        top_k_queries=top_k_queries,
        top_k_documents=top_k_documents,
    )
    print(f"✅ Processed {result['processed_count']} transcriptions")


def batch_correct_whisper_text_with_hematology(
    db: Session,
    llm_model_name: str = "gpt-5.2",
    prompt_version: str = "v1",
    limit: int = 10,
    top_k_queries: int = 2,
    top_k: int = 5,
) -> None:
    from app.ai_agents.tools.correction_tools import (
        BatchCorrectWhisperTextWithHematologyTool,
    )

    result = BatchCorrectWhisperTextWithHematologyTool().execute(
        db=db,
        llm_model_name=llm_model_name,
        prompt_version=prompt_version,
        limit=limit,
        top_k_queries=top_k_queries,
        top_k=top_k,
    )
    print(
        "✅ Processed "
        f"{result['processed_count']} transcriptions with Hematology Dictionary RAG"
    )


def correct_whisper_text_with_agent(
    whisper_text: str,
    transcription_id: int,
    model_name: str = "gpt-5.2",
    initial_strategy: Optional[str] = None,
    max_iterations: int = 3,
    **kwargs,
) -> str:
    from app.ai_agents.agents.transcription_agent import TranscriptionAgent

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
    llm_model_name: str = "gpt-5.2",
    prompt_version: str = "v1",
    limit: int = 10,
    max_iterations: int = 3,
    initial_strategy: Optional[str] = None,
) -> None:
    from app.ai_agents.agents.transcription_agent import TranscriptionAgent

    llm_model_id = model_manager.ensure_model_in_db(db, llm_model_name)
    transcriptions = transcription_repository.get_all_transcriptions(db)
    processed_count = 0
    agent = TranscriptionAgent(max_iterations=max_iterations)

    for transcription in transcriptions[:limit]:
        existing_output = llm_output_repository.get_llm_output_by_transcription_and_model(
            db, transcription.id, llm_model_id
        )
        if existing_output and existing_output.text_agent:
            print(
                f"⏭️  Skipping transcription ID {transcription.id} - Agent output already exists for model {llm_model_name}"
            )
            continue

        try:
            result = agent.correct_transcription(
                whisper_text=transcription.text,
                transcription_id=transcription.id,
                model_name=llm_model_name,
                initial_strategy=initial_strategy,
            )
            corrected_text = result.get("corrected_text", transcription.text)
            method_used = result.get("method", "unknown")
            quality = result.get("quality", {})

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

            if existing_output:
                llm_output = llm_output_repository.update_llm_output(
                    db=db, llm_output_id=existing_output.id, **update_params
                )
            else:
                llm_output = llm_output_repository.create_llm_output(
                    db=db, **create_params
                )

            print(
                f"✅ transcription ID {transcription.id} -> LLM output ID {llm_output.id} "
                f"(Agent, method: {method_used}, quality: {quality.get('confidence', 'unknown')}): "
                f"{corrected_text[:50]}..."
            )
            processed_count += 1
        except Exception as e:
            print(f"❌ Failed to correct transcription ID {transcription.id}: {e}")

    print(f"✅ Processed {processed_count} transcriptions with Agent")
