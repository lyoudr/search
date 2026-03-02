from sqlalchemy.orm import Session

from app.repositories import llm_output_repository, transcription_repository
from app.services.model_manager import model_manager


class LLMAgentTool:
    """Base tool class for LLMAgent."""

    name: str = "base_llm_tool"

    def execute(self, **kwargs) -> dict:
        raise NotImplementedError


class BatchCorrectWhisperTextTool(LLMAgentTool):
    name = "batch_correct_whisper_text"

    def execute(
        self,
        db: Session,
        llm_model_name: str = "gpt-5.2",
        prompt_version: str = "v1",
        limit: int = 10,
        use_rag: bool = False,
        top_k_queries: int = 3,
        top_k_documents: int = 5,
    ) -> dict:
        from app.services.correction_core import correct_whisper_text

        llm_model_id = model_manager.ensure_model_in_db(db, llm_model_name)
        transcriptions = transcription_repository.get_all_transcriptions(db)
        processed_count = 0

        for transcription in transcriptions[:limit]:
            existing_output = llm_output_repository.get_llm_output_by_transcription_and_model(
                db, transcription.id, llm_model_id
            )
            field_name = "text_with_rag" if use_rag else "text"

            if existing_output and getattr(existing_output, field_name, None):
                mode_str = "RAG" if use_rag else "without RAG"
                print(
                    f"⏭️  Skipping transcription ID {transcription.id} - "
                    f"{mode_str} output already exists for model {llm_model_name}"
                )
                continue

            try:
                corrected_text = correct_whisper_text(
                    transcription.text,
                    model_name=llm_model_name,
                    use_rag=use_rag,
                    transcription_id=transcription.id,
                    top_k_queries=top_k_queries,
                    top_k_documents=top_k_documents,
                )

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

                if existing_output:
                    llm_output = llm_output_repository.update_llm_output(
                        db=db, llm_output_id=existing_output.id, **update_params
                    )
                else:
                    llm_output = llm_output_repository.create_llm_output(
                        db=db, **create_params
                    )
                mode_str = "RAG" if use_rag else "without RAG"
                print(
                    f"✅ transcription ID {transcription.id} -> LLM output ID "
                    f"{llm_output.id} ({mode_str}): {corrected_text[:50]}..."
                )
                processed_count += 1
            except Exception as e:
                print(f"❌ Failed to correct transcription ID {transcription.id}: {e}")

        return {"processed_count": processed_count}


class BatchCorrectWhisperTextWithHematologyTool(LLMAgentTool):
    name = "batch_correct_whisper_text_with_hematology"

    def execute(
        self,
        db: Session,
        llm_model_name: str = "gpt-5.2",
        prompt_version: str = "v1",
        limit: int = 10,
        top_k_queries: int = 2,
        top_k: int = 5,
    ) -> dict:
        from app.services.correction_core import correct_whisper_text_with_hematology

        llm_model_id = model_manager.ensure_model_in_db(db, llm_model_name)
        transcriptions = transcription_repository.get_all_transcriptions(db)
        processed_count = 0

        for transcription in transcriptions[:limit]:
            existing_output = llm_output_repository.get_llm_output_by_transcription_and_model(
                db, transcription.id, llm_model_id
            )
            if existing_output and existing_output.text_with_hematology:
                print(
                    f"⏭️  Skipping transcription ID {transcription.id} - Hematology "
                    f"Dictionary output already exists for model {llm_model_name}"
                )
                continue

            try:
                corrected_text = correct_whisper_text_with_hematology(
                    transcription.text,
                    model_name=llm_model_name,
                    transcription_id=transcription.id,
                    top_k_queries=top_k_queries,
                    top_k=top_k,
                )
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

                if existing_output:
                    llm_output = llm_output_repository.update_llm_output(
                        db=db, llm_output_id=existing_output.id, **update_params
                    )
                else:
                    llm_output = llm_output_repository.create_llm_output(
                        db=db, **create_params
                    )
                print(
                    f"✅ transcription ID {transcription.id} -> LLM output ID "
                    f"{llm_output.id} (Hematology): {corrected_text[:50]}..."
                )
                processed_count += 1
            except Exception as e:
                print(f"❌ Failed to correct transcription ID {transcription.id}: {e}")

        return {"processed_count": processed_count}


