from typing import Any, Dict, Iterable, List, Optional

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
