from openai import OpenAI
from sqlalchemy.orm import Session

from app.repositories import audio_repository

client = OpenAI()


def correct_whisper_text_gpt4(whisper_text: str) -> str:
    prompt = (
        "你是一位醫療語句格式化助理，請根據以下段落修正口語醫療語句，使其語法正確：\n"
        "1. 不補上標點符號\n"
        "2. 只修正詞彙錯誤\n\n"
        f"原文：{whisper_text}\n"
        f"修正："
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()


# Use Chain of Thought prompting means "breaking down complex problems into smaller, logic steps"
# Just like humans do when solving multi-part problems
def chain_of_thought_gpt4(db: Session, your_new_input:str) -> str:

    examples = audio_repository.get_examples_from_db_for_cot(db, limit=5)

    prompt = (
        "你是一位醫療語句格式化助理，請依照以下步驟，將口語醫療語句轉換成有標點、用詞正確、語意清楚的書面紀錄風格：\n"
        "步驟：\n"
        "1. 補上標點符號\n"
        "2. 修正文法詞彙錯誤\n"
    )

    for example in examples:
        prompt += (
            f"原文：{example['input']}\n"
            f"推理過程：{example['reasoning']}\n"
            f"修正：{example['output']}\n\n"
        )

    prompt += (
        f"原文：{your_new_input}\n"
        f"推理過程："
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages = [
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def batch_correct_whisper_text_with_gpt4(db: Session, limit: int = 10):
    records = audio_repository.get_whisper_text_from_db(db, limit)
    for record in records:
        try:
            corrected = correct_whisper_text_gpt4(record.whisper_text)
            record.llm_text = corrected
            print(f"✅ Corrected ID {record.id}: {corrected}")
        except Exception as e:
            print(f"❌ Failed to correct ID {record.id}: {e}")
    
    db.commit()


