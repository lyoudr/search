from sqlalchemy.orm import Session
from typing import List

from app.models.analyze import Analyze 
from app.schemas.analyze import AnalyzeBase
from app.services.wer import wer


def get_examples_from_db_for_cot(db: Session, limit: int = 10) -> List[dict]:
    records = db.query(Analyze).filter(
        Analyze.whisper_text.isnot(None),
        Analyze.ground_truth.isnot(None)
    ).limit(limit).all()

    EXAMPLES = [
        {
            "input": record.whisper_text,
            "reasoning": (
                "1. 補上漏掉的標點符號\n"
                "2. 修正文法詞彙錯誤\n"
            ),
            "output": record.ground_truth
        }
        for record in records
    ]
    return EXAMPLES

def get_whisper_text_from_db(db: Session, limit: int = 10) -> List[str]:
    records = db.query(Analyze).filter(
        Analyze.whisper_text.isnot(None)
    ).limit(limit).all()

    return records

def get_whisper_analyze_records(db: Session):
    records = db.query(Analyze).filter(
        Analyze.ground_truth.isnot(None),
        Analyze.whisper_text.isnot(None),
    ).all()
    return records


def update_whisper_analyze_wer(db: Session, records: List[Analyze]):
    for record in records:
        record.whisper_wer = float(wer(record.whisper_text, record.ground_truth))
    db.commit()

def get_llm_analyze_records(db: Session):
    records = db.query(Analyze).filter(
        Analyze.ground_truth.isnot(None),
        Analyze.llm_text.isnot(None),
    ).all()
    return records

def update_llm_analyze_wer(db: Session, records: List[Analyze]):
    for record in records:
        record.llm_wer = float(wer(record.llm_text, record.ground_truth))
    db.commit()

def create_analyze_record(db: Session, payload: AnalyzeBase) -> Analyze:
    """
    Create a new analyze record in the database.
    
    :param db: Database session
    :param payload: AnalyzeBase schema containing the data to be stored
    :return: The created Analyze model instance
    """
    analyze_record = Analyze(
        file_path=payload.file_path,
        whisper_text=payload.whisper_text,
        llm_text=payload.llm_text,
        ground_truth=payload.ground_truth
    )
    db.add(analyze_record)
    db.commit()
    db.refresh(analyze_record)
    return analyze_record


def check_file_exist(db: Session, file_path: str) -> bool:
    return db.query(Analyze).filter_by(file_path = file_path).first()
