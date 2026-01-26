"""
Document Parser Service
Handles parsing of various document formats (.doc, .docx, .pdf, .txt)
"""
import os
from pathlib import Path
import subprocess
import tempfile

def parse_doc_file(file_path: str) -> str:
    """
    Parse a Word document (.docx or .doc) and extract clean text.

    Strategy:
    - .docx → python-docx
    - .doc  → Not directly supported; raise informative error
             (recommend converting to .docx first)
    - Fallback: mammoth for cleaner text output

    :param file_path: Path to the Word file
    :return: Extracted text content
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    # ---------- .docx ----------
    if ext == ".docx":
        # Try python-docx first
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            if paragraphs:
                return "\n".join(paragraphs)
        except Exception:
            pass

        # Fallback to mammoth for cleaner text
        try:
            import mammoth
            with open(file_path, "rb") as f:
                result = mammoth.extract_raw_text(f)
            text = result.value.strip()
            if text:
                return text
        except Exception:
            pass

        raise RuntimeError(f"Failed to parse .docx file: {file_path}")

    # ---------- .doc ----------
    if ext == ".doc":
        raise NotImplementedError(
            ".doc format is not supported directly. "
            "Please convert to .docx first (LibreOffice, Word, etc.)"
        )

    # ---------- Unsupported ----------
    raise ValueError(f"Unsupported file format: {file_path}")



def parse_txt_file(file_path: str) -> str:
    """
    Parse a .txt file and extract text content.
    
    :param file_path: Path to the .txt file
    :return: Extracted text content
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, 'r', encoding='gbk') as f:
            return f.read()


def parse_document(file_path: str) -> str:
    """
    Parse a document file based on its extension.
    Supports: .doc, .docx, .txt
    
    :param file_path: Path to the document file
    :return: Extracted text content
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.txt':
        return parse_txt_file(file_path)
    elif file_ext in ['.doc', '.docx']:
        return parse_doc_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")

