"""
Document-related services in one place:
- document parsing
- text chunking
- document processing pipeline
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.base_vector_loader import BaseVectorLoader


SUPPORTED_DOCUMENT_EXTENSIONS = {".doc", ".docx", ".txt"}
TEXT_FILE_ENCODINGS = ("utf-8", "gbk")


def _ensure_file_exists(file_path: str) -> Path:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path


def _parse_docx_file(file_path: Path) -> str:
    try:
        from docx import Document

        doc = Document(str(file_path))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        if paragraphs:
            return "\n".join(paragraphs)
    except Exception:
        pass

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


def _parse_doc_file_by_extension(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext == ".docx":
        return _parse_docx_file(file_path)
    if ext == ".doc":
        raise NotImplementedError(
            ".doc format is not supported directly. "
            "Please convert to .docx first (LibreOffice, Word, etc.)"
        )
    raise ValueError(f"Unsupported file format: {ext}")


def parse_txt_file(file_path: str) -> str:
    path = _ensure_file_exists(file_path)
    for encoding in TEXT_FILE_ENCODINGS:
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "text-decoder",
        b"",
        0,
        1,
        f"Unable to decode {file_path} with encodings: {TEXT_FILE_ENCODINGS}",
    )


def parse_doc_file(file_path: str) -> str:
    path = _ensure_file_exists(file_path)
    return _parse_doc_file_by_extension(path)


def parse_document(file_path: str) -> str:
    path = _ensure_file_exists(file_path)
    file_ext = path.suffix.lower()
    if file_ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(f"Unsupported file format: {file_ext}")
    if file_ext == ".txt":
        return parse_txt_file(file_path)
    return _parse_doc_file_by_extension(path)


class TextChunker:
    """Service for chunking text documents."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            separators = ["\n\n", "\n", "。", "！", "？", " ", ""]

        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain is required. Install with: pip install langchain"
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len,
        )

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[dict]:
        chunks = self.splitter.split_text(text)
        result = []
        for i, chunk in enumerate(chunks):
            result.append(
                {
                    "text": chunk,
                    "metadata": {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        **(metadata or {}),
                    },
                }
            )
        return result

    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc["text"], doc.get("metadata"))
            all_chunks.extend(chunks)
        return all_chunks


class DocumentProcessor(BaseVectorLoader):
    """Main service for processing documents and storing in Pinecone."""

    def __init__(
        self,
        embedding_model: str = "openai",
        chunk_size: int = 128,
        chunk_overlap: int = 20,
        pinecone_index_name: str = "medical-documents",
    ):
        super().__init__(
            embedding_model=embedding_model, index_name=pinecone_index_name
        )
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def process_document(
        self,
        file_path: str,
        document_metadata: Optional[Dict] = None,
    ) -> Dict:
        print(f"📄 Parsing document: {file_path}")
        text = parse_document(file_path)
        print(f"✅ Extracted {len(text)} characters from document")

        print("✂️  Chunking text...")
        chunks = self.chunker.chunk_text(
            text,
            metadata={
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                **(document_metadata or {}),
            },
        )
        print(f"✅ Created {len(chunks)} chunks")

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        vector_result = self.upsert_text_records(
            texts=texts,
            metadatas=metadatas,
            id_prefix="document_chunk",
            item_label="document chunks",
        )

        return {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "text_length": len(text),
            "num_chunks": len(chunks),
            "num_embeddings": vector_result.get("num_entries", 0),
            "status": vector_result.get("status", "success"),
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        query_embedding = self.embedding_service.embed_text(query)
        return self.pinecone_service.query(
            query_vector=query_embedding,
            top_k=top_k,
            filter=filter,
        )
