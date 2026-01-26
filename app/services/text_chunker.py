"""
Text Chunking Service
Handles splitting documents into smaller chunks for embedding
"""
from typing import List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback to simple text splitter if langchain not available
    RecursiveCharacterTextSplitter = None


class TextChunker:
    """Service for chunking text documents"""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize text chunker.
        
        :param chunk_size: Maximum size of each chunk (in characters)
        :param chunk_overlap: Number of characters to overlap between chunks
        :param separators: List of separators to use for splitting (default: ["\n\n", "\n", " ", ""])
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default separators for Chinese and English text
        if separators is None:
            separators = ["\n\n", "\n", "。", "！", "？", " ", ""]
        
        if RecursiveCharacterTextSplitter is None:
            raise ImportError("langchain is required. Install with: pip install langchain")
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[dict]:
        """
        Split text into chunks.
        
        :param text: Text to chunk
        :param metadata: Optional metadata to add to each chunk
        :return: List of chunk dictionaries with 'text' and 'metadata' keys
        """
        chunks = self.splitter.split_text(text)
        
        result = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                'text': chunk,
                'metadata': {
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    **(metadata or {})
                }
            }
            result.append(chunk_data)
        
        return result
    
    def chunk_documents(self, documents: List[dict]) -> List[dict]:
        """
        Chunk multiple documents.
        
        :param documents: List of documents, each with 'text' and optional 'metadata'
        :return: List of chunked documents
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk_text(doc['text'], doc.get('metadata'))
            all_chunks.extend(chunks)
        
        return all_chunks

