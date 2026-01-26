"""
Document Processor Service
Orchestrates the complete pipeline: parse -> chunk -> embed -> store in Pinecone
"""
from typing import List, Dict, Optional
import os

from app.services.document_parser import parse_document
from app.services.text_chunker import TextChunker
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


class DocumentProcessor:
    """Main service for processing documents and storing in Pinecone"""
    
    def __init__(
        self,
        embedding_model: str = "openai",
        chunk_size: int = 128,
        chunk_overlap: int = 20,
        pinecone_index_name: str = "medical-documents"
    ):
        """
        Initialize document processor.
        
        :param embedding_model: Embedding model to use ("openai" or "huggingface")
        :param chunk_size: Size of text chunks
        :param chunk_overlap: Overlap between chunks
        :param pinecone_index_name: Name of Pinecone index
        """
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.pinecone_service = PineconeService(index_name=pinecone_index_name)
    
    def process_document(
        self,
        file_path: str,
        document_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process a document: parse, chunk, embed, and store in Pinecone.
        
        :param file_path: Path to the document file
        :param document_metadata: Optional metadata to add to all chunks
        :return: Processing result with statistics
        """
        # Step 1: Parse document
        print(f"📄 Parsing document: {file_path}")
        text = parse_document(file_path)
        print(f"✅ Extracted {len(text)} characters from document")
        
        # Step 2: Chunk text
        print(f"✂️  Chunking text...")
        chunks = self.chunker.chunk_text(
            text,
            metadata={
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                **(document_metadata or {})
            }
        )
        print(f"✅ Created {len(chunks)} chunks")
        
        # Step 3: Create embeddings
        print(f"🔢 Creating embeddings...")
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_service.embed_batch(texts)
        print(f"✅ Created {len(embeddings)} embeddings")
        
        # Step 4: Store in Pinecone
        print(f"💾 Storing in Pinecone...")
        metadatas = [chunk['metadata'] for chunk in chunks]
        self.pinecone_service.upsert_vectors(
            vectors=embeddings,
            texts=texts,
            metadatas=metadatas
        )
        print(f"✅ Stored {len(embeddings)} vectors in Pinecone")
        
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'text_length': len(text),
            'num_chunks': len(chunks),
            'num_embeddings': len(embeddings),
            'status': 'success'
        }
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents in Pinecone.
        
        :param query: Search query text
        :param top_k: Number of results to return
        :param filter: Optional metadata filter
        :return: List of search results
        """
        # Create embedding for query
        query_embedding = self.embedding_service.embed_text(query)
        
        # Search in Pinecone
        results = self.pinecone_service.query(
            query_vector=query_embedding,
            top_k=top_k,
            filter=filter
        )
        
        return results

