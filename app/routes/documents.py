from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import shutil
from pathlib import Path


from app.services.document_processor import DocumentProcessor
from app.config.settings import get_settings

router = APIRouter(tags=["documents"], prefix="/documents")
settings = get_settings()

# Initialize document processor
doc_processor = DocumentProcessor(
    embedding_model="openai",
    chunk_size=512,
    chunk_overlap=50,
    pinecone_index_name="medical-documents"
)


@router.post(
    "/upload",
    summary="Upload and process a document to Pinecone"
)
def upload_document(
    file: UploadFile = File(...),
    embedding_model: str = "openai",
    chunk_size: int = 128,
    chunk_overlap: int = 20
):
    """
    Upload a document (.doc, .docx, .txt) and process it:
    1. Parse the document to extract text
    2. Chunk the text into smaller pieces
    3. Create embeddings for each chunk
    4. Store embeddings in Pinecone
    
    :param file: Document file to upload
    :param embedding_model: Embedding model to use ("openai" or "huggingface")
    :param chunk_size: Size of text chunks
    :param chunk_overlap: Overlap between chunks
    """
    # Validate file type
    allowed_extensions = ['.doc', '.docx', '.txt']
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    upload_dir = Path(settings.SOURCE_DIR) / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        processor = DocumentProcessor(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        result = processor.process_document(str(file_path))
        
        return {
            "status": "success",
            "message": "Document processed and stored in Pinecone",
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")
    
    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()


@router.post(
    "/process/{file_name}",
    summary="Process an existing file in the search folder"
)
def process_existing_file(
    file_name: str,
    embedding_model: str = "openai",
    chunk_size: int = 512,
    chunk_overlap: int = 50
):
    """
    Process an existing file in the search folder.
    
    :param file_name: Name of the file (e.g., "blood_cancer.doc")
    :param embedding_model: Embedding model to use
    :param chunk_size: Size of text chunks
    :param chunk_overlap: Overlap between chunks
    """
    # Find file in search folder
    file_path = Path(settings.SOURCE_DIR).parent / file_name
    
    if not file_path.exists():
        # Try in current directory
        file_path = Path(file_name)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_name}")
    
    try:
        # Process document
        processor = DocumentProcessor(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        result = processor.process_document(str(file_path))
        
        return {
            "status": "success",
            "message": "Document processed and stored in Pinecone",
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@router.post(
    "/search",
    summary="Search for similar documents in Pinecone"
)
def search_documents(
    query: str,
    top_k: int = 5,
    filter: Optional[dict] = None
):
    """
    Search for similar documents in Pinecone using semantic search.
    
    :param query: Search query text
    :param top_k: Number of results to return
    :param filter: Optional metadata filter (JSON object)
    :return: List of similar documents with scores
    """
    try:
        results = doc_processor.search(
            query=query,
            top_k=top_k,
            filter=filter
        )
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "count": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")





