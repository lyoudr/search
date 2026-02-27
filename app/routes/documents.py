from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import shutil
from pathlib import Path


from app.services.document_services import DocumentProcessor
from app.services.hematology_services import HematologyDictionaryLoader
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
    "/process/{file_path:path}",
    summary="Process an existing file in the search folder"
)
def process_existing_file(
    file_path: str,
    embedding_model: str = "openai",
    chunk_size: int = 512,
    chunk_overlap: int = 50
):
    """
    Process an existing file in the search folder.
    
    :param file_path: Path to the file relative to SOURCE_DIR (e.g., "data/blood_cancer_new.docx" or "blood_cancer.doc")
    :param embedding_model: Embedding model to use
    :param chunk_size: Size of text chunks
    :param chunk_overlap: Overlap between chunks
    """
    # Try multiple possible locations
    possible_paths = [
        Path(settings.SOURCE_DIR) / file_path,  # Relative to SOURCE_DIR (e.g., app/sources/data/blood_cancer_new.docx)
        Path(settings.SOURCE_DIR).parent / file_path,  # Relative to parent of SOURCE_DIR
        Path(file_path),  # Absolute path or current directory
    ]
    
    file_path_obj = None
    for path in possible_paths:
        if path.exists():
            file_path_obj = path
            break
    
    if not file_path_obj:
        raise HTTPException(
            status_code=404, 
            detail=f"File not found: {file_path}. Tried: {[str(p) for p in possible_paths]}"
        )
    
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


@router.post(
    "/hematology-dictionary/upload",
    summary="Upload and process hematology dictionary CSV to Pinecone"
)
def upload_hematology_dictionary(
    file: UploadFile = File(...),
    embedding_model: str = "openai",
    text_field: str = "Term",
    include_standard_term: bool = True
):
    """
    Upload a hematology dictionary CSV file and process it:
    1. Parse the CSV file
    2. Create embeddings for each entry
    3. Store embeddings in Pinecone "hematology" index
    
    :param file: CSV file to upload
    :param embedding_model: Embedding model to use ("openai" or "huggingface")
    :param text_field: Field name to use for embedding (default: "Term")
    :param include_standard_term: Whether to include Standard_Term in the text for embedding
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported for hematology dictionary"
        )
    
    # Save uploaded file temporarily
    upload_dir = Path(settings.SOURCE_DIR) / "uploads"
    upload_dir.mkdir(exist_ok=True)
    
    file_path = upload_dir / file.filename
    
    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process CSV
        loader = HematologyDictionaryLoader(embedding_model=embedding_model)
        result = loader.load_csv_to_vector_db(
            csv_file_path=str(file_path),
            text_field=text_field,
            include_standard_term=include_standard_term
        )
        
        return {
            "status": "success",
            "message": "Hematology dictionary processed and stored in Pinecone 'hematology' index",
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process hematology dictionary: {str(e)}")
    
    finally:
        # Clean up uploaded file
        if file_path.exists():
            file_path.unlink()


@router.post(
    "/hematology-dictionary/process/{file_path:path}",
    summary="Process an existing hematology dictionary CSV file"
)
def process_hematology_dictionary(
    file_path: str,
    embedding_model: str = "openai",
    text_field: str = "Term",
    include_standard_term: bool = True
):
    """
    Process an existing hematology dictionary CSV file in the search folder.
    
    :param file_path: Path to the CSV file relative to SOURCE_DIR (e.g., "data/hematology_dictionary.csv")
    :param embedding_model: Embedding model to use
    :param text_field: Field name to use for embedding (default: "Term")
    :param include_standard_term: Whether to include Standard_Term in the text for embedding
    """
    # Try multiple possible locations
    possible_paths = [
        Path(settings.SOURCE_DIR) / file_path,  # Relative to SOURCE_DIR
        Path(settings.SOURCE_DIR).parent / file_path,  # Relative to parent of SOURCE_DIR
        Path(file_path),  # Absolute path or current directory
    ]
    
    file_path_obj = None
    for path in possible_paths:
        if path.exists():
            file_path_obj = path
            break
    
    if not file_path_obj:
        raise HTTPException(
            status_code=404, 
            detail=f"File not found: {file_path}. Tried: {[str(p) for p in possible_paths]}"
        )
    
    if not file_path_obj.suffix.lower() == '.csv':
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported for hematology dictionary"
        )
    
    try:
        # Process CSV
        loader = HematologyDictionaryLoader(embedding_model=embedding_model)
        result = loader.load_csv_to_vector_db(
            csv_file_path=str(file_path_obj),
            text_field=text_field,
            include_standard_term=include_standard_term
        )
        
        return {
            "status": "success",
            "message": "Hematology dictionary processed and stored in Pinecone 'hematology' index",
            "result": result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process hematology dictionary: {str(e)}")





