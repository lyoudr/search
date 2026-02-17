"""
Hematology Dictionary Loader Service
Loads hematology dictionary CSV and stores it in Pinecone vector database
"""
import os
import csv
from typing import List, Dict, Optional
import uuid

from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


class HematologyDictionaryLoader:
    """Service for loading hematology dictionary CSV into Pinecone vector database"""
    
    def __init__(
        self,
        embedding_model: str = "openai",
        index_name: str = "hematology"
    ):
        """
        Initialize hematology dictionary loader.
        
        :param embedding_model: Embedding model to use
        :param index_name: Name of Pinecone index for hematology dictionary
        """
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.pinecone_service = PineconeService(index_name=index_name)
    
    def load_csv_to_vector_db(
        self,
        csv_file_path: str,
        text_field: str = "Term",
        include_standard_term: bool = True
    ):
        """
        Load CSV file and store entries in Pinecone vector database.
        
        :param csv_file_path: Path to the CSV file
        :param text_field: Field name to use for embedding (default: "Term")
        :param include_standard_term: Whether to include Standard_Term in the text for embedding
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")
        
        print(f"📖 Reading CSV file: {csv_file_path}")
        
        # Read CSV file
        entries = []
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
        
        print(f"✅ Read {len(entries)} entries from CSV")
        
        # Prepare texts for embedding
        texts = []
        metadatas = []
        
        for entry in entries:
            # Combine Term and Standard_Term for better semantic search
            if include_standard_term and entry.get("Standard_Term"):
                text = f"{entry.get(text_field, '')} ({entry.get('Standard_Term', '')})"
            else:
                text = entry.get(text_field, '')
            
            if not text.strip():
                continue  # Skip empty entries
            
            texts.append(text)
            
            # Prepare metadata
            metadata = {
                'term': entry.get("Term", ""),
                'standard_term': entry.get("Standard_Term", ""),
                'case': entry.get("Case", ""),
                'section': entry.get("Section", ""),
                'notes': entry.get("Notes", ""),
                'type': 'hematology_dictionary'
            }
            metadatas.append(metadata)
        
        if not texts:
            print("⚠️  No valid entries found in CSV")
            return
        
        print(f"🔢 Creating embeddings for {len(texts)} entries...")
        
        # Create embeddings
        embeddings = self.embedding_service.embed_batch(texts)
        
        print(f"✅ Created {len(embeddings)} embeddings")
        
        # Generate IDs
        ids = [f"hematology_{uuid.uuid4().hex}" for _ in range(len(texts))]
        
        print(f"💾 Storing {len(embeddings)} vectors in Pinecone index 'hematology'...")
        
        # Store in Pinecone
        self.pinecone_service.upsert_vectors(
            vectors=embeddings,
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✅ Successfully loaded {len(embeddings)} entries into 'hematology' vector database")
        
        return {
            'num_entries': len(embeddings),
            'status': 'success'
        }
    
    def search_dictionary(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search the hematology dictionary.
        
        :param query: Query text
        :param top_k: Number of results to return
        :param filter: Optional metadata filter
        :return: List of matching dictionary entries
        """
        # Create embedding for query
        query_embedding = self.embedding_service.embed_text(query)
        
        # Prepare filter
        search_filter = {'type': 'hematology_dictionary'}
        if filter:
            search_filter.update(filter)
        
        # Query Pinecone
        results = self.pinecone_service.query(
            query_vector=query_embedding,
            top_k=top_k,
            filter=search_filter,
            include_metadata=True
        )
        
        return results


# Convenience function for loading the dictionary
def load_hematology_dictionary(
    csv_file_path: str = "app/sources/data/hematology_dictionary.csv",
    embedding_model: str = "openai"
):
    """
    Convenience function to load hematology dictionary CSV into vector database.
    
    :param csv_file_path: Path to CSV file (default: app/sources/data/hematology_dictionary.csv)
    :param embedding_model: Embedding model to use (default: "openai")
    """
    loader = HematologyDictionaryLoader(embedding_model=embedding_model)
    return loader.load_csv_to_vector_db(csv_file_path)


if __name__ == "__main__":
    # Example usage
    import sys
    
    # Get CSV file path from command line or use default
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "app/sources/data/hematology_dictionary.csv"
    
    # Load dictionary
    result = load_hematology_dictionary(csv_file_path=csv_path)
    print(f"\n✅ Loading complete: {result}")

