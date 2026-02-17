"""
Hematology / Oncology Vocabulary Loader
Loads hematology_oncology_vocab_5000_*.txt files into a dedicated Pinecone index.

Each non-empty line in the vocab files is treated as a separate term and stored as a vector
with simple metadata so it can be queried later.
"""

import os
import glob
from typing import List, Dict, Optional
import uuid

from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


class HematologyVocabLoader:
    """
    Service for loading hematology / oncology vocabulary text files into
    a dedicated Pinecone index.
    """

    def __init__(
        self,
        embedding_model: str = "openai",
        index_name: str = "hematology-vocab",
    ):
        """
        :param embedding_model: Embedding model to use (default: \"openai\")
        :param index_name: Name of Pinecone index to store vocab terms
        """
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.pinecone_service = PineconeService(index_name=index_name)

    def _read_vocab_files(
        self,
        pattern: str = "app/sources/data/hematology_oncology_vocab_5000_*.txt",
    ) -> List[Dict]:
        """
        Read all vocab files matching the given glob pattern.

        Returns a list of dicts with:
          - text: the term text
          - metadata: metadata dict
        """
        file_paths = sorted(glob.glob(pattern))

        if not file_paths:
            print(f"⚠️  No vocab files found matching pattern: {pattern}")
            return []

        print(f"📂 Found {len(file_paths)} vocab files:")
        for p in file_paths:
            print(f"   - {p}")

        entries: List[Dict] = []

        for file_idx, file_path in enumerate(file_paths):
            if not os.path.exists(file_path):
                print(f"⚠️  File not found (skipping): {file_path}")
                continue

            print(f"📖 Reading vocab file: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    term = line.strip()
                    if not term:
                        continue

                    entries.append(
                        {
                            "text": term,
                            "metadata": {
                                "term": term,
                                "source_file": os.path.basename(file_path),
                                "line_number": line_idx + 1,
                                "type": "hematology_vocab",
                            },
                        }
                    )

        print(f"✅ Collected {len(entries)} vocab terms from all files")
        return entries

    def load_vocab_to_vector_db(
        self,
        pattern: str = "app/sources/data/hematology_oncology_vocab_5000_*.txt",
    ):
        """
        Load all vocab files into the Pinecone index.

        :param pattern: Glob pattern for vocab files
        """
        entries = self._read_vocab_files(pattern=pattern)
        if not entries:
            print("⚠️  No vocab entries to load")
            return {"num_entries": 0, "status": "no_entries"}

        texts = [e["text"] for e in entries]
        metadatas = [e["metadata"] for e in entries]

        print(f"🔢 Creating embeddings for {len(texts)} vocab terms...")
        embeddings = self.embedding_service.embed_batch(texts)
        print(f"✅ Created {len(embeddings)} embeddings")

        # Generate IDs
        ids = [f"hematology_vocab_{uuid.uuid4().hex}" for _ in range(len(texts))]

        print(
            f"💾 Storing {len(embeddings)} vocab vectors in Pinecone index "
            f"'{self.pinecone_service.index_name}'..."
        )

        self.pinecone_service.upsert_vectors(
            vectors=embeddings,
            texts=texts,
            metadatas=metadatas,
            ids=ids,
        )

        print(
            f"✅ Successfully loaded {len(embeddings)} vocab entries into "
            f"'{self.pinecone_service.index_name}'"
        )

        return {"num_entries": len(embeddings), "status": "success"}


def load_hematology_vocab(
    pattern: str = "app/sources/data/hematology_oncology_vocab_5000_*.txt",
    embedding_model: str = "openai",
    index_name: str = "hematology-vocab",
):
    """
    Convenience function to load all hematology_oncology_vocab_5000_*.txt files
    into a dedicated Pinecone index.
    """
    loader = HematologyVocabLoader(
        embedding_model=embedding_model,
        index_name=index_name,
    )
    return loader.load_vocab_to_vector_db(pattern=pattern)


if __name__ == "__main__":
    # Allow running as a script:
    #   python -m app.services.hematology_vocab_loader
    # or python app/services/hematology_vocab_loader.py
    import sys

    # Optional: custom glob pattern or index name via CLI
    pattern_arg: Optional[str] = None
    index_name_arg: Optional[str] = None

    if len(sys.argv) > 1:
        pattern_arg = sys.argv[1]
    if len(sys.argv) > 2:
        index_name_arg = sys.argv[2]

    result = load_hematology_vocab(
        pattern=pattern_arg or "app/sources/data/hematology_oncology_vocab_5000_*.txt",
        index_name=index_name_arg or "hematology-vocab",
    )
    print(f"\n✅ Loading complete: {result}")

