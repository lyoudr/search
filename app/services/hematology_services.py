"""
Hematology-related services in one place:
- hematology dictionary loader
- hematology vocabulary loader
- hematology retriever
"""

import csv
import glob
import os
from typing import Dict, List, Optional

from app.services.base_vector_loader import BaseVectorLoader
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService


class HematologyDictionaryLoader(BaseVectorLoader):
    """Service for loading hematology dictionary CSV into Pinecone vector database."""

    def __init__(self, embedding_model: str = "openai", index_name: str = "hematology"):
        super().__init__(embedding_model=embedding_model, index_name=index_name)

    def load_csv_to_vector_db(
        self,
        csv_file_path: str,
        text_field: str = "Term",
        include_standard_term: bool = True,
    ):
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        print(f"📖 Reading CSV file: {csv_file_path}")
        entries = []
        with open(csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)

        print(f"✅ Read {len(entries)} entries from CSV")
        texts = []
        metadatas = []
        for entry in entries:
            if include_standard_term and entry.get("Standard_Term"):
                text = f"{entry.get(text_field, '')} ({entry.get('Standard_Term', '')})"
            else:
                text = entry.get(text_field, "")
            if not text.strip():
                continue
            texts.append(text)
            metadatas.append(
                {
                    "term": entry.get("Term", ""),
                    "standard_term": entry.get("Standard_Term", ""),
                    "case": entry.get("Case", ""),
                    "section": entry.get("Section", ""),
                    "notes": entry.get("Notes", ""),
                    "type": "hematology_dictionary",
                }
            )

        if not texts:
            print("⚠️  No valid entries found in CSV")
            return {"num_entries": 0, "status": "no_entries"}

        return self.upsert_text_records(
            texts=texts,
            metadatas=metadatas,
            id_prefix="hematology",
            item_label="dictionary entries",
        )

    def search_dictionary(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        query_embedding = self.embedding_service.embed_text(query)
        search_filter = {"type": "hematology_dictionary"}
        if filter:
            search_filter.update(filter)
        return self.pinecone_service.query(
            query_vector=query_embedding,
            top_k=top_k,
            filter=search_filter,
            include_metadata=True,
        )


class HematologyVocabLoader(BaseVectorLoader):
    """Service for loading hematology / oncology vocabulary text files."""

    def __init__(self, embedding_model: str = "openai", index_name: str = "hematology-vocab"):
        super().__init__(embedding_model=embedding_model, index_name=index_name)

    def _read_vocab_files(
        self,
        pattern: str = "app/sources/data/hematology_oncology_vocab_5000_*.txt",
    ) -> List[Dict]:
        file_paths = sorted(glob.glob(pattern))
        if not file_paths:
            print(f"⚠️  No vocab files found matching pattern: {pattern}")
            return []

        print(f"📂 Found {len(file_paths)} vocab files:")
        for p in file_paths:
            print(f"   - {p}")

        entries: List[Dict] = []
        for file_path in file_paths:
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
        entries = self._read_vocab_files(pattern=pattern)
        if not entries:
            print("⚠️  No vocab entries to load")
            return {"num_entries": 0, "status": "no_entries"}

        texts = [e["text"] for e in entries]
        metadatas = [e["metadata"] for e in entries]
        return self.upsert_text_records(
            texts=texts,
            metadatas=metadatas,
            id_prefix="hematology_vocab",
            item_label="vocab terms",
        )


class HematologyRetriever:
    """Service for retrieving entries from hematology dictionary index."""

    def __init__(
        self,
        embedding_model: str = "openai",
        query_index_name: str = "query-index",
        hematology_index_name: str = "hematology",
    ):
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.query_index = PineconeService(index_name=query_index_name)
        self.hematology_index = PineconeService(index_name=hematology_index_name)

    def retrieve_entries_for_correction(
        self,
        transcription_id: int,
        transcription_text: str,
        top_k_queries: int = 2,
        top_k: int = 5,
    ) -> List[str]:
        try:
            transcription_embedding = self.embedding_service.embed_text(transcription_text)
            query_results = self.query_index.query(
                query_vector=transcription_embedding,
                top_k=top_k_queries * 2,
                filter={
                    "transcription_id": {"$eq": int(transcription_id)},
                    "type": {"$eq": "medical_term"},
                },
                include_metadata=True,
            )
            print(
                f"🔍 Query results for transcription_id={transcription_id}: "
                f"{len(query_results)} results"
            )

            terms = []
            for result in query_results:
                term = result.get("metadata", {}).get("term")
                if term and term not in terms:
                    terms.append(term)
                    if len(terms) >= top_k_queries:
                        break

            if not terms:
                print(
                    f"⚠️  No medical terms found in query_index for transcription "
                    f"{transcription_id}"
                )
                return []

            print(
                f"🔍 Found {len(terms)} queries from query_index "
                f"(transcription_id={transcription_id}): {', '.join(terms)}"
            )
            all_entries = []
            seen_entries = set()
            for term in terms:
                term_embedding = self.embedding_service.embed_text(term)
                hematology_results = self.hematology_index.query(
                    query_vector=term_embedding,
                    top_k=top_k,
                    include_metadata=True,
                )
                for result in hematology_results:
                    text = result.get("metadata", {}).get("text", "")
                    if not text:
                        term_val = result.get("metadata", {}).get("term", "")
                        standard_term = result.get("metadata", {}).get("standard_term", "")
                        text = f"{term_val} ({standard_term})" if standard_term else term_val
                    if text and text not in seen_entries:
                        all_entries.append(text)
                        seen_entries.add(text)

            print(
                f"✅ Retrieved {len(all_entries)} unique entries from hematology dictionary"
            )
            return all_entries[: top_k * top_k_queries]
        except Exception as e:
            print(f"⚠️  Failed to retrieve entries from hematology dictionary: {e}")
            return []


def load_hematology_dictionary(
    csv_file_path: str = "app/sources/data/hematology_dictionary.csv",
    embedding_model: str = "openai",
):
    loader = HematologyDictionaryLoader(embedding_model=embedding_model)
    return loader.load_csv_to_vector_db(csv_file_path)


def load_hematology_vocab(
    pattern: str = "app/sources/data/hematology_oncology_vocab_5000_*.txt",
    embedding_model: str = "openai",
    index_name: str = "hematology-vocab",
):
    loader = HematologyVocabLoader(embedding_model=embedding_model, index_name=index_name)
    return loader.load_vocab_to_vector_db(pattern=pattern)
