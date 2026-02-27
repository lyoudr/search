"""
Medical-documents related services in one place:
- medical document retriever
- medical term vector store (query-index)
"""

import uuid
import re
from typing import Dict, List, Optional, Set

from app.services.embedding_service import EmbeddingService
from app.services.model_manager import model_manager
from app.services.pinecone_service import PineconeService


class MedicalDocumentRetriever:
    """Retrieve relevant documents from medical-documents using query-index terms."""

    def __init__(
        self,
        embedding_model: str = "openai",
        query_index_name: str = "query-index",
        document_index_name: str = "medical-documents",
    ):
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.query_index = PineconeService(index_name=query_index_name)
        self.document_index = PineconeService(index_name=document_index_name)

    def retrieve_documents_for_correction(
        self,
        transcription_id: int,
        transcription_text: str,
        top_k_queries: int = 2,
        top_k_documents: int = 3,
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

            all_document_texts = []
            seen_texts = set()
            for term in terms:
                term_embedding = self.embedding_service.embed_text(term)
                doc_results = self.document_index.query(
                    query_vector=term_embedding,
                    top_k=top_k_documents,
                    include_metadata=True,
                )
                for result in doc_results:
                    text = result.get("metadata", {}).get("text")
                    if text and text not in seen_texts:
                        all_document_texts.append(text)
                        seen_texts.add(text)

            print(
                f"✅ Retrieved {len(all_document_texts)} unique documents from "
                f"medical-documents (ground truth)"
            )
            return all_document_texts[: top_k_documents * top_k_queries]
        except Exception as e:
            print(f"⚠️  Failed to retrieve documents: {e}")
            return []


class MedicalTermVectorStore:
    """Store and retrieve medical terms as vectors in query-index."""

    def __init__(self, embedding_model: str = "openai", query_index_name: str = "query-index"):
        self.embedding_service = EmbeddingService(model_name=embedding_model)
        self.pinecone_service = PineconeService(index_name=query_index_name)

    def store_terms(
        self,
        terms: List[str],
        transcription_id: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ):
        if not terms:
            return

        embeddings = self.embedding_service.embed_batch(terms)
        metadatas = []
        ids = []
        for i, term in enumerate(terms):
            term_metadata = {
                "term": term,
                "type": "medical_term",
                **(metadata or {}),
            }
            if transcription_id is not None:
                term_metadata["transcription_id"] = int(transcription_id)
            metadatas.append(term_metadata)
            ids.append(
                f"term_{transcription_id}_{i}_{uuid.uuid4().hex[:8]}"
                if transcription_id
                else f"term_{uuid.uuid4().hex}"
            )

        self.pinecone_service.upsert_vectors(
            vectors=embeddings,
            texts=terms,
            metadatas=metadatas,
            ids=ids,
        )
        print(f"✅ Stored {len(terms)} medical terms in Pinecone")

    def retrieve_similar_terms(
        self,
        query: str,
        top_k: int = 10,
        filter: Optional[Dict] = None,
    ) -> List[Dict]:
        query_embedding = self.embedding_service.embed_text(query)
        return self.pinecone_service.query(
            query_vector=query_embedding,
            top_k=top_k,
            filter=filter or {"type": "medical_term"},
        )

    def get_relevant_terms_for_correction(
        self,
        transcription_text: str,
        top_k: int = 20,
    ) -> List[str]:
        results = self.retrieve_similar_terms(query=transcription_text, top_k=top_k)
        terms = []
        seen = set()
        for result in results:
            term = result.get("metadata", {}).get("term")
            if term and term not in seen:
                terms.append(term)
                seen.add(term)
        return terms


class MedicalTermExtractor:
    """Extract medical terminology from transcriptions using LLM."""

    def extract_terms(self, text: str, model_name: str = "gpt-5.2") -> List[str]:
        prompt = (
            "你是一位醫療術語專家。請從以下醫療轉錄文本中提取所有醫療專有術語。\n"
            "請只提取專業的醫療術語、疾病名稱、藥物名稱、檢查項目、治療方法等。\n"
            "不要提取一般詞彙或非醫療相關的詞。\n"
            "請將提取的術語以逗號分隔，每行一個術語，不要編號，不要其他說明。\n\n"
            f"轉錄文本：{text}\n\n"
            "提取的醫療術語："
        )

        try:
            response = model_manager.generate_text(
                model_name=model_name,
                prompt=prompt,
                max_length=512,
                temperature=0.1,
            )

            terms = []
            for line in response.strip().split("\n"):
                line = line.strip()
                line = re.sub(r"^\d+[\.、．]\s*", "", line)
                for term in line.split(","):
                    term = term.strip()
                    if term and len(term) > 1:
                        terms.append(term)

            seen = set()
            unique_terms = []
            for term in terms:
                if term not in seen:
                    seen.add(term)
                    unique_terms.append(term)

            return unique_terms
        except Exception as e:
            raise ValueError(f"Failed to extract medical terms with LLM: {e}")

    def extract_unique_terms(
        self, texts: List[str], model_name: str = "gpt-5.2"
    ) -> Set[str]:
        all_terms = set()
        for text in texts:
            terms = self.extract_terms(text, model_name)
            all_terms.update(terms)
        return all_terms
