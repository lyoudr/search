from typing import TypedDict
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from app.config.settings import get_settings

settings = get_settings()

pc = Pinecone(api_key=settings.PINECONE_API_KEY, environment='us-east-1')
index = pc.Index(settings.PINECONE_INDEX)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = PineconeVectorStore(index=index, embedding=embeddings)

# ✅ Define tool return type
class SearchResults(TypedDict):
    documents: list[str]

# ✅ Tool that wraps Pinecone retrieval
@tool
def vector_search_tool(query: str) -> SearchResults:
    """Search Pinecone for documents similar to the input query"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    return {"documents": [doc.page_content for doc in docs]}