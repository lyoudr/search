# from Bio import Entrez 
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# import pinecone
# import os
# from airflow.decorators import dag, task 
# import pendulum

# # Setup
# Entrez.email = ""  # Replace with your email
# OPENAI_API_KEY = ""
# PINECONE_API_KEY = ""
# INDEX_NAME = "pubmed-index"


# handle = Entrez.esearch(db="pubmed", term="cancer", retmax=100)
# id_list = Entrez.read(handle)["IdList"]

# # Fetch full abstracts
# handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="text")
# pubmed_texts = handle.read()

# @dag(
#     schedule=None,
#     start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
#     catchup=False,
#     tags=["crawl"],
# )
# def fetch_and_write_to_pinecone():
#     # Step 1: Search PubMed
#     @task()
#     def fetch_pubmed_abstracts(term="cancer", max_results=10):
#         handle = Entrez.esearch(db="pubmed", term=term, retmax=max_results)
#         id_list = Entrez.read(handle)["IdList"]
#         handle = Entrez.efetch(db="pubmed", id=",".join(id_list), rettype="abstract", retmode="text")
#         raw_text = handle.read()
#         return raw_text.split("\n\n")
#     # Step 2: Chunk text
#     @task()
#     def chunk_documents(docs):
#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         # Convert LangChain Document objects to plain dictionaries
#         return [{"content": d.page_content} for d in splitter.create_documents(docs)]
#     # Step 3: Embed and store in Pinecone
#     @task()
#     def store_in_pinecone(chunks):
#         embedder = OpenAIEmbeddings()
#         texts = [chunk["content"] for chunk in chunks]
#         vectors = embedder.embed_documents(texts)
#         # Initialize Pinecone by creating a Pinecone object
#         # Replace 'us-west-2' with your actual Pinecone environment region
#         pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment='us-east-1')
#         if INDEX_NAME not in pinecone_client.list_indexes().names():
#             pinecone_client.create_index(INDEX_NAME, dimension=len(vectors[0]), spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1"))
        
#         index = pinecone_client.Index(INDEX_NAME)
#         items = [
#             (f"doc-{i}", vector, {"text": chunk["content"]})
#             for i, (vector, chunk) in enumerate(zip(vectors, chunks))
#         ]
#         index.upsert(items)

#     raw_abstracts = fetch_pubmed_abstracts(term="cancer", max_results=20)
#     chunks = chunk_documents(raw_abstracts)
#     store_in_pinecone(chunks)

# fetch_and_write_to_pinecone()

# # Pinecone Database:
# # Index: table in relational database
# # Namespace: Namespaces are logical partitions within a single index. 
# # They allow you to segment your data
# # into distinct subsets, enabling independent management and querying of different datasets.