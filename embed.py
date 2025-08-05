import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from sqlalchemy import create_engine, text
"""
Remember to call `huggingface-cli`
And huggingface token will be stored at ~/.cache/huggingface/token
"""
# ==== 1. Connect to PostgreSQL (pgvector enabled) ===
PG_CONN_STR = "postgresql+psycopg2://dev:eexyxqv4@192.168.50.149:5432/dr_cell?options=-csearch_path%3Dmain"
engine = create_engine(PG_CONN_STR)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

# ==== 2. Read articles from table ==== 
with engine.connect() as conn:
    result = conn.execute(text('''
        SELECT 
            a.id, 
            a.title, 
            a.summary,
            c.name AS category_name 
        FROM main.article a 
        JOIN main.category c ON a.category_id = c.id
        WHERE a.embedding = FALSE
    '''))
    rows = result.fetchall()

# ==== 3. Prepare documents ====
docs = []
for row in rows:
    doc_id, title, summary, category_name = row 
    full_text = f"{title}\n\n{summary}"
    docs.append(
        Document(
            page_content=full_text, 
            metadata={
                "id": doc_id,
                "category": category_name 
            }
        )
    )

# ==== 4. Chunk the articles ====
splitter = RecursiveCharacterTextSplitter(chunk_size = 512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# ==== 5. Load embeddings ====
embedding = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-m3"
)

# ==== 6. Store in PGVector ====
vectorstore = PGVector(
    connection_string=PG_CONN_STR,
    collection_name="article_embeddings",  # your pgvector table
    embedding_function=embedding,
)

vectorstore.add_documents(chunks)
print("✅ Embeddings added to PGVector")

# ==== 7. Update `embedding` flag ==== 
with engine.connect() as conn:
    conn.execute(text("UPDATE main.article SET embedding = TRUE WHERE embedding = FALSE"))
    conn.commit()
print("✅ Updated embedding flag in main.article")

# ==== 8. Perform Similarity Search ===
# Consine similarity (common for text embeddings)
# Euclidean distance (L2 norm)
# Inner product (dot product)

query = "外泌體如何應用在癌症治療"
results = vectorstore.similarity_search(
    query,
    k=5,
    filter={"category":"外泌體療法"}
)