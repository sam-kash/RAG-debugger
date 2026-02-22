import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("rag_collection")

def add_documents(documents):
    for i, doc in enumerate(documents):
        embedding = model.encode(doc["context"]).tolist()

        collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[doc["content"]],
            metadatas=[{"source":doc ["source"]}]
        )

def query_vector_store(query, top_k = 3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
         n_results=top_k
    )

    return results