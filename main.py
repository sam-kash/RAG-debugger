import os
from dotenv import load_dotenv
from openai import OpenAI

from chunker import load_documents
from vector_store import add_documents, query_vector_store

# Load env

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROK_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# load docs

print("Loading documents...")
docs = load_documents()

print("Adding documents to vector store...")
add_documents(docs)

print("\n🚀 RAG Debug Console Ready\n")

# Main loop

while True:
    question = input("Ask a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    # ---- Retrieve ----
    results = query_vector_store(question, top_k=3)

    retrieved_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    distances = results["distances"][0]

    print("\n Retrieved Chunks:\n")

    for i, chunk in enumerate(retrieved_chunks):
        print(f"Chunk {i+1}")
        print(f"Source: {sources[i]['source']}")
        print(f"Similarity Score (lower = better): {distances[i]}")
        print(chunk)
        print("-" * 50)

    # ---- Build Context ----
    context = "\n".join(retrieved_chunks)

    prompt = f"""
Answer strictly using the provided context.
If the answer is not found in the context, say:
"Not found in provided documents."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="grok-2-latest",   # change if needed
        messages=[
            {
                "role": "system",
                "content": "You answer only from given context. Do not use outside knowledge."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    answer = response.choices[0].message.content

    print("\n Final Answer:\n")
    print(answer)

    # ---- Source Citation ----
    print("\n Sources Used:")
    unique_sources = list(set([s["source"] for s in sources]))
    for src in unique_sources:
        print("-", src)

    print("\n" + "=" * 60 + "\n")