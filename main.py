import os
from dotenv import load_dotenv
from groq import Groq

from chunker import load_documents
from vector_store import add_documents, query_vector_store


# Load ENV + Groq Client
load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])


# Load Docs + Add to Vector Store
print("Loading documents...")
docs = load_documents()

print("Adding documents to vector store...")
add_documents(docs)

print("\nRAG Debug Console Ready\n")


# Main Loop
while True:
    question = input("Ask a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    # ---- Retrieve ----
    results = query_vector_store(question, top_k=5)

    retrieved_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    distances = results["distances"][0]

# Combine everything into structured objects
    retrieved_data = []

    for i in range(len(retrieved_chunks)):
        retrieved_data.append({
            "chunk": retrieved_chunks[i],
            "source": sources[i]["source"],
            "distance": distances[i]
        })

# Explicitly sort by similarity (lower distance = better)
    retrieved_data = sorted(retrieved_data, key=lambda x: x["distance"])

    print("\n Sorted Retrieved Chunks:\n")

    for i, item in enumerate(retrieved_data):
        print(f"Rank {i+1}")
        print(f"Source: {item['source']}")
        print(f"Distance: {item['distance']}")
        print(item["chunk"])
        print("-" * 50) 

    # ---- Threshold Filter ----
    filtered_chunks = []
    filtered_sources = []

    for i in range(len(distances)):
        if distances[i] < 1.8:  # tune this value
            filtered_chunks.append(retrieved_chunks[i])
            filtered_sources.append(sources[i])

    if not filtered_chunks:
        print(" No relevant documents found.")
        print("=" * 60)
        continue

    # ---- Build Context ----
    context = "\n".join(filtered_chunks)

    prompt = f"""
Answer strictly using the provided context.
If the answer is not found in the context, say:
"Not found in provided documents."

Context:
{context}

Question:
{question}
"""

    # ---- Call GROQ ----
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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

    # ---- Show Final Answer ----
    print("\n💡 Final Answer:\n")
    print(answer)

    # ---- Source Citation ----
    print("\nSources Used:")
    unique_sources = list(set([s["source"] for s in filtered_sources]))
    for src in unique_sources:
        print("-", src)

    print("\n" + "=" * 60 + "\n")