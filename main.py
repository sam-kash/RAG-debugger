import os
from dotenv import load_dotenv
from groq import Groq

from chunker import load_documents
from vector_store import add_documents, query_vector_store



#  Load ENV + Groq Client


load_dotenv()

client = Groq(
    api_key=os.environ["GROQ_API_KEY"]  # force error if missing
)

#  Load Docs + Add to Vector Store

print("Loading documents...")
docs = load_documents()

print("Adding documents to vector store...")
add_documents(docs)

print("\n RAG Debug Console Ready\n")



#  Main Loop


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

    # ---- Call GROQ ----
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # recommended Groq model
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
    print("\n Sources Used:")
    unique_sources = list(set([s["source"] for s in sources]))
    for src in unique_sources:
        print("-", src)

    print("\n" + "=" * 60 + "\n")