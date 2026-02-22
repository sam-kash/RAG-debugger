import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from chunker import load_documents
from vector_store import add_documents, query_vector_store

# Setup

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])

st.title("🧠 RAG Debug Visualizer")

@st.cache_resource
def setup_vector_store():
    docs = load_documents()
    add_documents(docs)

setup_vector_store()

# User Input

question = st.text_input("Ask a question")

top_k = st.slider("Top K Results", 1, 10, 5)
threshold = st.slider("Similarity Threshold (lower = stricter)", 0.0, 3.0, 1.8)

if question:
    results = query_vector_store(question, top_k=top_k)

    retrieved_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_data = []

    for i in range(len(retrieved_chunks)):
        retrieved_data.append({
            "chunk": retrieved_chunks[i],
            "source": sources[i]["source"],
            "distance": distances[i]
        })

    retrieved_data = sorted(retrieved_data, key=lambda x: x["distance"])

    st.subheader(" Retrieved Chunks")

    filtered_chunks = []

    for item in retrieved_data:
        if item["distance"] < threshold:
            filtered_chunks.append(item["chunk"])

        with st.expander(
            f"Source: {item['source']} | Distance: {round(item['distance'], 3)}"
        ):
            st.write(item["chunk"])

    if not filtered_chunks:
        st.warning("No relevant documents found.")
    else:
        context = "\n".join(filtered_chunks)

        prompt = f"""
        Answer strictly using the provided context.
        If not found, say 'Not found in provided documents.'

        Context:
        {context}

        Question:
        {question}
        """

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Answer only from context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        answer = response.choices[0].message.content

        st.subheader(" Final Answer")
        st.write(answer)