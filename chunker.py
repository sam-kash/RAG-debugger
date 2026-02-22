import os
import re

def semantic_chunk(text):
    # Split by sentence boundaries
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 300:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def load_documents(folder="docs"):
    documents = []

    for file in os.listdir(folder):
        with open(os.path.join(folder, file), "r") as f:
            text = f.read()
            chunks = semantic_chunk(text)

            for chunk in chunks:
                documents.append({
                    "content": chunk,
                    "source": file
                })

    return documents