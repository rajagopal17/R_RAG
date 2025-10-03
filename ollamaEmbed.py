import os
from pathlib import Path
import faiss
import numpy as np
import requests
import json     
import time
import ollama

# -- CONFIG --
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
INDEX_PATH = "my_faiss_index.faiss"  # Define where to save the index
METADATA_PATH = "my_metadata.json"    # Define where to save the metadata

##########################################
import os
from google import genai
from dotenv import load_dotenv
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)
###########################################

#create a function to chunk text
def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    step = size - overlap
    return [
        " ".join(words[i:i + size])
        for i in range(0, len(words), step)
        if words[i:i + size]
    ]
#create a function to get embeddings from ollama API:
def get_embedding(text: str) -> np.ndarray:
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)


#Read the file path from environment variable
file_path = "C:\\mygitrepo\\R_RAG\\fixedassets.txt"
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()
# Chunk the text
chunks = chunk_text(text)
#print(f"Total chunks created: {len(chunks)}")
#print(f"Sample chunk: {chunks[:50]}")
# Create a function to embed each chunk:
def embed_chunks(chunks):
    all_embeddings = []
    metadata = []
    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        all_embeddings.append(embedding)
        metadata.append({
            "chunk": chunk,
            "chunk_id": f"{idx}"
        })
        time.sleep(0.5)  # Sleep to avoid hitting rate limits
    return all_embeddings, metadata
# Embed the chunks
all_embeddings, metadata = embed_chunks(chunks)
#print(f"Embeddings shape: {all_embeddings}")
#print(f"Sample embedding: {all_embeddings[0][:5]}")  # Print first 5 dimensions of the first embedding
print(f"Metadata sample: {metadata[0]}")  # Print first metadata entry

######################################
# Create and save the FAISS index
dimension = len(all_embeddings[0])  # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension)
index.add(np.array(all_embeddings))
faiss.write_index(index, INDEX_PATH)

########################################
# Save metadata to a JSON file:
with open(METADATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, ensure_ascii=False, indent=4)
print(f"FAISS index saved to {INDEX_PATH}")

