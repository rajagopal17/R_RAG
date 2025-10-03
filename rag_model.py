import os
import faiss
import numpy as np
import json
from pathlib import Path
import requests
#######################################
from google import genai
from dotenv import load_dotenv
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=gemini_api_key)

# Define paths to your index and metadata files
INDEX_PATH = "my_faiss_index.faiss"
METADATA_PATH = "my_metadata.json"
#embedding url
EMBEDDING_URL = "http://localhost:11434/api/embeddings"

def load_index(index_path):
    """Loads the FAISS index from the specified path."""
    index = faiss.read_index(index_path)
    return index

def load_metadata(metadata_path):
    """Loads the metadata from the specified JSON file."""
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata

def get_embedding(text: str, embedding_url: str) -> np.ndarray:
    """Gets the embedding for a given text using the Ollama API."""
    response = requests.post(
        embedding_url,
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)

def search_index(index, metadata, query, top_k=5, embedding_url = EMBEDDING_URL):
    """
    Searches the FAISS index for the given query and returns the top_k results.

    Args:
        index: The FAISS index.
        metadata: The metadata associated with the index.
        query: The query string.
        top_k: The number of results to return.

    Returns:
        A list of dictionaries, where each dictionary contains the document name,
        chunk, and chunk ID of the top_k results.
    """
    query_embedding = get_embedding(query, embedding_url).reshape(1, -1)  # Reshape for FAISS
    D, I = index.search(query_embedding, top_k)  # D: distances, I: indices

    results = []
    for idx in I[0]:  # Iterate through the top_k indices
        results.append(metadata[idx])  # Retrieve metadata for each result
    return results

if __name__ == '__main__':
    # Load the index and metadata
    index = load_index(INDEX_PATH)
    metadata = load_metadata(METADATA_PATH)
     # input query into the terminal
    query = input("Enter your query: ")
    top_k = 10  # Number of results to retrieve

    # Search the index
    results = search_index(index, metadata, query, top_k)
    ##################################################
    ###pass the results to LLM for answer generation:
    response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents= f'''as a intelligent assistant provide answer to the query based 
    on the context provided in the below text. 
    \n" + {query} + "\n\n" + "Context: " + str({results})
    \n\n" + "Answer:"
    '''
 )
    print("Answer: ", response.text)
    ##################################################
