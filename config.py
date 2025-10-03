import faiss
import numpy as np
import json
from pathlib import Path
import requests

########################################
# Define paths to your index and metadata files
INDEX_PATH = "my_faiss_index.faiss"
METADATA_PATH = "my_metadata.json"
#embedding url
EMBEDDING_URL = "http://localhost:11434/api/embeddings"

########################################


