from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from database.chroma_db import collection

model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)

def retrieve(query, top_k=5):
    embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    return results
