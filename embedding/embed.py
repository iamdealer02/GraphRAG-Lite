import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer
import pandas as pd
from database.chroma_db import collection

BASE_DIR = Path(__file__).resolve().parent.parent

model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)

df = pd.read_parquet(BASE_DIR / "data" / "arxiv_chunked.parquet")
print(f"Loaded {len(df)} chunks")

# filter out chunks already in ChromaDB
all_ids = df["chunk_id"].tolist()
existing = collection.get(ids=all_ids)["ids"]
existing_set = set(existing)
df = df[~df["chunk_id"].isin(existing_set)].reset_index(drop=True)
print(f"{len(df)} new chunks to embed (skipping {len(existing_set)} already in ChromaDB)")

if len(df) == 0:
    print("Nothing to do.")
else:
    # batch embed all texts at once
    texts = df["text"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    # insert in batches of 100
    records = df.to_dict(orient="records")
    BATCH = 100
    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        batch_embs = embeddings[i : i + BATCH]
        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=[{
                "chunk_id":   c["chunk_id"],
                "title":      c["metadata"]["title"],
                "categories": ", ".join(c["metadata"]["categories"]),
                "year":       int(c["metadata"]["year"]),
                "doi":        c["metadata"]["doi"] or "",
                "concepts":   ", ".join(c["metadata"]["concepts"]),
                "authors":    c["metadata"]["authors"],
            } for c in batch],
            ids=[c["chunk_id"] for c in batch],
            embeddings=[e.tolist() for e in batch_embs],
        )
        print(f"Inserted {min(i + BATCH, len(records))}/{len(records)} chunks")

    print("Done — all chunks saved to ChromaDB")
    