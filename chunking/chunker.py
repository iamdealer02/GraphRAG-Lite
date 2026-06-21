import pandas as pd
df = pd.read_parquet("data/arxiv_processed.parquet")

def chunk_doc(record):
    return {
        "chunk_id": record["doc_id"],
        "text": record["text"].split("Abstract:")[-1].split("Categories:")[0].strip(),
        "metadata": {
            "doc_id":     record["doc_id"],
            "title":      record["title"],
            "categories": record["categories"],
            "year":       record["year"],
            "authors":    record["authors"],
            "doi":        record["metadata"]["doi"],
            "concepts":   record["concepts"].tolist(),
        }
    }

chunks = [chunk_doc(row) for _, row in df.iterrows()]

pd.DataFrame(chunks).to_parquet("data/arxiv_chunked.parquet")
print(f"Saved {len(chunks)} chunks")
