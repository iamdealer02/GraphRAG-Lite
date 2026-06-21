import json
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
import pandas as pd
import time
from pathlib import Path
import pandas as pd

model = SentenceTransformer("BAAI/bge-small-en-v1.5", local_files_only=True)
kw_model = KeyBERT(model=model)
docs = []
MAX_DOCS = 500

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

SNAPSHOT = DATA_DIR / "arxiv-metadata-oai-snapshot.json"

# Load as a lazy line-by-line iterator (file is ~3.5GB, don't load all at once)
def iter_arxiv(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

print("Loading arXiv snapshot...")
dataset = iter_arxiv(SNAPSHOT)
print("Done loading arXiv snapshot.")

def build_doc(record, idx):
    title      = (record.get("title") or "").replace("\n", " ").strip()
    abstract   = (record.get("abstract") or "").replace("\n", " ").strip()
    categories = (record.get("categories") or "").split()
    authors    = (record.get("authors") or "").replace("\n", " ").strip()
    print(record)
    text = (
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        f"Categories: {record.get('categories', '')}\n"
        f"Authors: {authors}"
    )

    return {
        "doc_id":     record.get("id", f"arxiv_{idx}"),
        "title":      title,
        "text":       text,
        "categories": categories,
        "year":       int((record.get("update_date") or "2000")[:4]),
        "authors":    authors,
        "metadata": {
            "doi":       record.get("doi"),
            "journal":   record.get("journal-ref"),
            "submitter": record.get("submitter"),
        },
    }


def extract_tags(text):
    keywords = kw_model.extract_keywords(
        text,
        top_n=5,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
    )
    return [k[0] for k in keywords]


for i, record in enumerate(dataset):
    if i >= MAX_DOCS:
        break

    print(f"Processing doc {i}...  id={record.get('id')}")

    doc = build_doc(record, i)

    start_time = time.time()
    doc["concepts"] = extract_tags(doc["text"])
    elapsed = time.time() - start_time

    print(f"Doc {doc['doc_id']} | {len(doc['text'])} chars | {elapsed:.2f}s | concepts={doc['concepts']}")
    docs.append(doc)

pd.DataFrame(docs).to_parquet(DATA_DIR / "arxiv_processed.parquet")
print(f"\nSaved {len(docs)} docs → {DATA_DIR / 'arxiv_processed.parquet'}")

df = pd.read_parquet("data/arxiv_processed.parquet")
record = df.iloc[0].to_dict()
for k, v in record.items():
    print(f"{k}: {v}")
