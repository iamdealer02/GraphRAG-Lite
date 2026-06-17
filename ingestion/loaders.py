import kagglehub
import shutil
from pathlib import Path

# ingestion/download.py  →  BASE_DIR = GraphRAG Lite/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"   # GraphRAG Lite/data/
DATA_DIR.mkdir(exist_ok=True)

path = kagglehub.dataset_download("Cornell-University/arxiv")
print("Downloaded to:", path)

src = Path(path) / "arxiv-metadata-oai-snapshot.json"
dst = DATA_DIR / "arxiv-metadata-oai-snapshot.json"

if not dst.exists():
    print("Copying to data folder...")
    shutil.copy2(src, dst)
    print("Saved to:", dst)
else:
    print("Already exists at:", dst)
