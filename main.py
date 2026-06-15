from datasets import load_dataset

dataset = load_dataset(
    "CShorten/ML-ArXiv-Papers",
    split="train",
    streaming=True
)
