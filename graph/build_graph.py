import networkx as nx
import pandas as pd
from pathlib import Path
from collections import defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_parquet(BASE_DIR / "data" / "arxiv_chunked.parquet")

G = nx.Graph()

# phase 1 would be based on shared concepts and similar categories

# creating a node
for _, row in df.iterrows():
    G.add_node(
        row["chunk_id"],
        title=row["metadata"]["title"],
        categories=", ".join(row["metadata"]["categories"]),
        concepts=", ".join(row["metadata"]["concepts"]),
    )

# indexing by concepts and categories

concept_index = defaultdict(list)
category_index = defaultdict(list)

# adding in the index

for _, row in df.iterrows():
    for concept in row["metadata"]["concepts"]:
        concept_index[concept].append(row["chunk_id"])
    for category in row["metadata"]["categories"]:
        category_index[category].append(row["chunk_id"])


# edges from shared concepts
for concept, chunk_ids in concept_index.items():
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            G.add_edge(chunk_ids[i], chunk_ids[j], relation="shared_concept", value=concept)

# edges from shared categories
for category, chunk_ids in category_index.items():
    for i in range(len(chunk_ids)):
        for j in range(i + 1, len(chunk_ids)):
            G.add_edge(chunk_ids[i], chunk_ids[j], relation="same_category", value=category)

print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# save
import json
data = nx.node_link_data(G)
with open(BASE_DIR / "data" / "arxiv_graph.json", "w") as f:
    json.dump(data, f)
print("Graph saved.")
