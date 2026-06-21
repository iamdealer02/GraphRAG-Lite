from retrieval.retrieval import retrieve 

while True:
    query = input("\nEnter query (or 'quit'): ")
    if query.lower() == "quit":
        break
    results = retrieve(query, top_k=5)
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n[{i+1}] {meta['title']} (dist: {dist:.4f})")
        print(f"     Categories: {meta['categories']}")
        print(f"     Concepts: {meta['concepts']}")
        print(f"     Text: {doc[:200]}...")
