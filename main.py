from retrieval.retrieval import retrieve 
from model.model import generate_answer

while True:
    query = input("\nEnter query (or 'quit'): ")
    if query.lower() == "quit":
        break
    results = retrieve(query, top_k=5)
    retrieved_chunks = results["documents"][0]
    answer = generate_answer(query, retrieved_chunks)
    print(f"\nAnswer: {answer}")
