import ollama

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    stream = ollama.chat(model="qwen3:4b", stream=True, messages=[
        {"role": "system", "content": "You are a research assistant. Answer based only on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ])
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)
    print()  # newline at end
