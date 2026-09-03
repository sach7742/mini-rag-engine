import os
from parser import load_and_chunk_text
from embedder import VectorEmbedder
from retriever import VectorRetriever
from openai import OpenAI

def generate_rag_response(query, retrieved_chunks, client):
    """
    Combines retrieved chunks into context and queries the local Ollama LLM.
    """
    context = "\n---\n".join([r['chunk'] for r in retrieved_chunks])
    
    prompt = f"""You are a concise AI assistant. Answer the user's question based ONLY on the provided context below.
If the answer cannot be found in the context, say "I don't have enough information in the context."

Context:
{context}

Question: {query}
Answer:"""

    try:
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR]: Could not connect to local Ollama instance.\nDetails: {e}"

def main():
    sample_file = "documents/sample.txt"
    print("==========================================")
    print("  MINI-RAG-ENGINE (LOCAL LLM VIA OLLAMA)  ")
    print("==========================================\n")
    
    # 1. Indexing Phase
    print("[1/3] Loading & Chunking Document...")
    chunks = load_and_chunk_text(sample_file)
    
    print("[2/3] Generating Vector Embeddings...")
    embedder = VectorEmbedder()
    vectors = embedder.embed_texts(chunks)
    
    retriever = VectorRetriever(chunks, vectors, embedder)
    print("[3/3] System Ready!\n")

    # Connect to local Ollama server running on port 11434
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    # 2. Query Loop
    while True:
        query = input("\nAsk a question about the document (or type 'exit' to quit): ").strip()
        if not query or query.lower() == 'exit':
            print("Exiting pipeline. Goodbye!")
            break

        print("\n[STEP 1] Retrieving relevant context...")
        results = retriever.search(query, top_k=2)

        print("\nTop Retrieved Document Chunks:")
        for idx, res in enumerate(results, 1):
            print(f"  [{idx}] (Score: {res['score']:.4f}) \"{res['chunk']}\"")

        print("\n[STEP 2] Llama 3.2 is reading context and generating final answer...")
        answer = generate_rag_response(query, results, client)

        print("\n=== FINAL AI ANSWER ===")
        print(answer)
        print("=" * 45)

if __name__ == "__main__":
    main()