import numpy as np
from parser import load_and_chunk_text
from embedder import VectorEmbedder

def cosine_similarity(a, b):
    """
    Computes the cosine similarity between a single vector 'a' 
    and a matrix of vectors 'b'.
    """
    dot_product = np.dot(b, a)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b, axis=1)
    return dot_product / (norm_a * norm_b)

class VectorRetriever:
    def __init__(self, chunks, vectors, embedder):
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder

    def search(self, query, top_k=2):
        """
        Embeds the query and retrieves the top_k most similar chunks.
        """
        # 1. Embed the query
        query_vector = self.embedder.embed_texts([query])[0]

        # 2. Compute similarity scores against all stored vectors
        scores = cosine_similarity(query_vector, self.vectors)

        # 3. Sort indices by highest score
        top_indices = np.argsort(scores)[::-1][:top_k]

        # 4. Return top chunks along with their scores
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx])
            })
        return results

if __name__ == "__main__":
    # Test full pipeline: Parse -> Embed -> Retrieve
    sample_file = "documents/sample.txt"
    chunks = load_and_chunk_text(sample_file)

    embedder = VectorEmbedder()
    vectors = embedder.embed_texts(chunks)

    retriever = VectorRetriever(chunks, vectors, embedder)

    # Test Query
    query = "What is Sachin's TOEIC score?"
    print(f"\n=== RETRIEVER TEST ===")
    print(f"Query: '{query}'\n")

    results = retriever.search(query, top_k=2)
    for rank, res in enumerate(results, 1):
        print(f"Rank {rank} (Score: {res['score']:.4f}):")
        print(f"\"{res['chunk']}\"\n")