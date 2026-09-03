import numpy as np
from sentence_transformers import SentenceTransformer
from parser import load_and_chunk_text

class VectorEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Loads a lightweight sentence-transformer model locally.
        """
        print(f"[INFO] Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        """
        Converts a list of text strings into NumPy float32 vector embeddings.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return np.array(embeddings, dtype=np.float32)

if __name__ == "__main__":
    # Test pipeline with parser
    sample_file = "documents/sample.txt"
    chunks = load_and_chunk_text(sample_file)

    embedder = VectorEmbedder()
    vectors = embedder.embed_texts(chunks)

    print("\n=== EMBEDDER TEST ===")
    print(f"Successfully generated embeddings for {len(chunks)} chunks.")
    print(f"Vector Matrix Shape: {vectors.shape} (Chunks x Vector Dimension)")
    print(f"Sample Vector (first 5 dimensions of Chunk 1):\n{vectors[0][:5]}")