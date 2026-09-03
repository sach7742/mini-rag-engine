import os

def load_and_chunk_text(file_path, chunk_size=150, overlap=30):
    """
    Reads a text file and splits it into overlapping character chunks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":
    sample_file = "documents/sample.txt"
    chunks = load_and_chunk_text(sample_file)
    
    print(f"=== PARSER TEST ===")
    print(f"Extracted {len(chunks)} chunks from {sample_file}:\n")
    for idx, c in enumerate(chunks, 1):
        print(f"--- Chunk {idx} ---")
        print(c)
        print()