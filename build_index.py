"""
Build semantic search index from documentation chunks.

This script creates embeddings for all documentation chunks and builds
a FAISS vector index for fast semantic retrieval.
"""

import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


def load_chunks(chunks_path):
    """Load documentation chunks from JSON file."""
    print(f"Loading chunks from: {chunks_path}")
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def create_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    """Create embeddings for all chunks using sentence transformers."""
    print(f"\nLoading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Creating embeddings...")
    # Extract content from chunks
    texts = [chunk['content'] for chunk in chunks]

    # Create embeddings with progress bar
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True
    )

    print(f"Created embeddings with shape: {embeddings.shape}")
    return embeddings, model


def build_faiss_index(embeddings):
    """Build FAISS index for fast similarity search."""
    print("\nBuilding FAISS index...")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity

    # Add embeddings to index
    index.add(embeddings.astype('float32'))

    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def save_index_and_data(index, chunks, embeddings, output_dir='models'):
    """Save FAISS index, chunks, and metadata."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving to {output_dir}/...")

    # Save FAISS index
    index_path = f"{output_dir}/faiss_index.bin"
    faiss.write_index(index, index_path)
    print(f"✓ Saved FAISS index: {index_path}")

    # Save chunks
    chunks_path = f"{output_dir}/chunks.json"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved chunks: {chunks_path}")

    # Save embeddings (optional, for debugging)
    embeddings_path = f"{output_dir}/embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"✓ Saved embeddings: {embeddings_path}")

    # Save metadata
    metadata = {
        'num_chunks': len(chunks),
        'embedding_dimension': embeddings.shape[1],
        'model_name': 'all-MiniLM-L6-v2'
    }
    metadata_path = f"{output_dir}/metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")


def main():
    """Main execution function."""
    # Load chunks
    chunks = load_chunks('documentation_chunks.json')

    # Create embeddings
    embeddings, model = create_embeddings(chunks)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Save everything
    save_index_and_data(index, chunks, embeddings)

    print("\n" + "="*80)
    print("✓ Index building complete!")
    print("="*80)
    print("\nYou can now use semantic_search.py to search the documentation.")


if __name__ == "__main__":
    main()
