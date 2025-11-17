"""
Semantic Search CLI for Twitter API Documentation

Usage:
    python semantic_search.py --query "How do I fetch tweets with expansions?"
    python semantic_search.py --query "user authentication" --top-k 10
"""

import argparse
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import sys
import os


class SemanticSearchEngine:
    """Semantic search engine using FAISS and sentence transformers."""

    def __init__(self, models_dir='models'):
        """Initialize the search engine with pre-built index."""
        self.models_dir = models_dir
        self.model = None
        self.index = None
        self.chunks = None
        self.metadata = None

        self._load_index()

    def _load_index(self):
        """Load FAISS index, chunks, and metadata."""
        try:
            # Load metadata
            metadata_path = os.path.join(self.models_dir, 'metadata.json')
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

            # Load model
            model_name = self.metadata.get('model_name', 'all-MiniLM-L6-v2')
            self.model = SentenceTransformer(model_name)

            # Load FAISS index
            index_path = os.path.join(self.models_dir, 'faiss_index.bin')
            self.index = faiss.read_index(index_path)

            # Load chunks
            chunks_path = os.path.join(self.models_dir, 'chunks.json')
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)

        except FileNotFoundError as e:
            print(f"Error: Index not found. Please run build_index.py first.", file=sys.stderr)
            print(f"Missing file: {e.filename}", file=sys.stderr)
            sys.exit(1)

    def search(self, query, top_k=5):
        """
        Search for relevant documentation chunks.

        Args:
            query: Search query string
            top_k: Number of top results to return

        Returns:
            List of dictionaries containing search results
        """
        # Create query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)

        # Search FAISS index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            chunk = self.chunks[idx]
            result = {
                'rank': i + 1,
                'score': float(distance),  # Cosine similarity score
                'chunk_id': chunk['id'],
                'type': chunk['metadata']['type'],
                'content': chunk['content'],
                'metadata': chunk['metadata']
            }
            results.append(result)

        return results

    def format_results(self, results, verbose=False):
        """Format search results for display."""
        output = {
            'query_info': {
                'num_results': len(results),
                'index_size': len(self.chunks)
            },
            'results': []
        }

        for result in results:
            formatted_result = {
                'rank': result['rank'],
                'relevance_score': round(result['score'], 4),
                'type': result['type'],
                'metadata': result['metadata']
            }

            if verbose:
                formatted_result['content'] = result['content']
            else:
                # Include truncated content
                content = result['content']
                if len(content) > 300:
                    content = content[:300] + "..."
                formatted_result['content_preview'] = content

            output['results'].append(formatted_result)

        return output


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Semantic search over Twitter API documentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python semantic_search.py --query "How do I fetch tweets with expansions?"
  python semantic_search.py --query "user authentication" --top-k 10
  python semantic_search.py --query "rate limits" --verbose
        """
    )

    parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Search query'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results to return (default: 5)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Include full content in results'
    )

    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory containing the index files (default: models)'
    )

    args = parser.parse_args()

    # Initialize search engine
    try:
        engine = SemanticSearchEngine(models_dir=args.models_dir)
    except Exception as e:
        print(f"Error initializing search engine: {e}", file=sys.stderr)
        sys.exit(1)

    # Perform search
    results = engine.search(args.query, top_k=args.top_k)

    # Format and output results
    output = engine.format_results(results, verbose=args.verbose)

    # Print JSON to stdout
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
