"""
Flask Web Application for Kautilya ML Challenge

Combines both tasks in a single web interface:
1. Semantic Search on Twitter API Documentation
2. Narrative Building from News Dataset
"""

from flask import Flask, render_template, request, jsonify
import json
import os
import sys

# Import the search engine and narrative builder
from semantic_search import SemanticSearchEngine
from narrative_builder_improved import FastNarrativeBuilder

app = Flask(__name__)

# Initialize models (do this once at startup)
print("Initializing models...", file=sys.stderr)
search_engine = None
narrative_builder = None

try:
    search_engine = SemanticSearchEngine(models_dir='models')
    print("✓ Semantic search engine loaded", file=sys.stderr)
except Exception as e:
    print(f"⚠ Warning: Could not load search engine: {e}", file=sys.stderr)

try:
    narrative_builder = FastNarrativeBuilder(dataset_path='Dataset_for_second_task.json')
    print("✓ Narrative builder loaded", file=sys.stderr)
except Exception as e:
    print(f"⚠ Warning: Could not load narrative builder: {e}", file=sys.stderr)


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/search', methods=['POST'])
def semantic_search():
    """API endpoint for semantic search."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        top_k = data.get('top_k', 5)

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        if search_engine is None:
            return jsonify({
                'error': 'Search engine not initialized. Please run build_index.py first.'
            }), 500

        # Perform search
        results = search_engine.search(query, top_k=top_k)
        output = search_engine.format_results(results, verbose=False)

        return jsonify(output)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/narrative', methods=['POST'])
def build_narrative():
    """API endpoint for narrative building."""
    try:
        data = request.get_json()
        topic = data.get('topic', '')
        threshold = data.get('threshold', 0.8)

        if not topic:
            return jsonify({'error': 'Topic is required'}), 400

        if narrative_builder is None:
            return jsonify({
                'error': 'Narrative builder not initialized. Check if dataset exists.'
            }), 500

        # Build narrative
        result = narrative_builder.build_narrative(topic, relevance_threshold=threshold)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'search_engine': search_engine is not None,
        'narrative_builder': narrative_builder is not None
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("🚀 Starting Kautilya ML Challenge Web Interface")
    print("="*80)
    print("\nOpen your browser and go to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
