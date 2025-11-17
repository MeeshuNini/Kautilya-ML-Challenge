"""
Extract and chunk documentation from Twitter API Postman Collection.

This script parses the Postman collection JSON and creates searchable chunks
containing API endpoint information, parameters, descriptions, and examples.
"""

import json
import os
from typing import List, Dict, Any


class PostmanDocumentationExtractor:
    """Extract and chunk Postman collection documentation."""

    def __init__(self, collection_path: str):
        """Initialize with path to Postman collection JSON."""
        self.collection_path = collection_path
        self.chunks = []
        self.chunk_id = 0

    def load_collection(self) -> Dict[str, Any]:
        """Load the Postman collection JSON file."""
        with open(self.collection_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_chunk(self, content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a chunk with content and metadata."""
        chunk = {
            'id': self.chunk_id,
            'content': content,
            'metadata': metadata
        }
        self.chunk_id += 1
        return chunk

    def extract_url_info(self, url_obj: Any) -> str:
        """Extract URL from Postman URL object."""
        if isinstance(url_obj, str):
            return url_obj
        elif isinstance(url_obj, dict):
            return url_obj.get('raw', '')
        return ''

    def extract_query_params(self, url_obj: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract query parameters from URL object."""
        params = []
        if isinstance(url_obj, dict) and 'query' in url_obj:
            for param in url_obj.get('query', []):
                if isinstance(param, dict):
                    params.append({
                        'key': param.get('key', ''),
                        'description': param.get('description', ''),
                        'required': not param.get('disabled', True)
                    })
        return params

    def extract_path_variables(self, url_obj: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract path variables from URL object."""
        variables = []
        if isinstance(url_obj, dict) and 'variable' in url_obj:
            for var in url_obj.get('variable', []):
                if isinstance(var, dict):
                    variables.append({
                        'key': var.get('key', ''),
                        'description': var.get('description', ''),
                        'type': var.get('type', 'string')
                    })
        return variables

    def process_request(self, item: Dict[str, Any], category_path: List[str]) -> None:
        """Process a single API request and create chunks."""
        request = item.get('request', {})
        if not request:
            return

        name = item.get('name', 'Unnamed Endpoint')
        method = request.get('method', 'GET')
        url_obj = request.get('url', {})
        url = self.extract_url_info(url_obj)
        description = request.get('description', '')

        category = ' > '.join(category_path) if category_path else 'General'

        # Chunk 1: Main endpoint information
        main_content = f"""Endpoint: {name}
Category: {category}
Method: {method}
URL: {url}

Description:
{description if description else 'No description available.'}
"""

        self.chunks.append(self.create_chunk(
            main_content,
            {
                'type': 'endpoint',
                'name': name,
                'category': category,
                'method': method,
                'url': url
            }
        ))

        # Chunk 2: Query parameters (if any)
        query_params = self.extract_query_params(url_obj)
        if query_params:
            params_content = f"""Endpoint: {name} - Query Parameters
URL: {url}

Available Query Parameters:
"""
            for param in query_params:
                required_text = "(Required)" if param['required'] else "(Optional)"
                params_content += f"\n- {param['key']} {required_text}\n  {param['description']}\n"

            self.chunks.append(self.create_chunk(
                params_content,
                {
                    'type': 'parameters',
                    'endpoint_name': name,
                    'category': category,
                    'url': url,
                    'param_count': len(query_params)
                }
            ))

        # Chunk 3: Path variables (if any)
        path_vars = self.extract_path_variables(url_obj)
        if path_vars:
            vars_content = f"""Endpoint: {name} - Path Variables
URL: {url}

Path Variables:
"""
            for var in path_vars:
                vars_content += f"\n- {var['key']} ({var['type']})\n  {var['description']}\n"

            self.chunks.append(self.create_chunk(
                vars_content,
                {
                    'type': 'path_variables',
                    'endpoint_name': name,
                    'category': category,
                    'url': url
                }
            ))

        # Chunk 4: Response examples (if any)
        responses = item.get('response', [])
        for idx, response in enumerate(responses):
            response_name = response.get('name', f'Example {idx + 1}')
            response_body = response.get('body', '')

            # Only create chunk if there's meaningful response content
            if response_body and len(response_body.strip()) > 0:
                # Truncate very long responses
                if len(response_body) > 2000:
                    response_body = response_body[:2000] + "\n... (truncated)"

                response_content = f"""Endpoint: {name} - Response Example
Response Name: {response_name}
URL: {url}

Example Response:
{response_body}
"""

                self.chunks.append(self.create_chunk(
                    response_content,
                    {
                        'type': 'response_example',
                        'endpoint_name': name,
                        'category': category,
                        'response_name': response_name,
                        'url': url
                    }
                ))

    def process_items(self, items: List[Dict[str, Any]], category_path: List[str] = None) -> None:
        """Recursively process items in the collection."""
        if category_path is None:
            category_path = []

        for item in items:
            # If item has nested items, it's a folder/category
            if 'item' in item and isinstance(item['item'], list):
                folder_name = item.get('name', 'Unnamed Category')
                new_path = category_path + [folder_name]

                # Create a chunk for the folder description if available
                folder_desc = item.get('description', '')
                if folder_desc:
                    folder_content = f"""Category: {folder_name}
Path: {' > '.join(new_path)}

Description:
{folder_desc}
"""
                    self.chunks.append(self.create_chunk(
                        folder_content,
                        {
                            'type': 'category',
                            'name': folder_name,
                            'path': ' > '.join(new_path)
                        }
                    ))

                # Process nested items
                self.process_items(item['item'], new_path)

            # If item has request, it's an actual endpoint
            elif 'request' in item:
                self.process_request(item, category_path)

    def extract_all(self) -> List[Dict[str, Any]]:
        """Extract all chunks from the collection."""
        print(f"Loading collection from: {self.collection_path}")
        collection = self.load_collection()

        # Add collection-level information
        info = collection.get('info', {})
        if info:
            collection_content = f"""Twitter API v2 - Collection Overview
Name: {info.get('name', 'Twitter API v2')}

Description:
{info.get('description', '')}
"""
            self.chunks.append(self.create_chunk(
                collection_content,
                {
                    'type': 'collection_overview',
                    'name': info.get('name', '')
                }
            ))

        # Process all items
        items = collection.get('item', [])
        print(f"Processing {len(items)} top-level items...")
        self.process_items(items)

        print(f"Extracted {len(self.chunks)} chunks")
        return self.chunks

    def save_chunks(self, output_path: str) -> None:
        """Save chunks to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved chunks to: {output_path}")


def main():
    """Main execution function."""
    # Path to the Postman collection
    collection_path = r"c:\Users\sahit\OneDrive\Desktop\kautilya_ml_challenge\semantic_search\postman-twitter-api\Twitter API v2.postman_collection.json"

    # Output path for chunks
    output_path = r"c:\Users\sahit\OneDrive\Desktop\kautilya_ml_challenge\documentation_chunks.json"

    # Extract and chunk
    extractor = PostmanDocumentationExtractor(collection_path)
    chunks = extractor.extract_all()

    # Save chunks
    extractor.save_chunks(output_path)

    # Print sample chunks
    print("\n" + "="*80)
    print("SAMPLE CHUNKS")
    print("="*80)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk #{chunk['id']}:")
        print(f"Type: {chunk['metadata']['type']}")
        print(f"Content preview:\n{chunk['content'][:300]}...")
        print("-"*80)


if __name__ == "__main__":
    main()
