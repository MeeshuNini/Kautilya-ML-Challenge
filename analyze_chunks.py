import json

with open('documentation_chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

print(f"Total chunks extracted: {len(chunks)}\n")

# Count by type
types = {}
for chunk in chunks:
    chunk_type = chunk['metadata']['type']
    types[chunk_type] = types.get(chunk_type, 0) + 1

print("Chunk Distribution by Type:")
for k, v in sorted(types.items()):
    print(f"  {k}: {v}")

print("\n" + "="*80)
print("Sample chunks from different types:")
print("="*80)

shown_types = set()
for chunk in chunks:
    chunk_type = chunk['metadata']['type']
    if chunk_type not in shown_types:
        print(f"\n[{chunk_type.upper()}] - Chunk #{chunk['id']}")
        print(chunk['content'][:400] + "...")
        print("-"*80)
        shown_types.add(chunk_type)
        if len(shown_types) >= 4:
            break
