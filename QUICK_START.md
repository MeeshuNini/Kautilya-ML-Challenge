# 🚀 Quick Start Guide

## Step-by-Step Setup

### 1. Install Dependencies (5-10 minutes)
```bash
cd "c:\Users\sahit\OneDrive\Desktop\kautilya_ml_challenge"
pip install -r requirements.txt
```

**First time setup downloads ~500MB of models**

---

## Task 1: Semantic Search

### 2. Build the Search Index (30 seconds)
```bash
python build_index.py
```

This will:
- ✅ Load 212 documentation chunks
- ✅ Create embeddings
- ✅ Build FAISS index
- ✅ Save to `models/` folder

### 3. Test Semantic Search
```bash
python semantic_search.py --query "How do I fetch tweets with expansions?"
```

**More examples:**
```bash
python semantic_search.py --query "user authentication"
python semantic_search.py --query "rate limits" --top-k 10
python semantic_search.py --query "media fields" --verbose
```

---

## Task 2: Narrative Builder

### 4. Test Narrative Builder
```bash
python narrative_builder.py --topic "Hyderabad Metro"
```

**More examples:**
```bash
python narrative_builder.py --topic "police training"
python narrative_builder.py --topic "land revenue"
python narrative_builder.py --topic "infrastructure development"
```

---

## Expected Output

### Semantic Search Output:
```json
{
  "query_info": {
    "num_results": 5,
    "index_size": 212
  },
  "results": [
    {
      "rank": 1,
      "relevance_score": 0.8523,
      "type": "parameters",
      "metadata": {...},
      "content_preview": "..."
    }
  ]
}
```

### Narrative Builder Output:
```json
{
  "topic": "Hyderabad Metro",
  "total_relevant_articles": 15,
  "narrative_summary": "Summary of the story...",
  "timeline": [...],
  "clusters": [...],
  "graph": {...}
}
```

---

## Troubleshooting

### Issue: "No module named 'sentence_transformers'"
**Solution:** Run `pip install -r requirements.txt`

### Issue: "Index not found"
**Solution:** Run `python build_index.py` first

### Issue: "Dataset not found"
**Solution:** Ensure `Dataset_for_second_task.json` is in the current directory

### Issue: Models downloading slowly
**Solution:** First run downloads ~500MB, be patient. Subsequent runs are fast.

---

## File Checklist

Before testing, ensure you have:
- ✅ `requirements.txt`
- ✅ `extract_and_chunk.py`
- ✅ `build_index.py`
- ✅ `semantic_search.py`
- ✅ `narrative_builder.py`
- ✅ `Dataset_for_second_task.json` (81MB)
- ✅ `semantic_search/postman-twitter-api/Twitter API v2.postman_collection.json`

After running `build_index.py`, you should have:
- ✅ `documentation_chunks.json`
- ✅ `models/faiss_index.bin`
- ✅ `models/chunks.json`
- ✅ `models/embeddings.npy`
- ✅ `models/metadata.json`

---

## Performance Expectations

| Task | First Run | Subsequent Runs |
|------|-----------|-----------------|
| **Semantic Search Setup** | ~30 seconds | N/A (one-time) |
| **Semantic Search Query** | <1 second | <1 second |
| **Narrative Builder** | ~20-30 seconds | ~20-30 seconds |

---

## Need Help?

See full documentation in [README.md](README.md)
