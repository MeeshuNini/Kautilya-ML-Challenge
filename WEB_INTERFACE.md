# 🌐 Web Interface Guide

A minimal Flask web application that combines both ML tasks in a single, user-friendly interface.

---

## 🚀 Quick Start

### 1. Ensure Prerequisites are Complete

Before starting the web interface, make sure you have:

✅ Installed all dependencies:
```bash
pip install -r requirements.txt
```

✅ Built the search index (for Task 1):
```bash
python build_index.py
```

✅ The news dataset is present:
- `Dataset_for_second_task.json` should be in the project root

### 2. Start the Web Server

```bash
python app.py
```

You should see:
```
🚀 Starting Kautilya ML Challenge Web Interface
================================================================================

Open your browser and go to: http://localhost:5000

Press Ctrl+C to stop the server
================================================================================
```

### 3. Open Your Browser

Navigate to: **http://localhost:5000**

---

## 📋 Features

### Task 1: Semantic Search (Left Panel)
- **Search Twitter API Documentation** semantically
- **Adjust number of results** (3, 5, 10, or 15)
- **View ranked results** with relevance scores
- **See metadata** (endpoint name, HTTP method, URL)

### Task 2: Narrative Builder (Right Panel)
- **Build narratives** on any topic
- **Adjust relevance threshold** (0.6 - 0.9)
- **View narrative summary** (AI-generated)
- **Browse timeline** of events chronologically
- **Explore clusters** of related articles
- **See relationship graph** statistics

---

## 🎨 Interface Overview

```
┌─────────────────────────────────────────────────────┐
│        Kautilya ML Challenge - Web Interface        │
└─────────────────────────────────────────────────────┘

┌──────────────────────┐  ┌──────────────────────────┐
│   Task 1: Search     │  │ Task 2: Narrative Build  │
│                      │  │                          │
│  [Search Query    ]  │  │  [Topic            ]     │
│  [Results: 5   ▼]    │  │  [Threshold: 0.8 ▼]      │
│  [Search Button]     │  │  [Build Button]          │
│                      │  │                          │
│  Results appear      │  │  Narrative appears       │
│  below...            │  │  below...                │
└──────────────────────┘  └──────────────────────────┘
```

---

## 💡 Example Usage

### Task 1: Semantic Search

1. **Enter a query** like:
   - "How do I fetch tweets with expansions?"
   - "user authentication"
   - "rate limits"

2. **Select number of results** (default: 5)

3. **Click "Search Documentation"**

4. **View results** ranked by relevance with:
   - Rank position
   - Relevance score
   - Content type (endpoint, parameters, etc.)
   - Preview of content

### Task 2: Narrative Builder

1. **Enter a topic** like:
   - "Hyderabad Metro"
   - "police training"
   - "infrastructure development"

2. **Select relevance threshold**:
   - 0.6 = Broader results
   - 0.7 = Balanced
   - **0.8 = Focused (recommended)**
   - 0.9 = Very focused

3. **Click "Build Narrative"** (takes 20-30 seconds)

4. **View results** including:
   - Narrative summary
   - Timeline of events
   - Article clusters
   - Relationship graph stats

---

## 🔧 API Endpoints

The web interface also exposes REST APIs:

### POST /api/search
Search Twitter API documentation

**Request:**
```json
{
  "query": "How do I fetch tweets?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query_info": {
    "num_results": 5,
    "index_size": 212
  },
  "results": [...]
}
```

### POST /api/narrative
Build narrative from news

**Request:**
```json
{
  "topic": "Hyderabad Metro",
  "threshold": 0.8
}
```

**Response:**
```json
{
  "topic": "Hyderabad Metro",
  "total_relevant_articles": 15,
  "narrative_summary": "...",
  "timeline": [...],
  "clusters": [...],
  "graph": {...}
}
```

### GET /api/health
Check service health

**Response:**
```json
{
  "status": "ok",
  "search_engine": true,
  "narrative_builder": true
}
```

---

## 🛠️ Troubleshooting

### "Search engine not initialized"
**Solution:** Run `python build_index.py` first

### "Narrative builder not initialized"
**Solution:** Ensure `Dataset_for_second_task.json` exists

### Port 5000 already in use
**Solution:** Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

### Loading takes too long
**Expected:**
- First narrative build: 20-30 seconds (models loading + processing)
- Subsequent builds: 10-20 seconds (models cached)
- Searches: <1 second

---

## 🎯 Features

### Visual Design
- ✅ Modern gradient UI
- ✅ Responsive layout (desktop & mobile)
- ✅ Real-time loading indicators
- ✅ Clean result display
- ✅ Color-coded information

### Functionality
- ✅ Dual-task interface
- ✅ Asynchronous processing
- ✅ Error handling
- ✅ Configurable parameters
- ✅ Clickable links to sources

### User Experience
- ✅ No page refresh needed
- ✅ Progress indicators
- ✅ Clear error messages
- ✅ Organized results
- ✅ Scrollable sections

---

## 📱 Mobile Support

The interface is responsive and works on:
- 💻 Desktop browsers
- 📱 Tablets
- 📱 Mobile phones

On smaller screens, tasks stack vertically for better readability.

---

## 🚦 Stopping the Server

Press `Ctrl+C` in the terminal to stop the Flask server.

---

## 📈 Performance

- **Search queries**: <1 second
- **Narrative building**: 10-30 seconds
- **Memory usage**: ~500MB (models loaded)
- **Concurrent users**: Handles multiple simultaneous requests

---

## 🔐 Security Notes

This is a development server for demonstration purposes. For production:

1. Use a production WSGI server (gunicorn, uWSGI)
2. Add authentication if needed
3. Enable HTTPS
4. Add rate limiting
5. Validate all inputs

---

## 📝 Summary

The web interface provides an easy way to:
1. Test both ML tasks without command line
2. Visualize results in a user-friendly format
3. Experiment with different parameters
4. Share results with others

**Enjoy exploring the ML capabilities!** 🎉
