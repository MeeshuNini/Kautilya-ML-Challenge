# #!/usr/bin/env python3
# """
# Improved Narrative Builder with embedding caching + optional FAISS index.

# Usage examples:
#     python narrative_builder_cached.py --topic "Hyderabad Metro"
#     python narrative_builder_cached.py --topic "police reforms" --model all-mpnet-base-v2 --use-faiss
#     python narrative_builder_cached.py --topic "infrastructure" --recompute --use-faiss
# """

# import argparse
# import json
# import os
# import sys
# import hashlib
# from datetime import datetime
# from collections import defaultdict
# import re
# import warnings

# import numpy as np

# from sentence_transformers import SentenceTransformer, util
# from sklearn.cluster import KMeans
# import networkx as nx

# # summarization optional
# try:
#     from transformers import pipeline
# except Exception:
#     pipeline = None

# # optional FAISS
# try:
#     import faiss
#     _HAS_FAISS = True
# except Exception:
#     faiss = None
#     _HAS_FAISS = False

# warnings.filterwarnings("ignore")


# def eprint(*args, **kwargs):
#     print(*args, file=sys.stderr, **kwargs)


# def parse_date_flexible(published):
#     """Robust ISO-like and common date parsing, returns datetime or None."""
#     if not published or not isinstance(published, str):
#         return None
#     s = published.strip()
#     try:
#         s2 = s.replace("Z", "+00:00") if s.endswith("Z") else s
#         return datetime.fromisoformat(s2)
#     except Exception:
#         pass
#     fmts = ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
#     for fmt in fmts:
#         try:
#             return datetime.strptime(s, fmt)
#         except Exception:
#             continue
#     return None


# class EmbeddingCache:
#     """
#     Manage embedding caching to disk. Saves npz with fields:
#       - embeddings (float32 ndarray)
#       - ids (list)  # optional mapping to original indices or ids
#       - dataset_hash (str)
#     Also optionally manages a FAISS index file alongside the npz.
#     """

#     def __init__(self, cache_dir="emb_cache"):
#         os.makedirs(cache_dir, exist_ok=True)
#         self.cache_dir = cache_dir

#     @staticmethod
#     def _hash_bytes(b: bytes) -> str:
#         return hashlib.md5(b).hexdigest()

#     def dataset_hash(self, dataset_path):
#         with open(dataset_path, "rb") as f:
#             data = f.read()
#         return self._hash_bytes(data)

#     def _cache_paths(self, model_name, namespace):
#         safe_model = model_name.replace("/", "_")
#         base = os.path.join(self.cache_dir, f"{safe_model}__{namespace}")
#         return base + ".npz", base + ".faiss"

#     def exists_and_valid(self, model_name, namespace, dataset_hash):
#         npz_path, faiss_path = self._cache_paths(model_name, namespace)
#         if not os.path.exists(npz_path):
#             return False
#         try:
#             meta = np.load(npz_path, allow_pickle=True)
#             cached_hash = str(meta.get("dataset_hash", ""))
#             return cached_hash == dataset_hash
#         except Exception:
#             return False

#     def load(self, model_name, namespace):
#         npz_path, faiss_path = self._cache_paths(model_name, namespace)
#         meta = np.load(npz_path, allow_pickle=True)
#         embeddings = meta["embeddings"]
#         ids = meta["ids"].tolist() if "ids" in meta else None
#         return embeddings, ids

#     def save(self, model_name, namespace, embeddings: np.ndarray, ids=None, dataset_hash=None, faiss_index=None):
#         npz_path, faiss_path = self._cache_paths(model_name, namespace)
#         # ensure float32 for faiss friendliness
#         emb_to_save = embeddings.astype(np.float32)
#         ids_to_save = np.array(ids if ids is not None else list(range(len(emb_to_save))), dtype=object)
#         np.savez_compressed(npz_path, embeddings=emb_to_save, ids=ids_to_save, dataset_hash=str(dataset_hash))
#         if faiss_index is not None and _HAS_FAISS:
#             faiss.write_index(faiss_index, faiss_path)


# class SemanticSearcher:
#     """
#     Provide search over embeddings either with FAISS (fast) or brute-force util.cos_sim.
#     """

#     def __init__(self, model, cache: EmbeddingCache, model_name: str, use_faiss: bool = False):
#         self.model = model
#         self.cache = cache
#         self.model_name = model_name
#         self.use_faiss = use_faiss and _HAS_FAISS
#         if use_faiss and not _HAS_FAISS:
#             eprint("FAISS requested but not installed. Falling back to brute-force search.")

#     def build_or_load_index(self, namespace, texts, ids, dataset_hash, rebuild=False):
#         """
#         Ensure cached embeddings exist and optionally build/load a FAISS index.
#         Returns (embeddings ndarray, ids list, faiss_index or None)
#         """
#         # Check cache
#         if not rebuild and self.cache.exists_and_valid(self.model_name, namespace, dataset_hash):
#             eprint(f"[cache] Loading embeddings for '{namespace}' from cache.")
#             embeddings, ids_loaded = self.cache.load(self.model_name, namespace)
#             return embeddings.astype(np.float32), ids_loaded, None  # caller may load faiss if needed

#         # Else compute embeddings
#         eprint(f"[compute] Computing embeddings for namespace='{namespace}' ({len(texts)} items)...")
#         # ensure numpy output and float32
#         embeddings = np.asarray(self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True), dtype=np.float32)
#         # Save embeddings to cache (FAISS index may be built separately)
#         self.cache.save(self.model_name, namespace, embeddings, ids=ids, dataset_hash=dataset_hash)
#         return embeddings, ids, None

#     def build_faiss_index(self, embeddings, namespace, metric=faiss.METRIC_INNER_PRODUCT):
#         """
#         Build an inner-product FAISS index (with normalization) or L2 depending on use.
#         This function returns the built index.
#         """
#         if not _HAS_FAISS:
#             raise RuntimeError("FAISS is not available in this environment.")
#         d = embeddings.shape[1]
#         # Normalize embeddings for cosine similarity using inner product
#         faiss.normalize_L2(embeddings)
#         # IndexFlatIP is simplest exact index
#         index = faiss.IndexFlatIP(d)
#         index.add(embeddings)
#         return index

#     def search(self, query, embeddings, top_k=10, faiss_index=None):
#         """
#         Query: str. If faiss_index provided and use_faiss True, use it. Else use brute-force.
#         Returns list of (idx, score) sorted desc.
#         """
#         q_emb = np.asarray(self.model.encode([query], convert_to_numpy=True), dtype=np.float32)
#         # if using faiss, normalize and query
#         if self.use_faiss and faiss_index is not None:
#             faiss.normalize_L2(q_emb)
#             D, I = faiss_index.search(q_emb, top_k)
#             # D are inner product scores (because normalized)
#             results = [(int(I[0][i]), float(D[0][i])) for i in range(len(I[0])) if I[0][i] != -1]
#             return results

#         # brute force using sentence-transformers util
#         sims = util.cos_sim(q_emb, embeddings)[0].cpu().numpy()
#         ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
#         return [(int(idx), float(score)) for idx, score in ranked]


# class ImprovedNarrativeBuilder:
#     """
#     Main class: modular, with embedding caching and optional FAISS.
#     Methods are small and composable.
#     """

#     def __init__(self, dataset_path="Dataset_for_second_task.json", model_name="all-MiniLM-L6-v2",
#                  cache_dir="emb_cache", use_faiss=False, summarizer_model="facebook/bart-large-cnn"):
#         self.dataset_path = dataset_path
#         self.model_name = model_name
#         self.use_faiss = use_faiss and _HAS_FAISS
#         self.cache = EmbeddingCache(cache_dir=cache_dir)

#         eprint(f"Loading embedding model: {self.model_name} ...")
#         self.embedding_model = SentenceTransformer(self.model_name)

#         # summarizer optional
#         self.summarizer = None
#         if pipeline is not None:
#             try:
#                 self.summarizer = pipeline("summarization", model=summarizer_model, device=-1)
#             except Exception as ex:
#                 eprint(f"Warning: summarizer unavailable: {ex}")

#         # placeholders
#         self.articles = []
#         self.filtered_articles = []
#         self.relevant_articles = []

#         # helper searcher
#         self.searcher = SemanticSearcher(self.embedding_model, self.cache, self.model_name, use_faiss=self.use_faiss)

#     # ---------------- dataset/load helpers ----------------
#     def load_dataset(self):
#         eprint(f"Loading dataset from '{self.dataset_path}' ...")
#         if not os.path.exists(self.dataset_path):
#             raise FileNotFoundError(self.dataset_path)
#         with open(self.dataset_path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         self.articles = data.get("items", []) if isinstance(data, dict) else data
#         eprint(f"Loaded {len(self.articles)} articles.")

#     # ---------------- filtering ----------------
#     def filter_by_rating(self, min_rating=8):
#         self.filtered_articles = [
#             article for article in self.articles
#             if float(article.get("source_rating", 0) or 0) >= float(min_rating)
#         ]
#         eprint(f"Filtered to {len(self.filtered_articles)} articles with rating >= {min_rating}")

#     # ---------------- query expansion & keyword scoring ----------------
#     def expand_query(self, topic):
#         expansions = {
#             "metro": ["metro rail", "subway", "rapid transit", "underground"],
#             "police": ["law enforcement", "officers", "department", "cops"],
#             "infrastructure": ["development", "construction", "projects", "facilities"],
#             "election": ["voting", "polls", "campaign", "electoral"],
#         }
#         expanded = [topic]
#         tl = topic.lower()
#         for k, syns in expansions.items():
#             if k in tl:
#                 expanded.extend(syns)
#         return " ".join(expanded)

#     def keyword_score(self, topic, text):
#         topic_words = set(re.findall(r"\b\w+\b", topic.lower()))
#         text_words = set(re.findall(r"\b\w+\b", (text or "").lower()))
#         stopwords = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "but", "is", "are"}
#         topic_words -= stopwords
#         text_words -= stopwords
#         if not topic_words:
#             return 0.0
#         overlap = len(topic_words & text_words)
#         return overlap / len(topic_words)

#     # ---------------- embedding helpers (with caching) ----------------
#     def _namespace_texts_for(self, namespace):
#         """
#         Prepare texts and ids for a namespace.
#         namespaces supported:
#           - combined: title + story
#           - title: title only
#           - content: story only
#         """
#         if namespace == "combined":
#             texts = [f"{a.get('title','')} {a.get('story','')}" for a in self.filtered_articles]
#         elif namespace == "title":
#             texts = [a.get("title", "") for a in self.filtered_articles]
#         elif namespace == "content":
#             texts = [a.get("story", "") for a in self.filtered_articles]
#         else:
#             raise ValueError("Unsupported namespace")
#         ids = [a.get("id", idx) for idx, a in enumerate(self.filtered_articles)]
#         return texts, ids

#     def get_embeddings(self, namespace="combined", recompute=False):
#         """
#         Return embeddings ndarray and ids for namespace. Uses cache if available.
#         """
#         texts, ids = self._namespace_texts_for(namespace)
#         dataset_hash = self.cache.dataset_hash(self.dataset_path)
#         if not recompute and self.cache.exists_and_valid(self.model_name, namespace, dataset_hash):
#             emb, cached_ids = self.cache.load(self.model_name, namespace)
#             eprint(f"[cache] loaded embeddings for '{namespace}' ({emb.shape[0]} x {emb.shape[1]})")
#             return np.asarray(emb, dtype=np.float32), cached_ids
#         # compute
#         emb, ids_out, _ = self.searcher.build_or_load_index(namespace, texts, ids, dataset_hash, rebuild=recompute)
#         eprint(f"[compute] embeddings computed for '{namespace}' ({emb.shape[0]} x {emb.shape[1]})")
#         return np.asarray(emb, dtype=np.float32), ids

#     # ---------------- search / find relevant articles ----------------
#     def find_relevant_articles_simple(self, topic, threshold=0.8, top_k=None, recompute_embeddings=False):
#         eprint(f"Simple semantic search for '{topic}' (threshold={threshold})")
#         # Use combined namespace
#         embeddings, ids = self.get_embeddings(namespace="combined", recompute=recompute_embeddings)

#         # If using FAISS, build index and use it
#         faiss_index = None
#         if self.use_faiss:
#             eprint("Building FAISS index (normalized inner product)...")
#             faiss_index = self.searcher.build_faiss_index(embeddings, "combined")
#             # Save faiss alongside npz (optional)
#             self.cache.save(self.model_name, "combined", embeddings, ids=ids, dataset_hash=self.cache.dataset_hash(self.dataset_path), faiss_index=faiss_index)

#         # Search
#         results = self.searcher.search(topic, embeddings, top_k=(top_k or 50), faiss_index=faiss_index)
#         # Filter by threshold
#         selected = [r for r in results if r[1] >= threshold]
#         # Map back to articles and add score
#         self.relevant_articles = []
#         for idx, score in selected:
#             art_idx = int(ids[idx]) if ids is not None else idx
#             article = self.filtered_articles[art_idx]
#             article_copy = dict(article)
#             article_copy["relevance_score"] = float(score)
#             self.relevant_articles.append(article_copy)
#         self.relevant_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
#         eprint(f"Found {len(self.relevant_articles)} relevant articles (simple)")

#     def find_relevant_articles_weighted(self, topic, threshold=0.8, title_weight=0.7, recompute_embeddings=False):
#         eprint(f"Weighted search for '{topic}' (title_weight={title_weight}, threshold={threshold})")
#         title_emb, _ = self.get_embeddings(namespace="title", recompute=recompute_embeddings)
#         content_emb, _ = self.get_embeddings(namespace="content", recompute=recompute_embeddings)

#         topic_emb = np.asarray(self.embedding_model.encode([topic], convert_to_numpy=True), dtype=np.float32)
#         title_sims = util.cos_sim(topic_emb, title_emb)[0].cpu().numpy()
#         content_sims = util.cos_sim(topic_emb, content_emb)[0].cpu().numpy()
#         combined = (float(title_weight) * title_sims) + ((1.0 - float(title_weight)) * content_sims)

#         indices = [i for i, sim in enumerate(combined) if float(sim) >= float(threshold)]
#         self.relevant_articles = []
#         for i in indices:
#             article = dict(self.filtered_articles[i])
#             article["relevance_score"] = float(combined[i])
#             article["title_score"] = float(title_sims[i])
#             article["content_score"] = float(content_sims[i])
#             self.relevant_articles.append(article)
#         self.relevant_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
#         eprint(f"Found {len(self.relevant_articles)} relevant articles (weighted)")

#     def find_relevant_articles_hybrid(self, topic, threshold=0.7, semantic_weight=0.7, recompute_embeddings=False):
#         eprint(f"Hybrid search for '{topic}' (semantic_weight={semantic_weight}, threshold={threshold})")
#         expanded = self.expand_query(topic)
#         combined_texts, _ = self._namespace_texts_for("combined")
#         embeddings, _ = self.get_embeddings(namespace="combined", recompute=recompute_embeddings)

#         sem_emb = np.asarray(self.embedding_model.encode([expanded], convert_to_numpy=True), dtype=np.float32)
#         sem_sims = util.cos_sim(sem_emb, embeddings)[0].cpu().numpy()
#         kw_sims = [self.keyword_score(topic, t) for t in combined_texts]
#         semantic_weight = float(semantic_weight)
#         combined_scores = [(semantic_weight * float(s)) + ((1.0 - semantic_weight) * float(k)) for s, k in zip(sem_sims, kw_sims)]
#         indices = [i for i, sc in enumerate(combined_scores) if sc >= threshold]
#         self.relevant_articles = []
#         for i in indices:
#             article = dict(self.filtered_articles[i])
#             article["relevance_score"] = float(combined_scores[i])
#             article["semantic_score"] = float(sem_sims[i])
#             article["keyword_score"] = float(kw_sims[i])
#             self.relevant_articles.append(article)
#         self.relevant_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
#         eprint(f"Found {len(self.relevant_articles)} relevant articles (hybrid)")

#     def find_relevant_articles(self, topic, threshold=0.8, scoring_strategy="simple", **kwargs):
#         recompute = kwargs.pop("recompute_embeddings", False)
#         if scoring_strategy == "weighted":
#             self.find_relevant_articles_weighted(topic, threshold=threshold, recompute_embeddings=recompute, **kwargs)
#         elif scoring_strategy == "hybrid":
#             self.find_relevant_articles_hybrid(topic, threshold=threshold, recompute_embeddings=recompute, **kwargs)
#         else:
#             self.find_relevant_articles_simple(topic, threshold=threshold, recompute_embeddings=recompute, **kwargs)

#     # ---------------- narrative building ----------------
#     def generate_narrative_summary(self, topic, max_articles=10):
#         if not self.relevant_articles:
#             return f"No articles found related to '{topic}'."
#         top = self.relevant_articles[:max_articles]
#         combined_text = f"Topic: {topic}\n\n"
#         for art in top[:5]:
#             story = art.get("story", "") or ""
#             # take first 2-3 sentences
#             sentences = re.split(r"\. |\n", story)[:3]
#             summary_text = ". ".join([s.strip() for s in sentences if s.strip()]) + ('.' if sentences else '')
#             combined_text += summary_text + "\n\n"

#         if self.summarizer and len(combined_text) > 100:
#             try:
#                 txt = combined_text[:1024]
#                 res = self.summarizer(txt, max_length=200, min_length=50, do_sample=False)
#                 return res[0]["summary_text"]
#             except Exception as ex:
#                 eprint(f"Summarization failed: {ex}, falling back to extractive")

#         # fallback extractive
#         sentences = []
#         for art in top[:5]:
#             story = art.get("story", "") or ""
#             first = (story.split(". ")[0].strip() if story else art.get("title", ""))
#             if first:
#                 sentences.append(first)
#         return " ".join(sentences[:8]) + "."

#     def create_timeline(self):
#         timeline = []
#         for article in self.relevant_articles:
#             published = article.get("published_at", "") or article.get("published", "") or article.get("date", "")
#             date_obj = parse_date_flexible(published)
#             date_str = date_obj.strftime("%Y-%m-%d") if date_obj else "Unknown"
#             story = article.get("story", "") or ""
#             first_sentence = story.split(". ")[0] if story else article.get("title", "")
#             timeline.append({
#                 "date_obj": date_obj,
#                 "date": date_str,
#                 "headline": article.get("title", ""),
#                 "url": article.get("url", ""),
#                 "why_it_matters": first_sentence,
#                 "relevance_score": round(float(article.get("relevance_score", 0)), 3)
#             })
#         timeline.sort(key=lambda x: (x["date_obj"] is None, x["date_obj"] or datetime.max))
#         for item in timeline:
#             item.pop("date_obj", None)
#         return timeline

#     def create_narrative_clusters(self, n_clusters=5):
#         if len(self.relevant_articles) < 2:
#             return [{
#                 "theme": "Main Topic",
#                 "articles": [{"title": a.get("title", ""), "url": a.get("url", ""), "relevance": round(float(a.get("relevance_score", 0)), 3)}
#                              for a in self.relevant_articles]
#             }]
#         texts = [f"{a.get('title','')} {a.get('story','')}" for a in self.relevant_articles]
#         emb = np.asarray(self.embedding_model.encode(texts, convert_to_numpy=True), dtype=np.float32)
#         if len(self.relevant_articles) < n_clusters:
#             n_clusters = max(2, len(self.relevant_articles) // 2)
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
#         labels = kmeans.fit_predict(emb)
#         clusters = []
#         clusters_dict = defaultdict(list)
#         for idx, label in enumerate(labels):
#             clusters_dict[label].append(self.relevant_articles[idx])
#         for cluster_id, arts in clusters_dict.items():
#             all_titles = " ".join([a.get("title", "") for a in arts])
#             words = re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", all_titles)
#             theme = f"Cluster {cluster_id+1}"
#             if words:
#                 from collections import Counter
#                 common = Counter(words).most_common(2)
#                 theme = " ".join([w for w, _ in common])
#             clusters.append({
#                 "theme": theme,
#                 "article_count": len(arts),
#                 "articles": [
#                     {"title": a.get("title", ""), "url": a.get("url", ""), "date": a.get("published_at", "Unknown"),
#                      "relevance": round(float(a.get("relevance_score", 0)), 3)}
#                     for a in sorted(arts, key=lambda x: float(x.get("relevance_score", 0)), reverse=True)[:10]
#                 ]
#             })
#         return clusters

#     def _detect_relation_type(self, article1, article2, similarity):
#         try:
#             similarity = float(similarity)
#         except Exception:
#             similarity = 0.0
#         if similarity > 0.8:
#             return "builds_on"
#         elif similarity > 0.7:
#             return "adds_context"
#         elif similarity > 0.65:
#             text2 = (article2.get("story", "") or "").lower()
#             contradict_words = ["however", "but", "denies", "refutes", "opposes"]
#             if any(w in text2 for w in contradict_words):
#                 return "contradicts"
#             escalate_words = ["escalates", "intensifies", "worsens", "increases"]
#             if any(w in text2 for w in escalate_words):
#                 return "escalates"
#             return "adds_context"
#         return None

#     def build_narrative_graph(self):
#         G = nx.DiGraph()
#         for idx, a in enumerate(self.relevant_articles):
#             G.add_node(idx, title=a.get("title", ""), url=a.get("url", ""))
#         texts = [f"{a.get('title','')} {a.get('story','')}" for a in self.relevant_articles]
#         emb = np.asarray(self.embedding_model.encode(texts, convert_to_numpy=True), dtype=np.float32)
#         dates = [parse_date_flexible(a.get("published_at", "") or a.get("published", "")) or datetime.min for a in self.relevant_articles]
#         for i in range(len(self.relevant_articles)):
#             for j in range(i + 1, len(self.relevant_articles)):
#                 sim = float(util.cos_sim(emb[i:i+1], emb[j:j+1])[0][0].cpu().numpy())
#                 if sim > 0.6 and dates[j] >= dates[i]:
#                     relation = self._detect_relation_type(self.relevant_articles[i], self.relevant_articles[j], sim)
#                     if relation:
#                         G.add_edge(i, j, relation=relation, weight=round(sim, 3))
#         graph_data = {
#             "nodes": [{"id": idx, "title": node.get("title", ""), "url": node.get("url", "")} for idx, node in G.nodes(data=True)],
#             "edges": [{"source": u, "target": v, "relation": data["relation"], "weight": data["weight"]} for u, v, data in G.edges(data=True)]
#         }
#         return graph_data

#     def build_narrative(self, topic, relevance_threshold=0.8, scoring_strategy="simple", recompute_embeddings=False, **kwargs):
#         # load data & filter
#         self.load_dataset()
#         self.filter_by_rating(min_rating=8)

#         # find relevant
#         self.find_relevant_articles(topic, threshold=relevance_threshold, scoring_strategy=scoring_strategy, recompute_embeddings=recompute_embeddings, **kwargs)

#         if not self.relevant_articles:
#             return {
#                 "error": f"No articles found for topic: {topic} (with relevance > {relevance_threshold})",
#                 "narrative_summary": "",
#                 "timeline": [],
#                 "clusters": [],
#                 "graph": {"nodes": [], "edges": []}
#             }

#         eprint("Generating narrative summary...")
#         summary = self.generate_narrative_summary(topic)

#         eprint("Creating timeline...")
#         timeline = self.create_timeline()

#         eprint("Clustering articles...")
#         clusters = self.create_narrative_clusters()

#         eprint("Building narrative graph...")
#         graph = self.build_narrative_graph()

#         return {
#             "topic": topic,
#             "relevance_threshold": relevance_threshold,
#             "scoring_strategy": scoring_strategy,
#             "model": self.model_name,
#             "total_relevant_articles": len(self.relevant_articles),
#             "narrative_summary": summary,
#             "timeline": timeline,
#             "clusters": clusters,
#             "graph": graph
#         }


# def main():
#     parser = argparse.ArgumentParser(description="Narrative Builder with cached embeddings and optional FAISS")
#     parser.add_argument("--topic", type=str, required=True)
#     parser.add_argument("--dataset", type=str, default="Dataset_for_second_task.json")
#     parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
#     parser.add_argument("--scoring", type=str, choices=["simple", "weighted", "hybrid"], default="simple")
#     parser.add_argument("--relevance-threshold", type=float, default=0.8)
#     parser.add_argument("--title-weight", type=float, default=0.7)
#     parser.add_argument("--semantic-weight", type=float, default=0.7)
#     parser.add_argument("--use-faiss", action="store_true", help="Use FAISS for fast search (requires faiss installed).")
#     parser.add_argument("--recompute", action="store_true", help="Force re-computation of embeddings even if cache exists.")
#     parser.add_argument("--cache-dir", type=str, default="emb_cache")
#     args = parser.parse_args()

#     try:
#         builder = ImprovedNarrativeBuilder(dataset_path=args.dataset, model_name=args.model, cache_dir=args.cache_dir, use_faiss=args.use_faiss)
#         kwargs = {}
#         if args.scoring == "weighted":
#             kwargs["title_weight"] = args.title_weight
#         elif args.scoring == "hybrid":
#             kwargs["semantic_weight"] = args.semantic_weight

#         result = builder.build_narrative(args.topic, relevance_threshold=args.relevance_threshold, scoring_strategy=args.scoring, recompute_embeddings=args.recompute, **kwargs)
#         print(json.dumps(result, indent=2, ensure_ascii=False))

#     except FileNotFoundError as fnf:
#         eprint(json.dumps({"error": f"Dataset not found: {args.dataset}"}))
#         sys.exit(1)
#     except Exception as e:
#         eprint(json.dumps({"error": f"Error building narrative: {str(e)}"}))
#         sys.exit(1)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Ultra-Fast Narrative Builder with persistent embedding storage.

This version creates embeddings ONCE and stores them permanently.
All subsequent runs directly load pre-computed embeddings.

Usage:
    # First time: Build embeddings (slow, one-time operation)
    python narrative_builder_fast.py --build-embeddings
    
    # All subsequent searches (instant)
    python narrative_builder_fast.py --topic "Hyderabad Metro"
    python narrative_builder_fast.py --topic "police reforms"
    python narrative_builder_fast.py --topic "infrastructure" --threshold 0.25
"""

import argparse
import json
import os
import sys
import hashlib
from datetime import datetime
from collections import defaultdict, Counter
import re
import warnings
from typing import List, Dict, Tuple, Optional
import time

import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import MiniBatchKMeans
import networkx as nx

# Optional FAISS
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

warnings.filterwarnings("ignore")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_date_flexible(published):
    """Optimized date parsing."""
    if not published or not isinstance(published, str):
        return None
    s = published.strip()
    
    if 'T' in s:
        try:
            s2 = s.replace("Z", "+00:00") if s.endswith("Z") else s
            dt = datetime.fromisoformat(s2)
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except Exception:
            pass
    
    for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


class PersistentEmbeddingStore:
    """
    Manages permanent storage of embeddings.
    Embeddings are computed ONCE and reused forever.
    """

    def __init__(self, cache_dir="embeddings_store"):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir
        self.embeddings_file = os.path.join(cache_dir, "embeddings.npz")
        self.metadata_file = os.path.join(cache_dir, "metadata.json")
        self.faiss_file = os.path.join(cache_dir, "index.faiss")

    def exists(self):
        """Check if embeddings already exist."""
        return os.path.exists(self.embeddings_file) and os.path.exists(self.metadata_file)

    def save(self, embeddings: np.ndarray, articles: list, model_name: str, dataset_info: dict):
        """
        Save embeddings permanently.
        Only needs to be called ONCE.
        """
        eprint(f"Saving embeddings to {self.embeddings_file}...")
        
        # Save embeddings
        np.savez_compressed(
            self.embeddings_file,
            embeddings=embeddings.astype(np.float32),
            article_ids=np.array([a.get('id', i) for i, a in enumerate(articles)], dtype=object)
        )
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "num_articles": len(articles),
            "embedding_dim": embeddings.shape[1],
            "created_at": datetime.now().isoformat(),
            "dataset_info": dataset_info
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        eprint(f"✓ Saved {len(articles)} embeddings ({embeddings.shape[1]}D)")

    def load(self):
        """
        Load pre-computed embeddings instantly.
        This is called every time you search.
        """
        if not self.exists():
            raise FileNotFoundError(
                "Embeddings not found! Run with --build-embeddings first."
            )
        
        eprint(f"Loading pre-computed embeddings...")
        start = time.time()
        
        data = np.load(self.embeddings_file, allow_pickle=True)
        embeddings = data['embeddings']
        article_ids = data['article_ids'].tolist()
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        elapsed = time.time() - start
        eprint(f"✓ Loaded {len(embeddings)} embeddings in {elapsed:.3f}s")
        
        return embeddings, article_ids, metadata

    def save_faiss_index(self, index):
        """Save FAISS index for ultra-fast search."""
        if _HAS_FAISS and index is not None:
            faiss.write_index(index, self.faiss_file)
            eprint(f"✓ Saved FAISS index to {self.faiss_file}")

    def load_faiss_index(self):
        """Load FAISS index."""
        if _HAS_FAISS and os.path.exists(self.faiss_file):
            return faiss.read_index(self.faiss_file)
        return None


class FastNarrativeBuilder:
    """
    Ultra-fast narrative builder that uses pre-computed embeddings.
    """

    def __init__(self, dataset_path="Dataset_for_second_task.json", 
                 model_name="all-MiniLM-L6-v2",
                 store_dir="embeddings_store"):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.store = PersistentEmbeddingStore(cache_dir=store_dir)
        
        self.embedding_model = None  # Lazy load only when needed
        self.articles = []
        self.filtered_articles = []
        self.relevant_articles = []
        
        # These will be loaded from disk
        self.embeddings = None
        self.article_ids = None
        self.faiss_index = None

    def _load_model(self):
        """Lazy load model only when building embeddings."""
        if self.embedding_model is None:
            eprint(f"Loading model: {self.model_name}...")
            start = time.time()
            self.embedding_model = SentenceTransformer(self.model_name)
            eprint(f"✓ Model loaded in {time.time()-start:.2f}s")

    def load_dataset(self):
        """Load articles from JSON."""
        eprint(f"Loading dataset from {self.dataset_path}...")
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.articles = data.get("items", []) if isinstance(data, dict) else data
        eprint(f"✓ Loaded {len(self.articles)} articles")

    def filter_by_rating(self, min_rating=8):
        """Filter articles by source rating."""
        self.filtered_articles = [
            a for a in self.articles
            if float(a.get("source_rating", 0) or 0) >= min_rating
        ]
        eprint(f"✓ Filtered to {len(self.filtered_articles)} articles (rating >= {min_rating})")

    def build_and_save_embeddings(self, force_rebuild=False):
        """
        ONE-TIME operation: Build embeddings and save them.
        This can take 30-60 seconds but only needs to run ONCE.
        """
        if self.store.exists() and not force_rebuild:
            eprint("\n⚠️  Embeddings already exist!")
            eprint("Use --force-rebuild to recreate them.")
            return

        eprint("\n" + "="*60)
        eprint("BUILDING EMBEDDINGS (one-time operation)")
        eprint("="*60)
        
        self.load_dataset()
        self.filter_by_rating(min_rating=8)
        
        if not self.filtered_articles:
            raise ValueError("No articles found after filtering!")

        # Load model
        self._load_model()

        # Prepare texts (combined title + content)
        eprint("\nPreparing texts for embedding...")
        texts = [
            f"{a.get('title', '')} {a.get('story', '')}"
            for a in self.filtered_articles
        ]
        
        # Generate embeddings
        eprint(f"\nComputing embeddings for {len(texts)} articles...")
        eprint("This will take 30-60 seconds but only happens ONCE!")
        start = time.time()
        
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device='cpu'
        )
        embeddings = embeddings.astype(np.float32)
        
        elapsed = time.time() - start
        eprint(f"\n✓ Embeddings computed in {elapsed:.2f}s ({len(texts)/elapsed:.1f} articles/sec)")
        
        # Save embeddings permanently
        dataset_info = {
            "path": self.dataset_path,
            "total_articles": len(self.articles),
            "filtered_articles": len(self.filtered_articles),
            "min_rating": 8
        }
        self.store.save(embeddings, self.filtered_articles, self.model_name, dataset_info)
        
        # Build and save FAISS index if available
        if _HAS_FAISS:
            eprint("\nBuilding FAISS index for ultra-fast search...")
            embeddings_normalized = embeddings.copy()
            faiss.normalize_L2(embeddings_normalized)
            
            d = embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(embeddings_normalized)
            
            self.store.save_faiss_index(index)
        
        eprint("\n" + "="*60)
        eprint("✓ EMBEDDINGS SAVED! All future searches will be instant.")
        eprint("="*60 + "\n")

    def load_embeddings(self):
        """
        INSTANT operation: Load pre-computed embeddings from disk.
        This takes <1 second.
        """
        self.embeddings, self.article_ids, metadata = self.store.load()
        
        # Load articles to get metadata
        self.load_dataset()
        self.filter_by_rating(min_rating=8)
        
        # Load FAISS index if available
        if _HAS_FAISS:
            self.faiss_index = self.store.load_faiss_index()
            if self.faiss_index:
                eprint("✓ FAISS index loaded for ultra-fast search")

    def expand_query(self, topic):
        """Expand query with synonyms."""
        expansions = {
            "metro": ["metro rail", "subway", "transit"],
            "police": ["law enforcement", "officers"],
            "infrastructure": ["development", "construction"],
            "election": ["voting", "polls", "campaign"],
            "hyderabad": ["telangana", "secunderabad"],
        }
        expanded = [topic]
        tl = topic.lower()
        for k, syns in expansions.items():
            if k in tl:
                expanded.extend(syns[:2])
        return " ".join(expanded)

    def search_articles(self, topic, threshold=0.3, top_k=100):
        """
        INSTANT search using pre-computed embeddings.
        """
        eprint(f"\nSearching for '{topic}' (threshold={threshold})...")
        start = time.time()
        
        if self.embeddings is None:
            raise RuntimeError("Embeddings not loaded! Call load_embeddings() first.")
        
        # Load model only for query encoding
        self._load_model()
        
        # Encode query
        expanded_topic = self.expand_query(topic)
        query_emb = self.embedding_model.encode(
            [expanded_topic],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype(np.float32)
        
        # Search using FAISS or brute force
        if _HAS_FAISS and self.faiss_index is not None:
            # Ultra-fast FAISS search
            faiss.normalize_L2(query_emb)
            D, I = self.faiss_index.search(query_emb, top_k)
            results = [(int(I[0][i]), float(D[0][i])) for i in range(len(I[0])) if I[0][i] != -1]
        else:
            # Fast vectorized search
            sims = np.dot(self.embeddings, query_emb.T).flatten()
            top_indices = np.argpartition(sims, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(-sims[top_indices])]
            results = [(int(idx), float(sims[idx])) for idx in top_indices]
        
        # Filter by threshold
        selected = [(idx, score) for idx, score in results if score >= threshold]
        
        # Map to articles
        self.relevant_articles = []
        for idx, score in selected:
            if idx < len(self.filtered_articles):
                article = dict(self.filtered_articles[idx])
                article["relevance_score"] = float(score)
                self.relevant_articles.append(article)
        
        self.relevant_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        elapsed = time.time() - start
        eprint(f"✓ Found {len(self.relevant_articles)} articles in {elapsed:.3f}s")
        
        if results:
            top_scores = [r[1] for r in results[:10]]
            eprint(f"  Top 10 scores: {[f'{s:.3f}' for s in top_scores]}")

    def generate_narrative_summary(self, topic, max_articles=5):
        """Generate extractive summary."""
        if not self.relevant_articles:
            return f"No articles found related to '{topic}'."
        
        sentences = []
        for art in self.relevant_articles[:max_articles]:
            story = art.get("story", "") or art.get("title", "")
            first = story.split(". ")[0].strip()
            if first and len(first) > 20:
                sentences.append(first)
        
        return " ".join(sentences[:5]) + "."

    def create_timeline(self):
        """Create chronological timeline."""
        timeline = []
        for article in self.relevant_articles[:50]:
            published = article.get("published_at") or article.get("published") or ""
            date_obj = parse_date_flexible(published)
            date_str = date_obj.strftime("%Y-%m-%d") if date_obj else "Unknown"
            
            story = article.get("story", "") or ""
            first_sentence = story.split(". ")[0] if story else article.get("title", "")
            
            timeline.append({
                "date_obj": date_obj,
                "date": date_str,
                "headline": article.get("title", ""),
                "url": article.get("url", ""),
                "why_it_matters": first_sentence[:200],
                "relevance_score": round(float(article.get("relevance_score", 0)), 3)
            })
        
        timeline.sort(key=lambda x: (x["date_obj"] is None, x["date_obj"] or datetime.max))
        for item in timeline:
            item.pop("date_obj", None)
        
        return timeline

    def create_narrative_clusters(self, n_clusters=3):
        """Cluster articles by theme."""
        if len(self.relevant_articles) < 3:
            return [{
                "theme": "Main Topic",
                "articles": [{"title": a.get("title", ""), "url": a.get("url", ""), 
                             "relevance": round(float(a.get("relevance_score", 0)), 3)}
                            for a in self.relevant_articles]
            }]
        
        # Use pre-computed embeddings where possible
        indices = []
        for art in self.relevant_articles:
            # Find index in filtered_articles
            for i, fa in enumerate(self.filtered_articles):
                if fa.get("url") == art.get("url") or fa.get("title") == art.get("title"):
                    indices.append(i)
                    break
        
        if not indices:
            return []
        
        # Get embeddings for relevant articles
        relevant_embeddings = self.embeddings[indices]
        
        n_clusters = min(n_clusters, max(2, len(self.relevant_articles) // 3))
        
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=100)
        labels = kmeans.fit_predict(relevant_embeddings)
        
        clusters_dict = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters_dict[label].append(self.relevant_articles[idx])
        
        clusters = []
        for cluster_id, arts in clusters_dict.items():
            all_words = " ".join([a.get("title", "") for a in arts[:5]])
            words = re.findall(r"\b[A-Z][a-z]{3,}\b", all_words)
            theme = f"Cluster {cluster_id+1}"
            if words:
                common = Counter(words).most_common(2)
                theme = " ".join([w for w, _ in common])
            
            clusters.append({
                "theme": theme,
                "article_count": len(arts),
                "articles": [
                    {"title": a.get("title", ""), "url": a.get("url", ""),
                     "relevance": round(float(a.get("relevance_score", 0)), 3)}
                    for a in sorted(arts, key=lambda x: float(x.get("relevance_score", 0)), reverse=True)[:5]
                ]
            })
        
        return clusters

    def build_narrative_graph(self):
        """Build relationship graph."""
        top_articles = self.relevant_articles[:30]
        
        G = nx.DiGraph()
        for idx, a in enumerate(top_articles):
            G.add_node(idx, title=a.get("title", "")[:100], url=a.get("url", ""))
        
        # Get embeddings for these articles
        indices = []
        for art in top_articles:
            for i, fa in enumerate(self.filtered_articles):
                if fa.get("url") == art.get("url"):
                    indices.append(i)
                    break
        
        if indices:
            article_embeddings = self.embeddings[indices]
            sim_matrix = np.dot(article_embeddings, article_embeddings.T)
            
            for i in range(len(top_articles)):
                for j in range(i + 1, len(top_articles)):
                    if sim_matrix[i, j] > 0.7:
                        G.add_edge(i, j, relation="related", weight=round(float(sim_matrix[i, j]), 3))
        
        graph_data = {
            "nodes": [{"id": idx, "title": node.get("title", ""), "url": node.get("url", "")} 
                     for idx, node in G.nodes(data=True)],
            "edges": [{"source": u, "target": v, "relation": data["relation"], "weight": data["weight"]} 
                     for u, v, data in G.edges(data=True)]
        }
        return graph_data

    def build_narrative(self, topic, relevance_threshold=0.3):
        """
        Main narrative building pipeline.
        Uses pre-computed embeddings for instant results.
        """
        total_start = time.time()
        
        # Load pre-computed embeddings (instant)
        self.load_embeddings()
        
        # Search (instant with FAISS)
        self.search_articles(topic, threshold=relevance_threshold)
        
        if not self.relevant_articles:
            return {
                "error": f"No articles found for topic: {topic}",
                "suggestion": "Try lowering threshold to 0.2 or use different terms"
            }
        
        eprint("\nGenerating narrative components...")
        summary = self.generate_narrative_summary(topic)
        timeline = self.create_timeline()
        clusters = self.create_narrative_clusters()
        graph = self.build_narrative_graph()
        
        total_time = time.time() - total_start
        eprint(f"\n{'='*60}")
        eprint(f"✓ TOTAL TIME: {total_time:.2f}s")
        eprint(f"{'='*60}\n")
        
        return {
            "topic": topic,
            "relevance_threshold": relevance_threshold,
            "model": self.model_name,
            "execution_time_seconds": round(total_time, 2),
            "total_relevant_articles": len(self.relevant_articles),
            "narrative_summary": summary,
            "timeline": timeline,
            "clusters": clusters,
            "graph": graph
        }


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-Fast Narrative Builder with Persistent Embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First time: Build embeddings (one-time, takes ~60s)
  python narrative_builder_fast.py --build-embeddings
  
  # Search (instant after embeddings are built)
  python narrative_builder_fast.py --topic "Hyderabad Metro"
  python narrative_builder_fast.py --topic "police reforms" --threshold 0.25
  
  # Rebuild embeddings if dataset changes
  python narrative_builder_fast.py --build-embeddings --force-rebuild
        """
    )
    parser.add_argument("--topic", type=str, help="Topic to search for")
    parser.add_argument("--dataset", type=str, default="Dataset_for_second_task.json")
    parser.add_argument("--threshold", type=float, default=0.3, help="Relevance threshold (0.2-0.4)")
    parser.add_argument("--build-embeddings", action="store_true", 
                       help="Build and save embeddings (one-time operation)")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="Force rebuild embeddings even if they exist")
    parser.add_argument("--store-dir", type=str, default="embeddings_store",
                       help="Directory to store embeddings")
    
    args = parser.parse_args()

    try:
        builder = FastNarrativeBuilder(
            dataset_path=args.dataset,
            store_dir=args.store_dir
        )
        
        if args.build_embeddings:
            # One-time embedding generation
            builder.build_and_save_embeddings(force_rebuild=args.force_rebuild)
        elif args.topic:
            # Fast search using pre-computed embeddings
            result = builder.build_narrative(
                args.topic,
                relevance_threshold=args.threshold
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            parser.print_help()
            eprint("\n⚠️  Please specify either --build-embeddings or --topic")
            sys.exit(1)

    except FileNotFoundError as e:
        if "Embeddings not found" in str(e):
            eprint("\n❌ Embeddings not found!")
            eprint("Run this first: python narrative_builder_fast.py --build-embeddings\n")
        else:
            eprint(f"\n❌ Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        import traceback
        eprint(f"\n❌ Error: {str(e)}\n")
        eprint(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()