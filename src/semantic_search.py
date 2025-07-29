@@ .. @@
 # src/semantic_search.py
 from sentence_transformers import SentenceTransformer
 import faiss
 import numpy as np
+import json
+from datetime import datetime

 # Load the embedding model (e.g., MiniLM)
 embedder = SentenceTransformer('all-MiniLM-L6-v2')

 # Initialize index and chunk mapping
 index = None
 chunks = []
+chunk_metadata = []

 def embed_text(text):
-    """Embed a single text (query or chunk)."""
+    """Embed a single text with error handling."""
+    if not text or not text.strip():
+        return np.zeros(384, dtype=np.float32)  # Return zero vector for empty text
+        
     embedding = embedder.encode([text])[0]
     return embedding.astype(np.float32)

-def build_index(text_chunks):
-    """Embed all chunks and build FAISS index."""
-    global index, chunks
-    embeddings = [embed_text(chunk) for chunk in text_chunks]
+def build_index(text_chunks, metadata=None):
+    """Enhanced index building with metadata support."""
+    global index, chunks, chunk_metadata
+    
+    if not text_chunks:
+        raise ValueError("No text chunks provided for indexing")
+    
+    print(f"Building index for {len(text_chunks)} chunks...")
+    
+    embeddings = []
+    valid_chunks = []
+    valid_metadata = []
+    
+    for i, chunk in enumerate(text_chunks):
+        if chunk and chunk.strip():
+            embedding = embed_text(chunk)
+            embeddings.append(embedding)
+            valid_chunks.append(chunk)
+            
+            # Add metadata for each chunk
+            chunk_meta = {
+                'chunk_id': i,
+                'length': len(chunk.split()),
+                'created_at': datetime.now().isoformat()
+            }
+            if metadata:
+                chunk_meta.update(metadata)
+            valid_metadata.append(chunk_meta)
+    
+    if not embeddings:
+        raise ValueError("No valid chunks found for indexing")
+        
     dimension = embeddings[0].shape[0]

     # Create FAISS index
     index = faiss.IndexFlatL2(dimension)
     index.add(np.array(embeddings))

-    chunks = text_chunks  # Save mapping
+    chunks = valid_chunks
+    chunk_metadata = valid_metadata
+    
+    print(f"Index built successfully with {len(valid_chunks)} chunks")

-def search(query_text, top_k=3):
-    """Search most similar chunks to the query."""
+def search(query_text, top_k=5, score_threshold=None):
+    """Enhanced search with scoring and filtering."""
     if index is None:
-        raise ValueError("Index not built yet.")
+        raise ValueError("Index not built yet. Please upload and process a document first.")
+    
+    if not query_text or not query_text.strip():
+        return []

     query_vec = embed_text(query_text).reshape(1, -1)
     distances, indices = index.search(query_vec, top_k)

     results = []
-    for i in indices[0]:
-        results.append(chunks[i])
+    for distance, idx in zip(distances[0], indices[0]):
+        if score_threshold is None or distance <= score_threshold:
+            result = {
+                'text': chunks[idx],
+                'score': float(distance),
+                'metadata': chunk_metadata[idx] if idx < len(chunk_metadata) else {},
+                'relevance': max(0, 1 - distance / 2)  # Convert distance to relevance score
+            }
+            results.append(result)
+    
     return results
+
+def get_index_stats():
+    """Get statistics about the current index."""
+    if index is None:
+        return {"status": "No index built"}
+    
+    return {
+        "status": "Index ready",
+        "total_chunks": len(chunks),
+        "index_size": index.ntotal,
+        "dimension": index.d,
+        "avg_chunk_length": sum(len(chunk.split()) for chunk in chunks) / len(chunks) if chunks else 0
+    }
+
+def search_by_category(query_text, category_filter=None, top_k=5):
+    """Search with category filtering."""
+    results = search(query_text, top_k * 2)  # Get more results to filter
+    
+    if category_filter:
+        filtered_results = []
+        for result in results:
+            chunk_text = result['text'].lower()
+            if category_filter.lower() in chunk_text:
+                filtered_results.append(result)
+                if len(filtered_results) >= top_k:
+                    break
+        return filtered_results
+    
+    return results[:top_k]
+
+def find_similar_chunks(chunk_text, top_k=3):
+    """Find chunks similar to a given chunk."""
+    if not chunks:
+        return []
+    
+    return search(chunk_text, top_k)