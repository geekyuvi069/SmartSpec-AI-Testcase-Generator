@@ .. @@
 # app.py
 from flask import Flask, request, jsonify, send_from_directory
 from flask_cors import CORS
 import os
+import json
 from datetime import datetime

-from src.preprocessing import extract_text_from_pdf, clean_text, split_into_chunks
-from src.semantic_search import build_index, search
+from src.preprocessing import extract_text_from_pdf, clean_text, split_into_chunks, extract_requirements_metadata
+from src.semantic_search import build_index, search, get_index_stats
+from utils.extract_text import load_existing_test_cases
+from utils.test_case_utils import TestCaseManager

 # Initialize Flask
 app = Flask(__name__, static_folder="web", static_url_path="")
 CORS(app)

 document_chunks = []
+document_metadata = {}
+test_case_manager = TestCaseManager()

 @app.route("/")
 def index():
     return send_from_directory("web", "index.html")

-@app.route("/upload", methods=["POST"])
-def upload():
+@app.route("/upload-srs", methods=["POST"])
+def upload_srs():
     global document_chunks

     if "file" not in request.files:
         return jsonify({"error": "No file provided"}), 400

     file = request.files["file"]
     if file.filename == "":
         return jsonify({"error": "Empty filename"}), 400

+    if not file.filename.lower().endswith('.pdf'):
+        return jsonify({"error": "Only PDF files are supported for SRS upload"}), 400

     date_folder = datetime.now().strftime("%Y-%m-%d")
     save_folder = os.path.join("data", date_folder)
     os.makedirs(save_folder, exist_ok=True)

     save_path = os.path.join(save_folder, file.filename)
     file.save(save_path)

-    # Process document
-    raw_text = extract_text_from_pdf(save_path)
-    clean = clean_text(raw_text)
-    chunks = split_into_chunks(clean)
-    document_chunks = chunks
+    try:
+        # Process document
+        raw_text = extract_text_from_pdf(save_path)
+        clean = clean_text(raw_text)
+        chunks = split_into_chunks(clean)
+        document_chunks = chunks
+        
+        # Extract metadata
+        global document_metadata
+        document_metadata = extract_requirements_metadata(clean)
+        document_metadata['filename'] = file.filename
+        document_metadata['upload_date'] = datetime.now().isoformat()

-    # Build embeddings index
-    build_index(chunks)
+        # Build embeddings index
+        build_index(chunks, document_metadata)

-    return jsonify({
-        "message": "Document processed successfully",
-        "chunks": len(chunks),
-        "savedPath": save_path
-    })
+        return jsonify({
+            "message": "SRS document processed successfully",
+            "chunks": len(chunks),
+            "metadata": document_metadata,
+            "savedPath": save_path
+        })
+    except Exception as e:
+        return jsonify({"error": f"Error processing document: {str(e)}"}), 500
+
+@app.route("/upload-test-cases", methods=["POST"])
+def upload_test_cases():
+    if "file" not in request.files:
+        return jsonify({"error": "No file provided"}), 400
+
+    file = request.files["file"]
+    if file.filename == "":
+        return jsonify({"error": "Empty filename"}), 400
+
+    if not file.filename.lower().endswith('.json'):
+        return jsonify({"error": "Only JSON files are supported for test cases"}), 400
+
+    try:
+        # Read and parse JSON content
+        content = file.read().decode('utf-8')
+        test_cases_data = json.loads(content)
+        
+        # Load into test case manager
+        test_case_manager.load_existing_test_cases(test_cases_data)
+        
+        # Save uploaded file for reference
+        date_folder = datetime.now().strftime("%Y-%m-%d")
+        save_folder = os.path.join("data", date_folder)
+        os.makedirs(save_folder, exist_ok=True)
+        
+        save_path = os.path.join(save_folder, f"existing_test_cases_{file.filename}")
+        with open(save_path, 'w', encoding='utf-8') as f:
+            json.dump(test_cases_data, f, indent=2)
+        
+        return jsonify({
+            "message": "Test cases loaded successfully",
+            "existing_count": len(test_case_manager.existing_test_cases),
+            "savedPath": save_path
+        })
+        
+    except json.JSONDecodeError as e:
+        return jsonify({"error": f"Invalid JSON format: {str(e)}"}), 400
+    except Exception as e:
+        return jsonify({"error": f"Error processing test cases: {str(e)}"}), 500

 @app.route("/query", methods=["POST"])
 def query():
     data = request.get_json()
     query_text = data.get("query", "").strip()
+    max_test_cases = data.get("max_test_cases", 5)
+    
     if not query_text:
         return jsonify({"error": "Empty query."}), 400

-    # Retrieve top similar chunks
-    retrieved_chunks = search(query_text, top_k=3)
+    try:
+        # Retrieve top similar chunks
+        search_results = search(query_text, top_k=max_test_cases)
+        
+        if not search_results:
+            return jsonify({
+                "query": query_text,
+                "testCases": [],
+                "message": "No relevant content found. Please ensure an SRS document is uploaded."
+            })

-    # Generate test cases from retrieved text
-    # For now, do simple templated generation
-    test_cases = []
-    for i, chunk in enumerate(retrieved_chunks, 1):
-        test_cases.append({
-            "title": f"Test Case {i}",
-            "description": f"Generated from relevant content chunk.",
-            "steps": chunk,
-            "expected": "Behavior as specified in the requirements."
-        })
+        # Generate test cases from retrieved chunks
+        generated_test_cases = []
+        new_cases_count = 0
+        duplicate_cases_count = 0
+        
+        for i, result in enumerate(search_results, 1):
+            chunk_text = result['text']
+            relevance = result['relevance']
+            
+            # Generate test cases from this chunk
+            chunk_test_cases = test_case_manager.generate_test_cases_from_chunk(
+                chunk_text, 
+                requirement_section=f"Query Result {i}"
+            )
+            
+            for test_case in chunk_test_cases:
+                # Add relevance score to test case
+                test_case['relevance_score'] = relevance
+                test_case['source_chunk'] = chunk_text[:200] + "..." if len(chunk_text) > 200 else chunk_text
+                
+                # Check for duplicates and add if unique
+                is_added, case_id = test_case_manager.add_test_case(test_case)
+                
+                if is_added:
+                    generated_test_cases.append(test_case)
+                    new_cases_count += 1
+                else:
+                    duplicate_cases_count += 1

-    return jsonify({
-        "query": query_text,
-        "testCases": test_cases
-    })
+        return jsonify({
+            "query": query_text,
+            "testCases": generated_test_cases,
+            "stats": {
+                "new_cases": new_cases_count,
+                "duplicates_avoided": duplicate_cases_count,
+                "total_existing": len(test_case_manager.existing_test_cases),
+                "search_results": len(search_results)
+            }
+        })
+        
+    except Exception as e:
+        return jsonify({"error": f"Error generating test cases: {str(e)}"}), 500
+
+@app.route("/download-test-cases", methods=["GET"])
+def download_test_cases():
+    try:
+        # Create merged output directory
+        output_dir = "merged_output"
+        os.makedirs(output_dir, exist_ok=True)
+        
+        # Generate filename with timestamp
+        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
+        filename = f"merged_test_cases_{timestamp}.json"
+        output_path = os.path.join(output_dir, filename)
+        
+        # Save merged test cases
+        merged_data = test_case_manager.save_merged_test_cases(output_path)
+        
+        return jsonify({
+            "message": "Test cases file generated successfully",
+            "filename": filename,
+            "path": output_path,
+            "stats": merged_data["metadata"]
+        })
+        
+    except Exception as e:
+        return jsonify({"error": f"Error generating download file: {str(e)}"}), 500
+
+@app.route("/download-file/<filename>", methods=["GET"])
+def download_file(filename):
+    try:
+        return send_from_directory("merged_output", filename, as_attachment=True)
+    except Exception as e:
+        return jsonify({"error": f"File not found: {str(e)}"}), 404
+
+@app.route("/status", methods=["GET"])
+def get_status():
+    """Get current system status."""
+    index_stats = get_index_stats()
+    
+    return jsonify({
+        "document_status": {
+            "chunks_loaded": len(document_chunks),
+            "metadata": document_metadata
+        },
+        "test_cases_status": {
+            "existing_count": len(test_case_manager.existing_test_cases),
+            "new_count": len(test_case_manager.new_test_cases)
+        },
+        "index_status": index_stats
+    })

 if __name__ == "__main__":
     os.makedirs("data", exist_ok=True)
+    os.makedirs("merged_output", exist_ok=True)
     app.run(debug=True)