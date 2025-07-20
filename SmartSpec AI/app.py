# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from datetime import datetime

from src.preprocessing import extract_text_from_pdf, clean_text, split_into_chunks
from src.semantic_search import build_index, search

# Initialize Flask
app = Flask(__name__, static_folder="web", static_url_path="")
CORS(app)

document_chunks = []

@app.route("/")
def index():
    return send_from_directory("web", "index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global document_chunks

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    date_folder = datetime.now().strftime("%Y-%m-%d")
    save_folder = os.path.join("data", date_folder)
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, file.filename)
    file.save(save_path)

    # Process document
    raw_text = extract_text_from_pdf(save_path)
    clean = clean_text(raw_text)
    chunks = split_into_chunks(clean)
    document_chunks = chunks

    # Build embeddings index
    build_index(chunks)

    return jsonify({
        "message": "Document processed successfully",
        "chunks": len(chunks),
        "savedPath": save_path
    })

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()
    query_text = data.get("query", "").strip()
    if not query_text:
        return jsonify({"error": "Empty query."}), 400

    # Retrieve top similar chunks
    retrieved_chunks = search(query_text, top_k=3)

    # Generate test cases from retrieved text
    # For now, do simple templated generation
    test_cases = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        test_cases.append({
            "title": f"Test Case {i}",
            "description": f"Generated from relevant content chunk.",
            "steps": chunk,
            "expected": "Behavior as specified in the requirements."
        })

    return jsonify({
        "query": query_text,
        "testCases": test_cases
    })

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    app.run(debug=True)
