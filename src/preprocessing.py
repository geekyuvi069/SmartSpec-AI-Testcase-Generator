@@ .. @@
 import re
 from PyPDF2 import PdfReader

 def extract_text_from_pdf(file_path):
     """
-    Extracts text from a PDF file.
+    Enhanced text extraction from PDF with better error handling.
     """
     text = ""
-    with open(file_path, "rb") as f:
-        reader = PdfReader(f)
-        for page in reader.pages:
-            page_text = page.extract_text()
-            if page_text:
-                text += page_text + "\n"
+    try:
+        with open(file_path, "rb") as f:
+            reader = PdfReader(f)
+            for page_num, page in enumerate(reader.pages):
+                page_text = page.extract_text()
+                if page_text:
+                    text += f"\n--- Page {page_num + 1} ---\n"
+                    text += page_text + "\n"
+    except Exception as e:
+        raise Exception(f"Error extracting text from PDF: {str(e)}")
+        
     return text


 def clean_text(text):
     """
-    Cleans text by normalizing spaces and removing extra newlines.
+    Enhanced text cleaning with structure preservation.
     """
-    # Replace multiple spaces/newlines with single space
-    text = re.sub(r"\s+", " ", text)
-    return text.strip()
+    # Remove excessive whitespace but preserve paragraph structure
+    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
+    
+    # Normalize spaces within lines
+    lines = text.split('\n')
+    cleaned_lines = []
+    
+    for line in lines:
+        cleaned_line = re.sub(r'\s+', ' ', line.strip())
+        if cleaned_line:
+            cleaned_lines.append(cleaned_line)
+        elif cleaned_lines and cleaned_lines[-1]:
+            cleaned_lines.append('')
+    
+    return '\n'.join(cleaned_lines)


-def split_into_chunks(text, chunk_size=200):
+def split_into_chunks(text, chunk_size=300, overlap=50):
     """
-    Splits text into chunks of specified word count.
+    Enhanced chunking with overlap for better context preservation.
     """
     words = text.split()
     chunks = []
-    for i in range(0, len(words), chunk_size):
-        chunk = " ".join(words[i:i + chunk_size])
-        chunks.append(chunk)
+    
+    for i in range(0, len(words), chunk_size - overlap):
+        end_idx = min(i + chunk_size, len(words))
+        chunk = " ".join(words[i:end_idx])
+        
+        if chunk.strip():  # Only add non-empty chunks
+            chunks.append(chunk.strip())
+            
+        if end_idx >= len(words):
+            break
+            
     return chunks
+
+def extract_requirements_metadata(text):
+    """
+    Extract metadata from SRS document.
+    """
+    metadata = {
+        'document_type': 'SRS',
+        'sections': [],
+        'requirements_count': 0,
+        'functional_requirements': 0,
+        'non_functional_requirements': 0
+    }
+    
+    # Count different types of requirements
+    functional_patterns = [
+        r'functional\s+requirement',
+        r'FR\d+',
+        r'the\s+system\s+(shall|must|should)'
+    ]
+    
+    non_functional_patterns = [
+        r'non-functional\s+requirement',
+        r'NFR\d+',
+        r'performance\s+requirement',
+        r'security\s+requirement'
+    ]
+    
+    text_lower = text.lower()
+    
+    for pattern in functional_patterns:
+        metadata['functional_requirements'] += len(re.findall(pattern, text_lower))
+    
+    for pattern in non_functional_patterns:
+        metadata['non_functional_requirements'] += len(re.findall(pattern, text_lower))
+    
+    metadata['requirements_count'] = metadata['functional_requirements'] + metadata['non_functional_requirements']
+    
+    return metadata