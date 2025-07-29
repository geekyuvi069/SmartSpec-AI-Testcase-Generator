import re
from PyPDF2 import PdfReader
import json

def extract_text_from_pdf(file_path):
    """
    Enhanced text extraction from PDF with better formatting.
    """
    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Add page separator for better chunking
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page_text + "\n"
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    return text

def clean_and_structure_text(text):
    """
    Advanced text cleaning with structure preservation.
    """
    # Remove excessive whitespace but preserve paragraph structure
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Normalize spaces within lines
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Clean each line but preserve intentional formatting
        cleaned_line = re.sub(r'\s+', ' ', line.strip())
        if cleaned_line:  # Only add non-empty lines
            cleaned_lines.append(cleaned_line)
        elif cleaned_lines and cleaned_lines[-1]:  # Preserve paragraph breaks
            cleaned_lines.append('')
    
    return '\n'.join(cleaned_lines)

def extract_requirements_sections(text):
    """
    Extract structured sections from SRS document.
    """
    sections = {}
    
    # Common SRS section patterns
    section_patterns = [
        r'(\d+\.?\s*)(functional\s+requirements?)',
        r'(\d+\.?\s*)(non-functional\s+requirements?)',
        r'(\d+\.?\s*)(system\s+requirements?)',
        r'(\d+\.?\s*)(user\s+requirements?)',
        r'(\d+\.?\s*)(interface\s+requirements?)',
        r'(\d+\.?\s*)(performance\s+requirements?)',
        r'(\d+\.?\s*)(security\s+requirements?)',
    ]
    
    current_section = "general"
    sections[current_section] = []
    
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if this line starts a new section
        section_found = False
        for pattern in section_patterns:
            if re.search(pattern, line_lower):
                current_section = re.search(pattern, line_lower).group(2).replace(' ', '_')
                sections[current_section] = []
                section_found = True
                break
        
        if not section_found and line.strip():
            sections[current_section].append(line.strip())
    
    # Convert lists to strings
    for section in sections:
        sections[section] = '\n'.join(sections[section])
    
    return sections

def load_existing_test_cases(file_path):
    """
    Load existing test cases from JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Ensure the data has the expected structure
        if isinstance(data, list):
            return {"test_cases": data}
        elif isinstance(data, dict) and "test_cases" in data:
            return data
        else:
            # Try to convert to expected format
            return {"test_cases": [data] if isinstance(data, dict) else []}
            
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading test cases: {str(e)}")