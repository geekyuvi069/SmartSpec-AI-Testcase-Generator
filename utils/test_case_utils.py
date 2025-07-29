import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any
import re

class TestCaseManager:
    def __init__(self):
        self.existing_test_cases = []
        self.new_test_cases = []
        
    def load_existing_test_cases(self, test_cases_data):
        """Load existing test cases from uploaded JSON."""
        if isinstance(test_cases_data, dict) and "test_cases" in test_cases_data:
            self.existing_test_cases = test_cases_data["test_cases"]
        elif isinstance(test_cases_data, list):
            self.existing_test_cases = test_cases_data
        else:
            self.existing_test_cases = []
            
        # Ensure each test case has a unique ID
        for i, test_case in enumerate(self.existing_test_cases):
            if "id" not in test_case:
                test_case["id"] = self.generate_test_case_id(test_case)
                
    def generate_test_case_id(self, test_case):
        """Generate unique ID for test case based on content."""
        content = f"{test_case.get('title', '')}{test_case.get('description', '')}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
        
    def create_test_case(self, title, description, steps, expected_result, 
                        requirement_section="", priority="Medium", test_type="Functional"):
        """Create a standardized test case object."""
        test_case = {
            "id": "",
            "title": title,
            "description": description,
            "steps": steps if isinstance(steps, list) else [steps],
            "expected_result": expected_result,
            "requirement_section": requirement_section,
            "priority": priority,
            "test_type": test_type,
            "created_date": datetime.now().isoformat(),
            "status": "Draft"
        }
        
        test_case["id"] = self.generate_test_case_id(test_case)
        return test_case
        
    def is_duplicate(self, new_test_case, similarity_threshold=0.8):
        """Check if test case is duplicate using content similarity."""
        new_content = f"{new_test_case.get('title', '')} {new_test_case.get('description', '')}"
        
        for existing_case in self.existing_test_cases:
            existing_content = f"{existing_case.get('title', '')} {existing_case.get('description', '')}"
            
            # Simple similarity check based on common words
            similarity = self.calculate_similarity(new_content, existing_content)
            if similarity > similarity_threshold:
                return True, existing_case["id"]
                
        return False, None
        
    def calculate_similarity(self, text1, text2):
        """Calculate simple word-based similarity between two texts."""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
        
    def add_test_case(self, test_case):
        """Add test case if not duplicate."""
        is_dup, existing_id = self.is_duplicate(test_case)
        
        if not is_dup:
            self.new_test_cases.append(test_case)
            return True, test_case["id"]
        else:
            return False, existing_id
            
    def merge_test_cases(self):
        """Merge existing and new test cases."""
        all_test_cases = self.existing_test_cases + self.new_test_cases
        
        # Sort by priority and creation date
        priority_order = {"High": 1, "Medium": 2, "Low": 3}
        
        all_test_cases.sort(key=lambda x: (
            priority_order.get(x.get("priority", "Medium"), 2),
            x.get("created_date", "")
        ))
        
        return {
            "metadata": {
                "total_test_cases": len(all_test_cases),
                "existing_count": len(self.existing_test_cases),
                "new_count": len(self.new_test_cases),
                "generated_date": datetime.now().isoformat(),
                "version": "1.0"
            },
            "test_cases": all_test_cases
        }
        
    def save_merged_test_cases(self, output_path):
        """Save merged test cases to JSON file."""
        merged_data = self.merge_test_cases()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
            
        return merged_data
        
    def generate_test_cases_from_chunk(self, chunk_text, requirement_section=""):
        """Generate test cases from a text chunk using rule-based approach."""
        test_cases = []
        
        # Extract potential test scenarios
        scenarios = self.extract_test_scenarios(chunk_text)
        
        for i, scenario in enumerate(scenarios, 1):
            test_case = self.create_test_case(
                title=f"Test Case for {requirement_section or 'Requirement'} - Scenario {i}",
                description=f"Verify {scenario['description']}",
                steps=scenario['steps'],
                expected_result=scenario['expected'],
                requirement_section=requirement_section,
                priority=scenario.get('priority', 'Medium'),
                test_type=scenario.get('type', 'Functional')
            )
            test_cases.append(test_case)
            
        return test_cases
        
    def extract_test_scenarios(self, text):
        """Extract test scenarios from requirement text."""
        scenarios = []
        
        # Look for common requirement patterns
        patterns = [
            r'the system (shall|must|should) (.+?)(?:\.|$)',
            r'user (can|may|shall) (.+?)(?:\.|$)',
            r'when (.+?), the system (.+?)(?:\.|$)',
            r'if (.+?), then (.+?)(?:\.|$)',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) >= 2:
                    scenario = self.create_scenario_from_match(match, text)
                    if scenario:
                        scenarios.append(scenario)
        
        # If no patterns found, create a general scenario
        if not scenarios:
            scenarios.append({
                'description': 'the functionality described in the requirement',
                'steps': [
                    'Navigate to the relevant system component',
                    'Execute the required functionality',
                    'Verify the system behavior'
                ],
                'expected': 'System behaves as specified in the requirement',
                'priority': 'Medium',
                'type': 'Functional'
            })
            
        return scenarios[:3]  # Limit to 3 scenarios per chunk
        
    def create_scenario_from_match(self, match, context):
        """Create test scenario from regex match."""
        groups = match.groups()
        
        if 'login' in context.lower() or 'authentication' in context.lower():
            return {
                'description': 'user authentication functionality',
                'steps': [
                    'Navigate to login page',
                    'Enter valid credentials',
                    'Click login button',
                    'Verify successful authentication'
                ],
                'expected': 'User is successfully authenticated and redirected to dashboard',
                'priority': 'High',
                'type': 'Security'
            }
        elif 'validation' in context.lower() or 'validate' in context.lower():
            return {
                'description': 'input validation functionality',
                'steps': [
                    'Enter invalid input data',
                    'Submit the form',
                    'Verify validation message appears'
                ],
                'expected': 'System displays appropriate validation error message',
                'priority': 'Medium',
                'type': 'Validation'
            }
        else:
            return {
                'description': f'functionality: {groups[-1] if groups else "system behavior"}',
                'steps': [
                    'Prepare test environment',
                    'Execute the specified functionality',
                    'Verify system response'
                ],
                'expected': 'System behaves according to the requirement specification',
                'priority': 'Medium',
                'type': 'Functional'
            }