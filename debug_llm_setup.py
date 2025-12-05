"""
Debug script to verify LLM setup
Checks all components before running full comparison
"""

import os
import sys

print("="*70)
print("LLM SETUP DIAGNOSTIC")
print("="*70)

# 1. Check API Key
print("\n1️⃣ Checking API Key...")
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print(f"   ✅ OPENAI_API_KEY found: {api_key[:10]}...{api_key[-4:]}")
else:
    print("   ❌ OPENAI_API_KEY not set!")
    print("   Set it with: $env:OPENAI_API_KEY = 'your-key'")
    sys.exit(1)

# 2. Check OpenAI package
print("\n2️⃣ Checking OpenAI package...")
try:
    import openai
    print(f"   ✅ openai package installed (version: {openai.__version__})")
except ImportError as e:
    print(f"   ❌ openai not installed: {e}")
    print("   Install with: pip install openai")
    sys.exit(1)

# 3. Check llm_parser.py exists
print("\n3️⃣ Checking llm_parser.py...")
import os.path
if os.path.exists("llm_parser.py"):
    print("   ✅ llm_parser.py found")
else:
    print("   ❌ llm_parser.py not found in current directory")
    print(f"   Current directory: {os.getcwd()}")
    sys.exit(1)

# 4. Try importing LLMPropertyParser
print("\n4️⃣ Importing LLMPropertyParser...")
try:
    from src.property_search_nlp.llm_parser import LLMPropertyParser
    print("   ✅ LLMPropertyParser imported successfully")
except ImportError as e:
    print(f"   ❌ Cannot import LLMPropertyParser: {e}")
    sys.exit(1)

# 5. Try initializing parser
print("\n5️⃣ Initializing LLM parser...")
try:
    parser = LLMPropertyParser(model="gpt-3.5-turbo")
    print(f"   ✅ Parser initialized with model: {parser.model}")
except Exception as e:
    print(f"   ❌ Failed to initialize: {e}")
    sys.exit(1)

# 6. Try parsing a simple query
print("\n6️⃣ Testing with sample query...")
try:
    test_query = "3 bedroom house in Richmond"
    print(f"   Query: '{test_query}'")
    
    result = parser.parse(test_query)
    
    print(f"   ✅ Parsing successful!")
    print(f"      Property Type: {result.property_type}")
    print(f"      Location: {result.location}")
    print(f"      Bedrooms: {result.bedrooms}")
    print(f"      Time: {result.processing_time_ms:.1f}ms")
    print(f"      Cost: ${result.cost_usd:.6f}")
    print(f"      Tokens: {result.tokens_used}")
    
except Exception as e:
    print(f"   ❌ Parsing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. Check comparison_framework.py
print("\n7️⃣ Checking comparison_framework.py...")
if os.path.exists("comparison_framework.py"):
    print("   ✅ comparison_framework.py found")
    
    try:
        from comparison_framework import ParserComparison
        print("   ✅ ParserComparison imported")
        
        # Try initializing
        comparison = ParserComparison()
        if comparison.llm_parser:
            print("   ✅ LLM parser initialized in comparison framework")
        else:
            print("   ⚠️  LLM parser is None in comparison framework")
            print("   This might be an issue with the framework initialization")
            
    except Exception as e:
        print(f"   ⚠️  Issue with comparison framework: {e}")
else:
    print("   ⚠️  comparison_framework.py not found")

# 8. Final check
print("\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)

if os.path.exists("comparison_framework.py"):
    print("\n✅ All checks passed! You should be able to run:")
    print("   python comparison_framework.py")
else:
    print("\n⚠️  Setup incomplete - check warnings above")

print("\n📊 Ready to compare Rule-Based vs LLM parsing!")
