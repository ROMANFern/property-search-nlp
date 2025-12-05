"""
Quick Setup and Test Script
Tests API connection and runs a simple comparison

"""
import os
import sys

def check_api_key():
    """Check if API key is set."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found!")
        print("\n📋 Setup Instructions:")
        print("1. Get your OpenAI API key from: https://platform.openai.com/api-keys")
        print("2. Set the environment variable:")
        print("\n   Windows PowerShell:")
        print('   $env:OPENAI_API_KEY = "your-key-here"')
        print("\n   Mac/Linux:")
        print('   export OPENAI_API_KEY="your-key-here"')
        print("\n3. Run this script again")
        return False
    
    print("✅ API key found!")
    print(f"   Key starts with: {api_key[:10]}...")
    return True


def check_dependencies():
    """Check if required packages are installed."""
    required = {
        'openai': 'openai',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing = []
    
    for package, import_name in required.items():
        try:
            __import__(import_name)
            print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True


def test_api_connection():
    """Test connection to OpenAI API."""
    print("\n🔗 Testing OpenAI API connection...")
    
    try:
        import openai
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Simple test call
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Say 'API connection successful!' and nothing else."}
            ],
            max_tokens=10
        )
        
        result = response.choices[0].message.content
        print(f"✅ API Response: {result}")
        print(f"   Tokens used: {response.usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False


def run_simple_test():
    """Run a simple parsing test."""
    print("\n🧪 Running simple parsing test...")
    
    try:
        from src.property_search_nlp.parser import PropertySearchParser
        import openai
        import json
        import time
        
        # Test query
        query = "3 bedroom house in Richmond under 800k"
        
        # Rule-based parser
        print(f"\n📝 Query: '{query}'")
        print("\n" + "="*60)
        print("RULE-BASED PARSER")
        print("="*60)
        
        rule_parser = PropertySearchParser()
        start = time.perf_counter()
        rule_result = rule_parser.parse(query)
        rule_time = (time.perf_counter() - start) * 1000
        
        print(f"Property Type: {rule_result.property_type}")
        print(f"Location:      {rule_result.location}")
        print(f"Budget:        ${rule_result.budget:,.0f}" if rule_result.budget else "Budget:        N/A")
        print(f"Bedrooms:      {rule_result.bedrooms}")
        print(f"Confidence:    {rule_result.confidence_score:.1f}%")
        print(f"Time:          {rule_time:.2f}ms")
        print(f"Cost:          $0.00 (FREE)")
        
        # LLM-based parser
        print("\n" + "="*60)
        print("LLM-BASED PARSER (GPT-3.5-Turbo)")
        print("="*60)
        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        system_prompt = """Extract structured information and return ONLY a JSON object.
        
Format: {"property_type": "House", "location": "Richmond", "budget": 800000, "bedrooms": 3}

Rules: Convert "800k" to 800000, capitalize properly."""

        start = time.perf_counter()
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Query: "{query}"'}
            ],
            temperature=0,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        llm_time = (time.perf_counter() - start) * 1000
        
        # Parse response
        llm_result = json.loads(response.choices[0].message.content)
        
        # Calculate cost (GPT-3.5-Turbo pricing)
        input_cost = (response.usage.prompt_tokens / 1_000_000) * 0.50
        output_cost = (response.usage.completion_tokens / 1_000_000) * 1.50
        total_cost = input_cost + output_cost
        
        print(f"Property Type: {llm_result.get('property_type', 'N/A')}")
        print(f"Location:      {llm_result.get('location', 'N/A')}")
        budget = llm_result.get('budget')
        print(f"Budget:        ${budget:,.0f}" if budget else "Budget:        N/A")
        print(f"Bedrooms:      {llm_result.get('bedrooms', 'N/A')}")
        print(f"Time:          {llm_time:.2f}ms")
        print(f"Tokens:        {response.usage.total_tokens}")
        print(f"Cost:          ${total_cost:.6f}")
        
        # Comparison
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Speed:  Rule-based is {llm_time/rule_time:.0f}x faster")
        print(f"Cost:   LLM costs ${total_cost:.6f} per query")
        print(f"        At 1K queries: ${total_cost * 1_000:.2f}")
        print(f"        At 10K queries: ${total_cost * 10_000:.2f}")
        print(f"        At 100K queries: ${total_cost * 100_000:.2f}")
        print(f"        At 1M queries: ${total_cost * 1_000_000:,.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main setup and test function."""
    print("="*60)
    print("PROPERTY QUERY PARSER - API SETUP & TEST (OpenAI)")
    print("="*60)
    
    # Check API key
    if not check_api_key():
        return
    
    print("\n" + "="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Test API connection
    if not test_api_connection():
        return
    
    # Run simple test
    if run_simple_test():
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        print("\n📚 Next Steps:")
        print("1. Run full comparison: python comparison_framework.py")
        print("2. Try GPT-4 for better accuracy: python llm_parser.py")
        print("3. Check the generated report: comparison_report.md")
        print("4. View visualizations: comparison_charts.png")
    else:
        print("\n❌ Setup incomplete - please fix errors above")


if __name__ == "__main__":
    main()
