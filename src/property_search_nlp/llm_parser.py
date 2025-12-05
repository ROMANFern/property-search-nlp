"""
LLM-Based Property Query Parser using OpenAI GPT

Uses GPT-4 or GPT-3.5-Turbo to extract structured information 
from property search queries.

"""

import json
import time
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
import openai
import os

@dataclass
class PropertyQuery:
    """Structured representation of a property search query."""
    property_type: Optional[str] = None
    location: Optional[str] = None
    budget: Optional[float] = None
    budget_type: str = 'max'
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    parking: Optional[int] = None
    features: List[str] = None
    confidence_score: float = 0.0
    original_query: str = ""
    
    # LLM-specific metrics
    processing_time_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None

    def __post_init__(self):
        if self.features is None:
            self.features = []

    def to_dict(self) -> Dict:
        return asdict(self)


class LLMPropertyParser:
    """
    Property query parser using OpenAI GPT models.
    
    Supports:
    - GPT-4 (more accurate, more expensive)
    - GPT-3.5-Turbo (faster, cheaper)
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """
        Initialize LLM parser.
        
        Args:
            model: "gpt-4", "gpt-4-turbo", or "gpt-3.5-turbo"
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key")
        
        # Set the API key for openai library
        openai.api_key = self.api_key
        
        # Model pricing (per 1M tokens as of Dec 2024)
        self.pricing = {
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        }
    
    def _build_messages(self, query: str) -> List[Dict]:
        """Build the messages for GPT."""
        system_prompt = """You are a property search query parser. Extract structured information from queries and return ONLY a JSON object with no other text.

Return format:
{
    "property_type": "House|Apartment|Unit|Townhouse|Villa|Studio|Penthouse|etc or null",
    "location": "Suburb name or null",
    "budget": numeric value or null,
    "budget_type": "max|min|around",
    "bedrooms": number or null,
    "bathrooms": number or null,
    "parking": number or null,
    "features": ["Pool", "Garden", "Modern", etc] or []
}

Rules:
- Convert all prices to numeric (e.g., "800k" → 800000, "1.2m" → 1200000)
- Capitalize property_type (e.g., "house" → "House")
- Capitalize location properly (e.g., "richmond" → "Richmond", "melbourne CBD" → "Melbourne CBD")
- budget_type: "max" for "under", "min" for "over", "around" for "approximately"
- Return ONLY the JSON object, no explanations."""

        user_prompt = f'Query: "{query}"'
        
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    
    def parse(self, query: str) -> PropertyQuery:
        """
        Parse a property search query using GPT.
        
        Args:
            query: Natural language search query
            
        Returns:
            PropertyQuery object with extracted information
        """
        start_time = time.time()
        
        try:
            # Call OpenAI API
            messages = self._build_messages(query)
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,  # Deterministic output
                max_tokens=300,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            # Extract response
            response_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            parsed_data = json.loads(response_text)
            
            # Calculate metrics
            processing_time = (time.time() - start_time) * 1000  # ms
            tokens_used = response.usage.total_tokens
            cost = self._calculate_cost(
                response.usage.prompt_tokens, 
                response.usage.completion_tokens
            )
            
            # Create PropertyQuery object
            result = PropertyQuery(
                property_type=parsed_data.get('property_type'),
                location=parsed_data.get('location'),
                budget=parsed_data.get('budget'),
                budget_type=parsed_data.get('budget_type', 'max'),
                bedrooms=parsed_data.get('bedrooms'),
                bathrooms=parsed_data.get('bathrooms'),
                parking=parsed_data.get('parking'),
                features=parsed_data.get('features', []),
                original_query=query,
                processing_time_ms=processing_time,
                tokens_used=tokens_used,
                cost_usd=cost
            )
            
            # Calculate confidence based on completeness
            result.confidence_score = self._calculate_confidence(result)
            
            return result
            
        except Exception as e:
            print(f"Error parsing query with LLM: {e}")
            # Return empty result with error info
            return PropertyQuery(
                original_query=query,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate cost in USD based on token usage.
        
        Pricing per 1M tokens (Dec 2024):
        GPT-3.5-Turbo: $0.50 input, $1.50 output
        GPT-4-Turbo: $10 input, $30 output
        GPT-4: $30 input, $60 output
        """
        pricing = self.pricing.get(self.model, self.pricing["gpt-3.5-turbo"])
        
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def _calculate_confidence(self, result: PropertyQuery) -> float:
        """Calculate confidence score based on extracted fields."""
        score = 0.0
        if result.property_type: score += 20
        if result.location: score += 25
        if result.budget: score += 25
        if result.bedrooms: score += 15
        if result.bathrooms: score += 5
        if result.parking: score += 5
        if result.features: score += 5
        return min(score, 100.0)


def demo_llm_parser():
    """Demonstrate LLM-based parser."""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not set!")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nRunning in demo mode with sample output...\n")
        demo_mode = True
    else:
        demo_mode = False
        print("Choose model:")
        print("1. GPT-3.5-Turbo (faster, cheaper: ~$0.0001/query)")
        print("2. GPT-4-Turbo (more accurate: ~$0.002/query)")
        choice = input("Enter 1 or 2 [default: 1]: ").strip() or "1"
        
        model = "gpt-3.5-turbo" if choice == "1" else "gpt-4-turbo"
        parser = LLMPropertyParser(model=model)
    
    test_queries = [
        "3 bedroom house in Richmond under 800k",
        "apartment with 2 bedrooms near Melbourne CBD max $500k",
        "modern unit in South Yarra with parking",
        "luxury penthouse over 2 million",
    ]
    
    print("="*80)
    print(f"LLM-BASED PROPERTY QUERY PARSER - DEMONSTRATION")
    if not demo_mode:
        print(f"Model: {parser.model}")
    print("="*80)
    print()
    
    total_cost = 0.0
    total_time = 0.0
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-"*80)
        
        if demo_mode:
            print("  [Demo mode - would call OpenAI API here]")
            print("  Property Type: House")
            print("  Location: Richmond")
            print("  Budget: $800,000")
            print("  Bedrooms: 3")
            print("  Processing Time: ~100ms (GPT-3.5) or ~500ms (GPT-4)")
            print("  Cost: ~$0.0001 (GPT-3.5) or ~$0.002 (GPT-4)")
        else:
            result = parser.parse(query)
            
            print(f"  Property Type: {result.property_type or 'N/A'}")
            print(f"  Location: {result.location or 'N/A'}")
            print(f"  Budget: ${result.budget:,.0f}" if result.budget else "  Budget: N/A")
            print(f"  Bedrooms: {result.bedrooms or 'N/A'}")
            print(f"  Bathrooms: {result.bathrooms or 'N/A'}")
            print(f"  Parking: {result.parking or 'N/A'}")
            print(f"  Features: {', '.join(result.features) if result.features else 'N/A'}")
            print(f"  Confidence: {result.confidence_score:.0f}%")
            print(f"  Processing Time: {result.processing_time_ms:.1f}ms")
            print(f"  Tokens Used: {result.tokens_used}")
            print(f"  Cost: ${result.cost_usd:.6f}")
            
            total_cost += result.cost_usd
            total_time += result.processing_time_ms
        
        print()
    
    if not demo_mode:
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Model: {parser.model}")
        print(f"Total queries: {len(test_queries)}")
        print(f"Total processing time: {total_time:.1f}ms")
        print(f"Average time per query: {total_time/len(test_queries):.1f}ms")
        print(f"Total cost: ${total_cost:.6f}")
        print(f"Average cost per query: ${total_cost/len(test_queries):.6f}")
        print(f"Cost for 1,000 queries: ${(total_cost/len(test_queries)) * 1_000:.2f}")
        print(f"Cost for 1M queries: ${(total_cost/len(test_queries)) * 1_000_000:.2f}")


if __name__ == "__main__":
    demo_llm_parser()