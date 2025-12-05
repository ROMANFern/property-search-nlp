"""
Property Search Query Parser
A Natural Language Processing system for extracting structured information
from property search queries.

Author: Manusha Fernando
Date: December 2025
Purpose: Portfolio project demonstrating NLP skills
"""

import re
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class PropertyQuery:
    """Structured representation of a property search query."""
    property_type: Optional[str] = None
    location: Optional[str] = None
    budget: Optional[float] = None
    budget_type: str = 'max'  # 'min', 'max', 'around'
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    parking: Optional[int] = None
    features: List[str] = None
    confidence_score: float = 0.0
    original_query: str = ""

    def __post_init__(self):
        if self.features is None:
            self.features = []

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class PropertySearchParser:
    """
    NLP-based parser for property search queries.
    
    Extracts key information including:
    - Property type (house, apartment, etc.)
    - Location/suburb
    - Budget/price range
    - Number of bedrooms, bathrooms, parking
    - Additional features (pool, garden, etc.)
    """

    def __init__(self):
        self.property_types = [
            'house', 'apartment', 'unit', 'townhouse', 'villa', 
            'studio', 'duplex', 'penthouse', 'home', 'flat',
            'cottage', 'mansion', 'bungalow'
        ]
        
        self.feature_keywords = {
            'pool': 'Pool',
            'swimming pool': 'Swimming Pool',
            'garden': 'Garden',
            'balcony': 'Balcony',
            'terrace': 'Terrace',
            'modern': 'Modern',
            'renovated': 'Renovated',
            'new': 'New Build',
            'luxury': 'Luxury',
            'investment': 'Investment Property',
            'family': 'Family Home',
            'furnished': 'Furnished',
            'air conditioning': 'Air Conditioning',
            'heating': 'Heating',
            'gym': 'Gym',
            'security': 'Security'
        }

    def parse(self, query: str) -> PropertyQuery:
        """
        Parse a natural language property search query.
        
        Args:
            query: Natural language search query
            
        Returns:
            PropertyQuery object with extracted information
        """
        result = PropertyQuery(original_query=query)
        
        # Extract each component
        result.property_type = self._extract_property_type(query)
        result.location = self._extract_location(query)
        result.budget, result.budget_type = self._extract_budget(query)
        result.bedrooms = self._extract_bedrooms(query)
        result.bathrooms = self._extract_bathrooms(query)
        result.parking = self._extract_parking(query)
        result.features = self._extract_features(query)
        
        # Calculate confidence score
        result.confidence_score = self._calculate_confidence(result)
        
        return result

    def _extract_property_type(self, query: str) -> Optional[str]:
        """Extract property type from query."""
        query_lower = query.lower()
        for prop_type in self.property_types:
            if re.search(r'\b' + prop_type + r's?\b', query_lower):
                return prop_type.capitalize()
        return None

    def _extract_location(self, query: str) -> Optional[str]:
        """Extract location/suburb from query."""
        # Common words to stop location extraction
        stop_words = ['with', 'and', 'or', 'under', 'over', 'around', 'near', 'the', 'a', 'an']
        
        # Patterns to match location mentions - case insensitive
        patterns = [
            r'\bin\s+([A-Z][a-zA-Z]+(?:\s+[A-Z]+)?)',
            r'\bnear\s+([A-Z][a-zA-Z]+(?:\s+[A-Z]+)?)',
            r'\bat\s+([A-Z][a-zA-Z]+(?:\s+[A-Z]+)?)',
            r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z]+)?)\s+area\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                
                # Split into words and filter stop words
                words = location.split()
                filtered_words = []
                
                for word in words:
                    if word.lower() in stop_words:
                        break  # Stop at first stop word
                    filtered_words.append(word)
                
                if not filtered_words:
                    continue
                
                # Reconstruct location with proper capitalization
                result_words = []
                for word in filtered_words:
                    # Keep acronyms uppercase (2-4 letter all-caps words)
                    if word.isupper() and 2 <= len(word) <= 4:
                        result_words.append(word)
                    else:
                        result_words.append(word.capitalize())
                
                return ' '.join(result_words)
        
        return None

    def _extract_budget(self, query: str) -> Tuple[Optional[float], str]:
        """Extract budget/price from query."""
        query_lower = query.lower()
        
        # Determine budget type (min, max, around)
        budget_type = 'max'
        if any(word in query_lower for word in ['over', 'above', 'more than']):
            budget_type = 'min'
        elif any(word in query_lower for word in ['around', 'approximately', 'about']):
            budget_type = 'around'
        
        # Price extraction patterns
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*(?:million|mil|m)\b', 1000000),
            (r'\$?\s*(\d+(?:,\d{3})+)(?:k)?\b', 1),
            (r'\$?\s*(\d{5,})\b', 1),
            (r'(\d+)k\b', 1000),
        ]
        
        for pattern, multiplier in patterns:
            match = re.search(pattern, query_lower)
            if match:
                amount = float(match.group(1).replace(',', ''))
                return amount * multiplier, budget_type
        
        return None, budget_type

    def _extract_bedrooms(self, query: str) -> Optional[int]:
        """Extract number of bedrooms."""
        patterns = [
            r'(\d+)\s*(?:bed(?:room)?s?)\b',
            r'(\d+)\s*br\b',
            r'(\d+)\s*b\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        return None

    def _extract_bathrooms(self, query: str) -> Optional[int]:
        """Extract number of bathrooms."""
        patterns = [
            r'(\d+)\s*(?:bath(?:room)?s?)\b',
            r'(\d+)\s*ba\b'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return int(match.group(1))
        return None

    def _extract_parking(self, query: str) -> Optional[int]:
        """Extract parking/garage spaces."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['garage', 'parking', 'carport', 'car space']):
            # Try to find a number before the parking keyword
            match = re.search(r'(\d+)\s*(?:car|garage|parking)', query_lower)
            if match:
                return int(match.group(1))
            return 1  # If mentioned but no number, assume 1
        return None

    def _extract_features(self, query: str) -> List[str]:
        """Extract additional property features."""
        query_lower = query.lower()
        features = []
        
        for keyword, feature in self.feature_keywords.items():
            if keyword in query_lower:
                features.append(feature)
        
        return features

    def _calculate_confidence(self, result: PropertyQuery) -> float:
        """Calculate confidence score based on extracted information."""
        score = 0.0
        
        if result.property_type:
            score += 20
        if result.location:
            score += 25
        if result.budget:
            score += 25
        if result.bedrooms:
            score += 15
        if result.bathrooms:
            score += 5
        if result.parking:
            score += 5
        if result.features:
            score += 5
        
        return min(score, 100.0)


def demo_parser():
    """Demonstrate the parser with sample queries."""
    parser = PropertySearchParser()
    
    sample_queries = [
        "3 bedroom house in Richmond under 800k",
        "apartment with 2 bedrooms near Melbourne CBD max $500k",
        "family home in Hawthorn with garage and pool budget 1.2 million",
        "modern unit in South Yarra under 600000 with parking",
        "house for sale in Kew 4 beds 2 baths around $1.5m",
        "investment property in Brunswick under 700k",
        "luxury penthouse Melbourne CBD over 2 million",
        "townhouse in Camberwell with 3 bedrooms under $900k"
    ]
    
    print("=" * 80)
    print("PROPERTY SEARCH QUERY PARSER - DEMONSTRATION")
    print("=" * 80)
    print()
    
    results = []
    for query in sample_queries:
        print(f"Query: {query}")
        print("-" * 80)
        
        result = parser.parse(query)
        results.append(result.to_dict())
        
        print(f"Property Type: {result.property_type}")
        print(f"Location: {result.location}")
        print(f"Budget: {f'${result.budget:,.0f}' if result.budget else 'N/A'} ({result.budget_type})")
        print(f"Bedrooms: {result.bedrooms if result.bedrooms else 'N/A'}")
        print(f"Bathrooms: {result.bathrooms if result.bathrooms else 'N/A'}")
        print(f"Parking: {result.parking if result.parking else 'N/A'}")
        print(f"Features: {', '.join(result.features) if result.features else 'N/A'}")
        print(f"Confidence: {result.confidence_score:.0f}%")
        print()
    
    # Create DataFrame for analysis
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("EXTRACTION STATISTICS")
    print("=" * 80)
    print(f"Total queries processed: {len(sample_queries)}")
    print(f"Property type extracted: {df['property_type'].notna().sum()} ({df['property_type'].notna().sum()/len(df)*100:.1f}%)")
    print(f"Location extracted: {df['location'].notna().sum()} ({df['location'].notna().sum()/len(df)*100:.1f}%)")
    print(f"Budget extracted: {df['budget'].notna().sum()} ({df['budget'].notna().sum()/len(df)*100:.1f}%)")
    print(f"Bedrooms extracted: {df['bedrooms'].notna().sum()} ({df['bedrooms'].notna().sum()/len(df)*100:.1f}%)")
    print(f"Average confidence: {df['confidence_score'].mean():.1f}%")


if __name__ == "__main__":
    demo_parser()