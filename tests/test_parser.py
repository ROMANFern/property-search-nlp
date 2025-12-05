"""
Unit tests for Property Search Query Parser
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.property_search_nlp.parser import PropertySearchParser, PropertyQuery

class TestPropertySearchParser:
    """Test suite for PropertySearchParser"""

    @pytest.fixture
    def parser(self):
        """Create parser instance for tests"""
        return PropertySearchParser()

    def test_extract_property_type_house(self, parser):
        """Test extraction of 'house' property type"""
        result = parser.parse("3 bedroom house in Richmond")
        assert result.property_type == "House"

    def test_extract_property_type_apartment(self, parser):
        """Test extraction of 'apartment' property type"""
        result = parser.parse("apartment near CBD")
        assert result.property_type == "Apartment"

    def test_extract_property_type_unit(self, parser):
        """Test extraction of 'unit' property type"""
        result = parser.parse("modern unit in South Yarra")
        assert result.property_type == "Unit"

    def test_extract_location_single_word(self, parser):
        """Test location extraction with single word suburb"""
        result = parser.parse("house in Richmond")
        assert result.location == "Richmond"

    def test_extract_location_multiple_words(self, parser):
        """Test location extraction with multi-word location"""
        result = parser.parse("apartment near Melbourne CBD")
        assert result.location == "Melbourne CBD"

    def test_extract_location_various_prepositions(self, parser):
        """Test location extraction with different prepositions"""
        queries = [
            "house in Richmond",
            "apartment near Richmond",
            "unit at Richmond"
        ]
        for query in queries:
            result = parser.parse(query)
            assert result.location == "Richmond"

    def test_extract_budget_thousands(self, parser):
        """Test budget extraction in thousands (k format)"""
        result = parser.parse("house under 800k")
        assert result.budget == 800000

    def test_extract_budget_millions(self, parser):
        """Test budget extraction in millions"""
        result = parser.parse("property around 1.5 million")
        assert result.budget == 1500000

    def test_extract_budget_full_number(self, parser):
        """Test budget extraction with full number"""
        result = parser.parse("apartment max $650,000")
        assert result.budget == 650000

    def test_budget_type_max(self, parser):
        """Test budget type detection for maximum"""
        result = parser.parse("house under 800k")
        assert result.budget_type == "max"

    def test_budget_type_min(self, parser):
        """Test budget type detection for minimum"""
        result = parser.parse("property over 2 million")
        assert result.budget_type == "min"

    def test_budget_type_around(self, parser):
        """Test budget type detection for approximate"""
        result = parser.parse("house around 1.2 million")
        assert result.budget_type == "around"

    def test_extract_bedrooms_full_word(self, parser):
        """Test bedroom extraction with full word"""
        result = parser.parse("3 bedroom house")
        assert result.bedrooms == 3

    def test_extract_bedrooms_abbreviation(self, parser):
        """Test bedroom extraction with abbreviation"""
        result = parser.parse("4 bed apartment")
        assert result.bedrooms == 4

    def test_extract_bathrooms(self, parser):
        """Test bathroom extraction"""
        result = parser.parse("house with 2 bathrooms")
        assert result.bathrooms == 2

    def test_extract_parking_with_number(self, parser):
        """Test parking extraction with explicit number"""
        result = parser.parse("apartment with 2 car garage")
        assert result.parking == 2

    def test_extract_parking_implicit(self, parser):
        """Test parking extraction when mentioned without number"""
        result = parser.parse("house with garage")
        assert result.parking == 1

    def test_extract_features_single(self, parser):
        """Test extraction of single feature"""
        result = parser.parse("house with pool")
        assert "Pool" in result.features

    def test_extract_features_multiple(self, parser):
        """Test extraction of multiple features"""
        result = parser.parse("modern house with pool and garden")
        assert "Modern" in result.features
        assert "Pool" in result.features
        assert "Garden" in result.features

    def test_confidence_score_high(self, parser):
        """Test confidence score with complete information"""
        result = parser.parse("3 bedroom house in Richmond under 800k with pool and garage")
        assert result.confidence_score >= 80

    def test_confidence_score_low(self, parser):
        """Test confidence score with minimal information"""
        result = parser.parse("property for sale")
        assert result.confidence_score < 50

    def test_complex_query_full_extraction(self, parser):
        """Test complete extraction from complex query"""
        query = "modern 4 bedroom house in Hawthorn with 2 bathrooms, pool and 2 car garage under 1.5 million"
        result = parser.parse(query)
        
        assert result.property_type == "House"
        assert result.location == "Hawthorn"
        assert result.budget == 1500000
        assert result.bedrooms == 4
        assert result.bathrooms == 2
        assert result.parking == 2
        assert "Pool" in result.features
        assert "Modern" in result.features

    def test_to_dict_conversion(self, parser):
        """Test conversion to dictionary"""
        result = parser.parse("3 bedroom house in Richmond")
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['property_type'] == 'House'
        assert result_dict['bedrooms'] == 3

    def test_to_json_conversion(self, parser):
        """Test conversion to JSON string"""
        result = parser.parse("apartment in CBD")
        json_str = result.to_json()
        
        assert isinstance(json_str, str)
        assert '"property_type"' in json_str

    def test_empty_query(self, parser):
        """Test parser behavior with empty query"""
        result = parser.parse("")
        assert result.confidence_score == 0

    def test_no_matches_query(self, parser):
        """Test parser with query containing no extractable info"""
        result = parser.parse("looking for something nice")
        assert result.property_type is None
        assert result.location is None
        assert result.budget is None

    def test_case_insensitivity(self, parser):
        """Test that parser is case insensitive"""
        queries = [
            "HOUSE in RICHMOND",
            "house in richmond",
            "House In Richmond"
        ]
        for query in queries:
            result = parser.parse(query)
            assert result.property_type == "House"
            assert result.location == "Richmond"

class TestPropertyQuery:
    """Test suite for PropertyQuery data model"""

    def test_property_query_initialization(self):
        """Test PropertyQuery initialization with defaults"""
        query = PropertyQuery()
        assert query.property_type is None
        assert query.features == []
        assert query.confidence_score == 0.0

    def test_property_query_with_values(self):
        """Test PropertyQuery initialization with values"""
        query = PropertyQuery(
            property_type="House",
            location="Richmond",
            bedrooms=3,
            budget=800000
        )
        assert query.property_type == "House"
        assert query.location == "Richmond"
        assert query.bedrooms == 3
        assert query.budget == 800000

    def test_property_query_features_list(self):
        """Test PropertyQuery with features list"""
        query = PropertyQuery(features=["Pool", "Garden"])
        assert len(query.features) == 2
        assert "Pool" in query.features

# Test fixtures and data
@pytest.fixture
def sample_queries():
    """Sample queries for batch testing"""
    return [
        "3 bedroom house in Richmond under 800k",
        "apartment with 2 bedrooms near Melbourne CBD max $500k",
        "family home in Hawthorn with garage and pool budget 1.2 million",
        "modern unit in South Yarra under 600000 with parking",
        "house for sale in Kew 4 beds 2 baths around $1.5m",
    ]


def test_batch_processing(sample_queries):
    """Test processing multiple queries"""
    parser = PropertySearchParser()
    results = [parser.parse(q) for q in sample_queries]
    
    assert len(results) == len(sample_queries)
    assert all(r.confidence_score > 0 for r in results)


# Performance tests
def test_parser_performance(sample_queries):
    """Test parser performance with multiple queries"""
    import time
    
    parser = PropertySearchParser()
    start_time = time.time()
    
    for _ in range(100):
        for query in sample_queries:
            parser.parse(query)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Should process 500 queries in reasonable time (< 5 seconds)
    assert elapsed < 5.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])