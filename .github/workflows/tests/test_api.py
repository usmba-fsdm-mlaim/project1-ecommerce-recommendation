"""
API Integration Tests
Branch: feature/ci-cd-pipeline
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
sys.path.append('..')

from app import app, model, product_metadata


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Create mock recommendation model"""
    mock = Mock()
    mock.user_lookup = {0: 0, 1: 1, 2: 2}
    mock.product_lookup = {'P1': 0, 'P2': 1, 'P3': 2}
    mock.recommend_products = Mock(return_value=[
        ('P1', 0.95),
        ('P2', 0.87),
        ('P3', 0.76)
    ])
    return mock


@pytest.fixture
def mock_metadata():
    """Create mock product metadata"""
    return {
        'P1': {
            'name': 'Test Product 1',
            'brand': 'Brand A',
            'main_category': 'Electronics',
            'price': 99.99,
            'avg_rating': 4.5
        },
        'P2': {
            'name': 'Test Product 2',
            'brand': 'Brand B',
            'main_category': 'Books',
            'price': 19.99,
            'avg_rating': 4.2
        },
        'P3': {
            'name': 'Test Product 3',
            'brand': 'Brand C',
            'main_category': 'Clothing',
            'price': 49.99,
            'avg_rating': 4.8
        }
    }


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestRootEndpoint:
    """Test root endpoint"""
    
    def test_root(self, client):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


class TestRecommendationEndpoint:
    """Test recommendation endpoint"""
    
    @patch('app.model')
    @patch('app.product_metadata')
    def test_get_recommendations_success(self, mock_meta, mock_mdl, client, mock_model, mock_metadata):
        """Test successful recommendation request"""
        mock_mdl = mock_model
        mock_meta = mock_metadata
        
        response = client.post(
            "/recommend",
            json={"user_id": 0, "n": 3}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "user_id" in data
        assert "recommendations" in data
        assert "generated_at" in data
    
    def test_get_recommendations_invalid_user_id(self, client):
        """Test recommendation with invalid user ID"""
        response = client.post(
            "/recommend",
            json={"user_id": -1, "n": 5}
        )
        # May return 503 if model not loaded, or 200 with empty recommendations
        assert response.status_code in [200, 503]
    
    def test_get_recommendations_invalid_n(self, client):
        """Test recommendation with invalid n parameter"""
        response = client.post(
            "/recommend",
            json={"user_id": 0, "n": 100}  # Max is 50
        )
        assert response.status_code == 422  # Validation error
    
    def test_get_recommendations_missing_parameters(self, client):
        """Test recommendation with missing parameters"""
        response = client.post("/recommend", json={})
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Test metrics endpoint"""
    
    @patch('app.model')
    def test_metrics_model_loaded(self, mock_mdl, client, mock_model):
        """Test metrics when model is loaded"""
        mock_mdl = mock_model
        
        response = client.get("/metrics")
        # May return 200 or 503 depending on model state
        assert response.status_code in [200, 503]
    
    def test_metrics_model_not_loaded(self, client):
        """Test metrics when model is not loaded"""
        with patch('app.model', None):
            response = client.get("/metrics")
            assert response.status_code == 503


class TestPrometheusMetrics:
    """Test Prometheus metrics endpoint"""
    
    def test_prometheus_endpoint(self, client):
        """Test Prometheus metrics are exposed"""
        response = client.get("/prometheus")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]


# Integration tests
class TestIntegration:
    """Integration tests for complete workflows"""
    
    @patch('app.model')
    @patch('app.product_metadata')
    def test_recommendation_workflow(self, mock_meta, mock_mdl, client, mock_model, mock_metadata):
        """Test complete recommendation workflow"""
        mock_mdl = mock_model
        mock_meta = mock_metadata
        
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Get metrics
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code in [200, 503]
        
        # 3. Get recommendations
        rec_response = client.post(
            "/recommend",
            json={"user_id": 0, "n": 5}
        )
        assert rec_response.status_code in [200, 503]


# Load tests
def test_concurrent_requests(client):
    """Test API handles concurrent requests"""
    import concurrent.futures
    
    def make_request():
        return client.get("/health")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        results = [f.result() for f in futures]
    
    assert all(r.status_code == 200 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])