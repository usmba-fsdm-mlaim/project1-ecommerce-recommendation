"""
Unit tests for recommendation model
Branch: feature/ml-model
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from recommendation_model import CollaborativeFilteringModel


@pytest.fixture
def sample_interaction_data():
    """Create sample interaction data"""
    data = {
        'user_id': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3],
        'product_id': ['P1', 'P2', 'P3', 'P1', 'P2', 'P4', 'P2', 'P3', 'P4', 'P1', 'P3'],
        'rating': [5.0, 4.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0]
    }
    return pd.DataFrame(data)


@pytest.fixture
def model():
    """Create a model instance"""
    return CollaborativeFilteringModel(n_recommendations=5, min_interactions=1)


class TestCollaborativeFilteringModel:
    """Test CollaborativeFilteringModel class"""
    
    def test_model_initialization(self, model):
        """Test model initialization"""
        assert model.n_recommendations == 5
        assert model.min_interactions == 1
        assert model.user_item_matrix is None
    
    def test_create_interaction_matrix(self, model, sample_interaction_data):
        """Test interaction matrix creation"""
        matrix = model.create_interaction_matrix(sample_interaction_data)
        
        assert model.user_item_matrix is not None
        assert len(model.user_lookup) > 0
        assert len(model.product_lookup) > 0
        assert model.global_mean > 0
    
    def test_compute_user_similarity(self, model, sample_interaction_data):
        """Test user similarity computation"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_user_similarity()
        
        assert model.user_similarity is not None
        assert model.user_similarity.shape[0] == len(model.user_lookup)
    
    def test_compute_item_similarity(self, model, sample_interaction_data):
        """Test item similarity computation"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_item_similarity()
        
        assert model.item_similarity is not None
        assert model.item_similarity.shape[0] == len(model.product_lookup)
    
    def test_predict_user_based(self, model, sample_interaction_data):
        """Test user-based predictions"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_user_similarity()
        
        user_idx = 0
        predictions = model.predict_user_based(user_idx)
        
        assert predictions is not None
        assert len(predictions) == len(model.product_lookup)
        assert isinstance(predictions, np.ndarray)
    
    def test_predict_item_based(self, model, sample_interaction_data):
        """Test item-based predictions"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_item_similarity()
        
        user_idx = 0
        predictions = model.predict_item_based(user_idx)
        
        assert predictions is not None
        assert len(predictions) == len(model.product_lookup)
    
    def test_predict_hybrid(self, model, sample_interaction_data):
        """Test hybrid predictions"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        user_idx = 0
        predictions = model.predict_hybrid(user_idx, alpha=0.5)
        
        assert predictions is not None
        assert len(predictions) == len(model.product_lookup)
    
    def test_recommend_products(self, model, sample_interaction_data):
        """Test product recommendations"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        user_id = 0
        recommendations = model.recommend_products(user_id, n=3)
        
        assert len(recommendations) <= 3
        assert all(isinstance(rec, tuple) for rec in recommendations)
        assert all(len(rec) == 2 for rec in recommendations)
    
    def test_recommend_cold_start(self, model, sample_interaction_data):
        """Test recommendations for new user (cold start)"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        # User not in training data
        user_id = 999
        recommendations = model.recommend_products(user_id, n=3)
        
        assert len(recommendations) > 0  # Should return popular items
    
    def test_evaluate(self, model, sample_interaction_data):
        """Test model evaluation"""
        # Split data
        train_data = sample_interaction_data.iloc[:8]
        test_data = sample_interaction_data.iloc[8:]
        
        model.create_interaction_matrix(train_data)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        metrics = model.evaluate(test_data)
        
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'coverage' in metrics
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert 0 <= metrics['coverage'] <= 1
    
    def test_save_and_load_model(self, model, sample_interaction_data):
        """Test model save and load"""
        # Train model
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model.save_model(f.name)
            
            # Load model
            loaded_model = CollaborativeFilteringModel.load_model(f.name)
            
            assert loaded_model.user_item_matrix is not None
            assert loaded_model.user_similarity is not None
            assert loaded_model.item_similarity is not None
            assert len(loaded_model.user_lookup) == len(model.user_lookup)
            assert len(loaded_model.product_lookup) == len(model.product_lookup)


class TestModelPerformance:
    """Test model performance characteristics"""
    
    def test_recommendations_not_already_rated(self, model, sample_interaction_data):
        """Test that recommendations exclude already rated items"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        user_id = 0
        recommendations = model.recommend_products(user_id, n=5)
        
        # Get user's rated items
        user_idx = model.user_lookup[user_id]
        rated_items = model.user_item_matrix.iloc[user_idx]
        rated_products = set(rated_items[rated_items > 0].index)
        
        # Check recommendations don't include rated items
        recommended_products = set(rec[0] for rec in recommendations)
        assert len(recommended_products.intersection(rated_products)) == 0
    
    def test_recommendation_scores_decreasing(self, model, sample_interaction_data):
        """Test that recommendation scores are in decreasing order"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        user_id = 0
        recommendations = model.recommend_products(user_id, n=5)
        
        scores = [score for _, score in recommendations]
        assert scores == sorted(scores, reverse=True)
    
    def test_different_users_different_recommendations(self, model, sample_interaction_data):
        """Test that different users get different recommendations"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        recs_user_0 = model.recommend_products(0, n=3)
        recs_user_1 = model.recommend_products(1, n=3)
        
        # Recommendations should be different (at least partially)
        prods_0 = set(rec[0] for rec in recs_user_0)
        prods_1 = set(rec[0] for rec in recs_user_1)
        
        # Allow some overlap but not identical
        assert prods_0 != prods_1 or len(prods_0) == 0


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_dataframe(self, model):
        """Test handling of empty dataframe"""
        empty_df = pd.DataFrame(columns=['user_id', 'product_id', 'rating'])
        
        with pytest.raises(Exception):
            model.create_interaction_matrix(empty_df)
    
    def test_single_user(self, model):
        """Test with single user"""
        data = pd.DataFrame({
            'user_id': [0, 0, 0],
            'product_id': ['P1', 'P2', 'P3'],
            'rating': [5.0, 4.0, 3.0]
        })
        
        model.create_interaction_matrix(data)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        # Should still work
        recommendations = model.recommend_products(0, n=2)
        assert isinstance(recommendations, list)
    
    def test_invalid_user_id(self, model, sample_interaction_data):
        """Test recommendations for invalid user ID"""
        model.create_interaction_matrix(sample_interaction_data)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        # Invalid user should get popular recommendations
        recommendations = model.recommend_products(999, n=3)
        assert len(recommendations) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])