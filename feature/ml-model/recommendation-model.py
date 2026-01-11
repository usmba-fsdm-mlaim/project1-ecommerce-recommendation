"""
Recommendation Model Training with MLflow Tracking
Branch: feature/ml-model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix
import mlflow
import mlflow.sklearn
from typing import List, Tuple, Dict
import pickle
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CollaborativeFilteringModel:
    """
    Hybrid Collaborative Filtering Recommendation System
    Combines User-Based and Item-Based Collaborative Filtering
    """
    
    def __init__(self, n_recommendations=10, min_interactions=2):
        self.n_recommendations = n_recommendations
        self.min_interactions = min_interactions
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.user_mean_ratings = None
        self.global_mean = None
        self.product_lookup = {}
        self.user_lookup = {}
        
    def create_interaction_matrix(self, df: pd.DataFrame) -> csr_matrix:
        """Create user-item interaction matrix"""
        logger.info("Creating interaction matrix...")
        
        # Filter users and products with minimum interactions
        user_counts = df['user_id'].value_counts()
        product_counts = df['product_id'].value_counts()
        
        valid_users = user_counts[user_counts >= self.min_interactions].index
        valid_products = product_counts[product_counts >= self.min_interactions].index
        
        df_filtered = df[
            (df['user_id'].isin(valid_users)) & 
            (df['product_id'].isin(valid_products))
        ]
        
        logger.info(f"Filtered to {len(df_filtered)} interactions")
        logger.info(f"Users: {len(valid_users)}, Products: {len(valid_products)}")
        
        # Create pivot table
        self.user_item_matrix = df_filtered.pivot_table(
            index='user_id',
            columns='product_id',
            values='rating',
            fill_value=0
        )
        
        # Store lookups
        self.user_lookup = {user: idx for idx, user in enumerate(self.user_item_matrix.index)}
        self.product_lookup = {prod: idx for idx, prod in enumerate(self.user_item_matrix.columns)}
        
        # Store metadata
        self.user_mean_ratings = df_filtered.groupby('user_id')['rating'].mean().to_dict()
        self.global_mean = df_filtered['rating'].mean()
        
        return csr_matrix(self.user_item_matrix.values)
    
    def compute_user_similarity(self):
        """Compute user-user similarity matrix"""
        logger.info("Computing user similarity...")
        self.user_similarity = cosine_similarity(self.user_item_matrix, dense_output=False)
        return self
    
    def compute_item_similarity(self):
        """Compute item-item similarity matrix"""
        logger.info("Computing item similarity...")
        self.item_similarity = cosine_similarity(self.user_item_matrix.T, dense_output=False)
        return self
    
    def predict_user_based(self, user_idx: int, top_k: int = 50) -> np.ndarray:
        """Predict ratings using user-based collaborative filtering"""
        if user_idx >= len(self.user_similarity):
            return np.zeros(self.user_item_matrix.shape[1])
        
        # Get similar users
        user_sim = self.user_similarity[user_idx].toarray().flatten()
        top_similar_users = np.argsort(user_sim)[::-1][1:top_k+1]
        
        # Weighted average of similar users' ratings
        numerator = np.zeros(self.user_item_matrix.shape[1])
        denominator = np.zeros(self.user_item_matrix.shape[1])
        
        for similar_user in top_similar_users:
            sim_score = user_sim[similar_user]
            if sim_score > 0:
                ratings = self.user_item_matrix.iloc[similar_user].values
                numerator += sim_score * ratings
                denominator += sim_score * (ratings > 0)
        
        predictions = np.divide(numerator, denominator, 
                               where=denominator!=0, 
                               out=np.zeros_like(numerator))
        
        return predictions
    
    def predict_item_based(self, user_idx: int, top_k: int = 50) -> np.ndarray:
        """Predict ratings using item-based collaborative filtering"""
        if user_idx >= self.user_item_matrix.shape[0]:
            return np.zeros(self.user_item_matrix.shape[1])
        
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        predictions = np.zeros(len(user_ratings))
        
        for item_idx in range(len(user_ratings)):
            if user_ratings[item_idx] == 0:  # Only predict for unrated items
                item_sim = self.item_similarity[item_idx].toarray().flatten()
                rated_items = np.where(user_ratings > 0)[0]
                
                if len(rated_items) > 0:
                    # Get top similar items that user has rated
                    similar_rated = [(i, item_sim[i]) for i in rated_items if item_sim[i] > 0]
                    similar_rated.sort(key=lambda x: x[1], reverse=True)
                    similar_rated = similar_rated[:top_k]
                    
                    if similar_rated:
                        numerator = sum(user_ratings[i] * sim for i, sim in similar_rated)
                        denominator = sum(sim for _, sim in similar_rated)
                        predictions[item_idx] = numerator / denominator if denominator > 0 else 0
        
        return predictions
    
    def predict_hybrid(self, user_idx: int, alpha: float = 0.5) -> np.ndarray:
        """Hybrid prediction combining user-based and item-based"""
        user_pred = self.predict_user_based(user_idx)
        item_pred = self.predict_item_based(user_idx)
        
        # Combine predictions
        hybrid_pred = alpha * user_pred + (1 - alpha) * item_pred
        
        return hybrid_pred
    
    def recommend_products(self, user_id: int, n: int = None) -> List[Tuple[str, float]]:
        """Generate top-N product recommendations for a user"""
        if n is None:
            n = self.n_recommendations
        
        if user_id not in self.user_lookup:
            # Cold start: recommend popular products
            return self._recommend_popular(n)
        
        user_idx = self.user_lookup[user_id]
        
        # Get predictions
        predictions = self.predict_hybrid(user_idx)
        
        # Get items user hasn't rated
        user_ratings = self.user_item_matrix.iloc[user_idx].values
        unrated_mask = user_ratings == 0
        
        # Mask already rated items
        predictions[~unrated_mask] = -np.inf
        
        # Get top N
        top_indices = np.argsort(predictions)[::-1][:n]
        
        product_ids = self.user_item_matrix.columns
        recommendations = [
            (product_ids[idx], predictions[idx]) 
            for idx in top_indices 
            if predictions[idx] > 0
        ]
        
        return recommendations
    
    def _recommend_popular(self, n: int) -> List[Tuple[str, float]]:
        """Recommend popular products for cold start"""
        product_scores = self.user_item_matrix.sum(axis=0) / (self.user_item_matrix > 0).sum(axis=0)
        top_products = product_scores.nlargest(n)
        return [(prod, score) for prod, score in top_products.items()]
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance"""
        logger.info("Evaluating model...")
        
        y_true = []
        y_pred = []
        
        for _, row in test_df.iterrows():
            user_id = row['user_id']
            product_id = row['product_id']
            true_rating = row['rating']
            
            if user_id in self.user_lookup and product_id in self.product_lookup:
                user_idx = self.user_lookup[user_id]
                prod_idx = self.product_lookup[product_id]
                
                predictions = self.predict_hybrid(user_idx)
                pred_rating = predictions[prod_idx]
                
                if pred_rating > 0:
                    y_true.append(true_rating)
                    y_pred.append(pred_rating)
        
        if len(y_true) == 0:
            return {'rmse': 0, 'mae': 0, 'coverage': 0}
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        coverage = len(y_pred) / len(test_df)
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'coverage': float(coverage),
            'n_predictions': len(y_pred)
        }
        
        logger.info(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Coverage: {coverage:.4f}")
        
        return metrics
    
    def save_model(self, path: str):
        """Save model to disk"""
        model_data = {
            'user_item_matrix': self.user_item_matrix,
            'user_similarity': self.user_similarity,
            'item_similarity': self.item_similarity,
            'user_lookup': self.user_lookup,
            'product_lookup': self.product_lookup,
            'user_mean_ratings': self.user_mean_ratings,
            'global_mean': self.global_mean,
            'n_recommendations': self.n_recommendations,
            'min_interactions': self.min_interactions
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str):
        """Load model from disk"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            n_recommendations=model_data['n_recommendations'],
            min_interactions=model_data['min_interactions']
        )
        
        model.user_item_matrix = model_data['user_item_matrix']
        model.user_similarity = model_data['user_similarity']
        model.item_similarity = model_data['item_similarity']
        model.user_lookup = model_data['user_lookup']
        model.product_lookup = model_data['product_lookup']
        model.user_mean_ratings = model_data['user_mean_ratings']
        model.global_mean = model_data['global_mean']
        
        logger.info(f"Model loaded from {path}")
        return model


def train_with_mlflow(data_path: str, experiment_name: str = "recommendation_model"):
    """Train model with MLflow tracking"""
    
    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
    
    # Hyperparameters to track
    params = {
        'n_recommendations': 10,
        'min_interactions': 2,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'n_users': df['user_id'].nunique(),
        'n_products': df['product_id'].nunique()
    }
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        logger.info("Training model...")
        model = CollaborativeFilteringModel(
            n_recommendations=params['n_recommendations'],
            min_interactions=params['min_interactions']
        )
        
        model.create_interaction_matrix(train_df)
        model.compute_user_similarity()
        model.compute_item_similarity()
        
        # Evaluate
        metrics = model.evaluate(test_df)
        mlflow.log_metrics(metrics)
        
        # Save model
        model_path = "models/recommendation_model.pkl"
        model.save_model(model_path)
        
        # Log model to MLflow
        mlflow.log_artifact(model_path)
        
        # Register model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="recommendation_model"
        )
        
        logger.info("Model training complete!")
        logger.info(f"Metrics: {metrics}")
        
        return model, metrics


if __name__ == "__main__":
    # Train model
    model, metrics = train_with_mlflow("data/cleaned_data.csv")
    
    # Test recommendations
    sample_user = list(model.user_lookup.keys())[0]
    recommendations = model.recommend_products(sample_user, n=5)
    
    print(f"\n=== Sample Recommendations for User {sample_user} ===")
    for product, score in recommendations:
        print(f"Product: {product}, Score: {score:.4f}")