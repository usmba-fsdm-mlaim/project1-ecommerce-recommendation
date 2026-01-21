"""
Automated Model Retraining Pipeline
Branch: feature/kubernetes-monitoring
"""

import pandas as pd
import numpy as np
import mlflow
import logging
from datetime import datetime, timedelta
import sys
import os
sys.path.append('..')

from data_preprocessing import DataPreprocessor
from recommendation_model import CollaborativeFilteringModel, train_with_mlflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrainingPipeline:
    """Automated pipeline for model retraining and validation"""
    
    def __init__(self, data_path, model_path, mlflow_uri="http://mlflow:5000"):
        self.data_path = data_path
        self.model_path = model_path
        self.mlflow_uri = mlflow_uri
        self.current_model = None
        self.new_model = None
        
        mlflow.set_tracking_uri(mlflow_uri)
    
    def check_data_quality(self, df):
        """Validate data quality before training"""
        logger.info("Checking data quality...")
        
        checks = {
            'total_records': len(df) >= 1000,
            'unique_users': df['user_id'].nunique() >= 50,
            'unique_products': df['product_id'].nunique() >= 100,
            'rating_range': (df['rating'].min() >= 1.0) and (df['rating'].max() <= 5.0),
            'null_values': df[['user_id', 'product_id', 'rating']].isnull().sum().sum() == 0,
            'date_range': (datetime.now() - pd.to_datetime(df['review_date']).max()).days <= 365
        }
        
        all_passed = all(checks.values())
        
        logger.info(f"Data quality checks: {checks}")
        
        if not all_passed:
            raise ValueError(f"Data quality checks failed: {checks}")
        
        return True
    
    def load_current_model(self):
        """Load the current production model"""
        logger.info("Loading current production model...")
        
        try:
            self.current_model = CollaborativeFilteringModel.load_model(self.model_path)
            logger.info("Current model loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Could not load current model: {e}")
            return False
    
    def train_new_model(self):
        """Train a new model with latest data"""
        logger.info("Training new model...")
        
        # Preprocess latest data
        preprocessor = DataPreprocessor(self.data_path)
        cleaned_data = preprocessor.run_pipeline("data/latest_cleaned.csv")
        
        # Validate data quality
        self.check_data_quality(cleaned_data)
        
        # Train with MLflow
        self.new_model, metrics = train_with_mlflow(
            "data/latest_cleaned.csv",
            experiment_name="model_retraining"
        )
        
        return metrics
    
    def compare_models(self, new_metrics):
        """Compare new model with current production model"""
        logger.info("Comparing models...")
        
        if self.current_model is None:
            logger.info("No current model, accepting new model")
            return True
        
        # Load current model metrics from MLflow
        client = mlflow.tracking.MlflowClient()
        
        try:
            # Get latest production model run
            experiment = client.get_experiment_by_name("recommendation_model")
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.stage='Production'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if runs:
                current_metrics = runs[0].data.metrics
                
                # Compare RMSE (lower is better)
                current_rmse = current_metrics.get('rmse', float('inf'))
                new_rmse = new_metrics.get('rmse', float('inf'))
                
                # Compare coverage (higher is better)
                current_coverage = current_metrics.get('coverage', 0)
                new_coverage = new_metrics.get('coverage', 0)
                
                # Model is better if RMSE is lower by at least 5% AND coverage is not worse
                rmse_improvement = (current_rmse - new_rmse) / current_rmse
                coverage_maintained = new_coverage >= current_coverage * 0.95
                
                logger.info(f"Current RMSE: {current_rmse:.4f}, New RMSE: {new_rmse:.4f}")
                logger.info(f"RMSE improvement: {rmse_improvement*100:.2f}%")
                logger.info(f"Current coverage: {current_coverage:.4f}, New coverage: {new_coverage:.4f}")
                
                is_better = (rmse_improvement >= 0.05) and coverage_maintained
                
                if is_better:
                    logger.info("✓ New model is better than current model")
                else:
                    logger.info("✗ New model is not better than current model")
                
                return is_better
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
        
        # If can't compare, require manual approval
        return False
    
    def promote_to_production(self):
        """Promote new model to production"""
        logger.info("Promoting new model to production...")
        
        # Save new model
        self.new_model.save_model(self.model_path)
        
        # Register in MLflow
        client = mlflow.tracking.MlflowClient()
        
        try:
            # Get latest model version
            latest_versions = client.get_latest_versions("recommendation_model")
            
            if latest_versions:
                latest_version = latest_versions[0].version
                
                # Promote to production
                client.transition_model_version_stage(
                    name="recommendation_model",
                    version=latest_version,
                    stage="Production"
                )
                
                logger.info(f"Model version {latest_version} promoted to Production")
        
        except Exception as e:
            logger.error(f"Error promoting model: {e}")
    
    def run_pipeline(self):
        """Run complete retraining pipeline"""
        logger.info("Starting retraining pipeline...")
        
        try:
            # Step 1: Load current model
            has_current = self.load_current_model()
            
            # Step 2: Train new model
            new_metrics = self.train_new_model()
            
            # Step 3: Compare models
            if self.compare_models(new_metrics):
                # Step 4: Promote to production
                self.promote_to_production()
                
                logger.info("✓ Retraining pipeline completed successfully")
                return True
            else:
                logger.warning("✗ New model not promoted - performance not improved")
                return False
        
        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            raise


def schedule_retraining():
    """Schedule periodic model retraining"""
    import schedule
    import time
    
    def job():
        logger.info("Scheduled retraining triggered")
        pipeline = RetrainingPipeline(
            data_path=os.getenv("DATA_PATH", "data/raw_data.csv"),
            model_path=os.getenv("MODEL_PATH", "models/recommendation_model.pkl")
        )
        pipeline.run_pipeline()
    
    # Schedule retraining every week
    schedule.every().monday.at("02:00").do(job)
    
    logger.info("Retraining scheduler started")
    
    while True:
        schedule.run_pending()
        time.sleep(3600)  # Check every hour


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model Retraining Pipeline")
    parser.add_argument("--mode", choices=["once", "schedule"], default="once",
                       help="Run once or schedule periodic retraining")
    parser.add_argument("--data-path", default="data/raw_data.csv",
                       help="Path to raw data")
    parser.add_argument("--model-path", default="models/recommendation_model.pkl",
                       help="Path to save model")
    
    args = parser.parse_args()
    
    if args.mode == "once":
        pipeline = RetrainingPipeline(args.data_path, args.model_path)
        success = pipeline.run_pipeline()
        sys.exit(0 if success else 1)
    else:
        schedule_retraining()