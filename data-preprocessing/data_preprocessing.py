"""
Data Preprocessing Pipeline for E-commerce Recommendation System
Branch: feature/data-preprocessing
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import re
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Clean and prepare Amazon product review data for recommendation system"""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        """Load raw CSV data"""
        logger.info(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(self.df)} records")
        return self
    
    def clean_basic_fields(self):
        """Clean basic text and numeric fields"""
        logger.info("Cleaning basic fields...")
        
        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates(subset=['id', 'reviews.username', 'reviews.date'])
        logger.info(f"Removed {initial_rows - len(self.df)} duplicate records")
        
        # Clean brand names
        self.df['brand'] = self.df['brand'].str.strip()
        
        # Clean product names
        self.df['name'] = self.df['name'].str.strip()
        
        # Handle missing reviews
        self.df = self.df[self.df['reviews.text'].notna()]
        self.df = self.df[self.df['reviews.rating'].notna()]
        
        logger.info(f"Records after cleaning: {len(self.df)}")
        return self
    
    def parse_prices(self):
        """Extract price information from JSON field"""
        logger.info("Parsing prices...")
        
        def extract_price(price_json):
            try:
                if pd.isna(price_json):
                    return np.nan
                prices = json.loads(price_json)
                if prices and len(prices) > 0:
                    return float(prices[0].get('amountMin', np.nan))
            except:
                return np.nan
        
        self.df['price'] = self.df['prices'].apply(extract_price)
        
        # Fill missing prices with median by brand
        self.df['price'] = self.df.groupby('brand')['price'].transform(
            lambda x: x.fillna(x.median())
        )
        
        return self
    
    def parse_categories(self):
        """Extract and clean category information"""
        logger.info("Parsing categories...")
        
        def extract_main_category(cat_string):
            if pd.isna(cat_string):
                return 'Unknown'
            cats = str(cat_string).split(',')
            return cats[0].strip() if cats else 'Unknown'
        
        self.df['main_category'] = self.df['categories'].apply(extract_main_category)
        
        return self
    
    def process_dates(self):
        """Convert date strings to datetime and extract features"""
        logger.info("Processing dates...")
        
        self.df['review_date'] = pd.to_datetime(self.df['reviews.date'], errors='coerce')
        self.df['date_added'] = pd.to_datetime(self.df['dateAdded'], errors='coerce')
        
        # Extract temporal features
        self.df['review_year'] = self.df['review_date'].dt.year
        self.df['review_month'] = self.df['review_date'].dt.month
        
        # Calculate product age at review time
        self.df['product_age_days'] = (
            self.df['review_date'] - self.df['date_added']
        ).dt.days
        
        return self
    
    def clean_user_data(self):
        """Clean and standardize user information"""
        logger.info("Cleaning user data...")
        
        # Clean usernames
        self.df['username'] = self.df['reviews.username'].fillna('Anonymous')
        self.df['username'] = self.df['username'].str.strip()
        
        # Create user ID
        le = LabelEncoder()
        self.df['user_id'] = le.fit_transform(self.df['username'])
        
        return self
    
    def create_product_features(self):
        """Create aggregated product features"""
        logger.info("Creating product features...")
        
        # Average rating per product
        product_stats = self.df.groupby('id').agg({
            'reviews.rating': ['mean', 'count', 'std'],
            'reviews.numHelpful': 'sum',
            'price': 'first'
        }).reset_index()
        
        product_stats.columns = [
            'id', 'avg_rating', 'num_reviews', 'rating_std', 
            'total_helpful', 'price'
        ]
        
        # Fill NaN std with 0
        product_stats['rating_std'] = product_stats['rating_std'].fillna(0)
        
        self.df = self.df.merge(product_stats, on='id', how='left', suffixes=('', '_agg'))
        
        return self
    
    def create_user_features(self):
        """Create aggregated user features"""
        logger.info("Creating user features...")
        
        user_stats = self.df.groupby('user_id').agg({
            'reviews.rating': ['mean', 'count'],
            'reviews.numHelpful': 'sum'
        }).reset_index()
        
        user_stats.columns = [
            'user_id', 'user_avg_rating', 'user_review_count', 'user_total_helpful'
        ]
        
        self.df = self.df.merge(user_stats, on='user_id', how='left')
        
        return self
    
    def create_final_dataset(self):
        """Create final clean dataset for modeling"""
        logger.info("Creating final dataset...")
        
        # Select relevant columns
        self.cleaned_df = self.df[[
            'id', 'user_id', 'username', 'asins', 'brand', 'name',
            'main_category', 'price', 'reviews.rating', 'reviews.text',
            'reviews.title', 'reviews.numHelpful', 'review_date',
            'avg_rating', 'num_reviews', 'rating_std', 'total_helpful',
            'user_avg_rating', 'user_review_count', 'user_total_helpful',
            'product_age_days'
        ]].copy()
        
        # Rename for clarity
        self.cleaned_df.rename(columns={
            'id': 'product_id',
            'reviews.rating': 'rating',
            'reviews.text': 'review_text',
            'reviews.title': 'review_title',
            'reviews.numHelpful': 'helpful_votes'
        }, inplace=True)
        
        # Remove any remaining NaN in critical columns
        self.cleaned_df = self.cleaned_df.dropna(subset=['product_id', 'user_id', 'rating'])
        
        logger.info(f"Final dataset shape: {self.cleaned_df.shape}")
        
        return self
    
    def save_cleaned_data(self, output_path='data/cleaned_data.csv'):
        """Save cleaned dataset"""
        import os  # Au cas où
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f"Saving cleaned data to {output_path}")
        self.cleaned_df.to_csv(output_path, index=False)
        
        # Save summary statistics
        summary = {
            'total_records': len(self.cleaned_df),
            'unique_users': self.cleaned_df['user_id'].nunique(),
            'unique_products': self.cleaned_df['product_id'].nunique(),
            'unique_brands': self.cleaned_df['brand'].nunique(),
            'avg_rating': float(self.cleaned_df['rating'].mean()),
            'date_range': {
                'start': str(self.cleaned_df['review_date'].min()),
                'end': str(self.cleaned_df['review_date'].max())
            }
        }
        
        with open('data/data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Data preprocessing complete!")
        return self
    
    def run_pipeline(self, output_path='data/cleaned_data.csv'):
        """Run complete preprocessing pipeline"""
        (self.load_data()
            .clean_basic_fields()
            .parse_prices()
            .parse_categories()
            .process_dates()
            .clean_user_data()
            .create_product_features()
            .create_user_features()
            .create_final_dataset()
            .save_cleaned_data(output_path))
        
        return self.cleaned_df


# Unit tests
def test_preprocessing():
    """Basic tests for preprocessing pipeline"""
    import tempfile
    
    # Create sample data
    sample_data = {
        'id': ['P1', 'P1', 'P2'],
        'asins': ['A1', 'A1', 'A2'],
        'brand': ['Brand1', 'Brand1', 'Brand2'],
        'categories': ['Cat1,Cat2', 'Cat1,Cat2', 'Cat3'],
        'name': ['Product 1', 'Product 1', 'Product 2'],
        'prices': ['[{"amountMin":10.99}]', '[{"amountMin":10.99}]', '[{"amountMin":20.99}]'],
        'dateAdded': ['2020-01-01', '2020-01-01', '2020-02-01'],
        'reviews.date': ['2020-06-01', '2020-07-01', '2020-08-01'],
        'reviews.rating': [5.0, 4.0, 3.0],
        'reviews.text': ['Great!', 'Good', 'OK'],
        'reviews.title': ['Love it', 'Nice', 'Fine'],
        'reviews.username': ['User1', 'User2', 'User1'],
        'reviews.numHelpful': [10.0, 5.0, 2.0],
        'reviews.sourceURLs': ['url1', 'url2', 'url3'],
        'dateUpdated': ['2020-01-01', '2020-01-01', '2020-02-01'],
        'keys': ['k1', 'k1', 'k2']
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        pd.DataFrame(sample_data).to_csv(f.name, index=False)
        temp_path = f.name
    
    processor = DataPreprocessor(temp_path)
    result = processor.run_pipeline('test_output.csv')
    
    assert len(result) == 3
    assert 'product_id' in result.columns
    assert 'user_id' in result.columns
    assert result['rating'].notna().all()
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    # Run preprocessing
    preprocessor = DataPreprocessor("7817_1.csv")
    cleaned_data = preprocessor.run_pipeline()
    
    print("\n=== Preprocessing Summary ===")
    print(f"Total records: {len(cleaned_data)}")
    print(f"Unique users: {cleaned_data['user_id'].nunique()}")
    print(f"Unique products: {cleaned_data['product_id'].nunique()}")
    print(f"Average rating: {cleaned_data['rating'].mean():.2f}")