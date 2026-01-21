"""
FastAPI REST API for Recommendation System
Branch: feature/api-development
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# Import model (try absolute package, then relative, then sys.path fallback)
try:
    from src.recommendation_model import CollaborativeFilteringModel
except Exception:
    try:
        from .recommendation_model import CollaborativeFilteringModel
    except Exception:
        import sys, os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from recommendation_model import CollaborativeFilteringModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
RECOMMENDATION_COUNT = Counter('recommendations_generated_total', 'Total recommendations generated')

# Initialize FastAPI
app = FastAPI(
    title="E-commerce Recommendation API",
    description="Product recommendation system with collaborative filtering",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
product_metadata = None


# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID for recommendations")
    n: int = Field(default=10, ge=1, le=50, description="Number of recommendations")


class Product(BaseModel):
    product_id: str
    name: str
    brand: str
    category: str
    price: float
    avg_rating: float
    score: float


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Product]
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str
    version: str


class MetricsResponse(BaseModel):
    total_users: int
    total_products: int
    model_coverage: float


# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model and data on startup"""
    global model, product_metadata
    
    logger.info("Loading model...")
    model_path = os.getenv("MODEL_PATH", "models/recommendation_model.pkl")
    
    try:
        model = CollaborativeFilteringModel.load_model(model_path)
        logger.info("Model loaded successfully")
        
        # Load product metadata
        data_path = os.getenv("DATA_PATH", "data/cleaned_data.csv")
        df = pd.read_csv(data_path)
        
        product_metadata = df.groupby('product_id').agg({
            'name': 'first',
            'brand': 'first',
            'main_category': 'first',
            'price': 'first',
            'avg_rating': 'first'
        }).to_dict('index')
        
        logger.info(f"Loaded metadata for {len(product_metadata)} products")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't fail startup, but API calls will return 503


def get_model():
    """Dependency to check if model is loaded"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def get_metadata():
    """Dependency to check if metadata is loaded"""
    if product_metadata is None:
        raise HTTPException(status_code=503, detail="Product metadata not loaded")
    return product_metadata


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "E-commerce Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "recommendations": "/recommend",
            "metrics": "/metrics",
            "prometheus": "/prometheus"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    model_instance: CollaborativeFilteringModel = Depends(get_model)
):
    """Get system metrics"""
    return {
        "total_users": len(model_instance.user_lookup),
        "total_products": len(model_instance.product_lookup),
        "model_coverage": len(model_instance.user_lookup) * len(model_instance.product_lookup)
    }


@app.get("/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest
):
    """
    Get product recommendations for a user
    
    - **user_id**: The ID of the user
    - **n**: Number of recommendations to return (1-50)
    """
    # Validate request via Pydantic has already occurred at this point.
    # Resolve model and metadata dependencies here so input validation
    # returns 422 before any 503 from missing model/metadata.
    model_instance = get_model()
    metadata = get_metadata()

    try:
        # Get recommendations from model
        recommendations = model_instance.recommend_products(
            user_id=request.user_id,
            n=request.n
        )
        
        # Enrich with metadata
        enriched_recommendations = []
        for product_id, score in recommendations:
            if product_id in metadata:
                meta = metadata[product_id]
                enriched_recommendations.append(
                    Product(
                        product_id=product_id,
                        name=meta['name'],
                        brand=meta['brand'],
                        category=meta['main_category'],
                        price=meta['price'],
                        avg_rating=meta['avg_rating'],
                        score=float(score)
                    )
                )
        
        RECOMMENDATION_COUNT.inc(len(enriched_recommendations))
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=enriched_recommendations,
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/history")
async def get_user_history(
    user_id: int,
    model_instance: CollaborativeFilteringModel = Depends(get_model)
):
    """Get user's interaction history"""
    if user_id not in model_instance.user_lookup:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_idx = model_instance.user_lookup[user_id]
    user_ratings = model_instance.user_item_matrix.iloc[user_idx]
    
    # Get rated products
    rated_products = user_ratings[user_ratings > 0].to_dict()
    
    return {
        "user_id": user_id,
        "total_interactions": len(rated_products),
        "avg_rating": sum(rated_products.values()) / len(rated_products) if rated_products else 0,
        "products": rated_products
    }


@app.get("/products/{product_id}")
async def get_product_info(
    product_id: str,
    metadata: dict = Depends(get_metadata)
):
    """Get product information"""
    if product_id not in metadata:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return {
        "product_id": product_id,
        **metadata[product_id]
    }


@app.get("/products/{product_id}/similar")
async def get_similar_products(
    product_id: str,
    n: int = 10,
    model_instance: CollaborativeFilteringModel = Depends(get_model),
    metadata: dict = Depends(get_metadata)
):
    """Get similar products using item-based similarity"""
    if product_id not in model_instance.product_lookup:
        raise HTTPException(status_code=404, detail="Product not found")
    
    prod_idx = model_instance.product_lookup[product_id]
    
    # Get item similarity scores (handle sparse matrix or ndarray)
    sim_row = model_instance.item_similarity[prod_idx]
    if hasattr(sim_row, "toarray"):
        similarities = sim_row.toarray().ravel()
    else:
        similarities = np.asarray(sim_row).ravel()
    
    # Get top N (excluding self)
    top_indices = similarities.argsort()[::-1][1:n+1]
    
    similar_products = []
    product_ids = model_instance.user_item_matrix.columns
    
    for idx in top_indices:
        pid = product_ids[idx]
        if pid in metadata:
            similar_products.append({
                "product_id": pid,
                "similarity_score": float(similarities[idx]),
                **metadata[pid]
            })
    
    return {
        "product_id": product_id,
        "similar_products": similar_products
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)