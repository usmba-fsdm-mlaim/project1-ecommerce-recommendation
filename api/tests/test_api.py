from fastapi.testclient import TestClient
import sys
import os

# Ajout du dossier parent au chemin système pour trouver app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # On importe depuis 'app', pas 'main'

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    payload = {
        "user_id": 1,
        "viewed_products": [10, 20]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert data["user_id"] == 1