from fastapi.testclient import TestClient
from main import app

# On crée un client de test qui simule un navigateur
client = TestClient(app)

def test_health_check():
    """Vérifie si le serveur répond 'healthy'"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_endpoint():
    """Vérifie si la prédiction fonctionne avec des données correctes"""
    # Données simulées (Input)
    payload = {
        "user_id": 1,
        "viewed_products": [101, 102]
    }
    
    # Envoi de la requête POST
    response = client.post("/predict", json=payload)
    
    # Vérifications (Assertions)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert data["user_id"] == 1
    assert len(data["recommendations"]) > 0