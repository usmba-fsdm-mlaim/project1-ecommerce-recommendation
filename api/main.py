from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialisation de l'application
app = FastAPI(title="Recommender System API", version="1.0.0")

# --- Modèle de données (Ce que l'utilisateur doit envoyer) ---
class UserHistory(BaseModel):
    user_id: int
    viewed_products: list[int]

# --- Base de données factice (En attendant le vrai modèle ML) ---
fake_model_db = {
    1: ["Casque Audio Sony", "Clavier Mécanique"],
    2: ["Livre Python", "Pc Portable Gamer"],
    10: ["Ecran 4k", "Souris sans fil"]
}

# --- Route 1: Health Check (Pour Kubernetes) ---
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# --- Route 2: Prediction (Le cœur du projet) ---
@app.post("/predict")
def predict(history: UserHistory):
    # On simule une prédiction : si l'ID existe on renvoie des produits, sinon une liste par défaut
    recommendations = fake_model_db.get(history.user_id, ["Produit Populaire A", "Produit Populaire B"])
    
    return {
        "user_id": history.user_id,
        "recommendations": recommendations,
        "model_version": "v1-dummy" # Utile pour le tracking MLflow plus tard
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)