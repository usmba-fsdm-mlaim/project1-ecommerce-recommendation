from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Recommender System API")

# Monter le dossier static pour servir index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

# Modèles de données
class UserHistory(BaseModel):
    user_id: int
    viewed_products: List[int]

class Response(BaseModel):
    user_id: int
    recommendations: List[int]

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict", response_model=Response)
def predict(history: UserHistory):
    # Logique fictive (Mock)
    recommendations = [101, 102, 103]
    return {"user_id": history.user_id, "recommendations": recommendations}