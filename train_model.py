import sys
sys.path.append('.')

from src.recommendation_model import train_with_mlflow
#hello 
if __name__ == "__main__":
    model, metrics = train_with_mlflow("data/cleaned_data.csv")
    print("✅ Model trained!")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   Coverage: {metrics['coverage']:.4f}")
