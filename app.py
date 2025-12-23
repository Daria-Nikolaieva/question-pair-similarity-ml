import pickle
import numpy as np
import pandas as pd
import torch
import joblib
from torch import nn

from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.Functions import (
    deploy_features
)

# Init app
app = FastAPI(title="Question Duplicate Detector")

# Load models
lgbm_model = joblib.load("models/lgbm_best_model.pkl")
scaler = joblib.load("models/sbert_scaler_0,28.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

device = "cuda" if torch.cuda.is_available() else "cpu"

class SBERTMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)


sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)


class PredictionRequest(BaseModel):
    question1: str
    question2: str

    class Config:
        schema_extra = {
            "example": {
                "question1": "How to learn machine learning?",
                "question2": "What is the best way to study ML?"
            }
        }

class PredictionResponse(BaseModel):
    probability: float
    label: int

    class Config:
        schema_extra = {
            "example": {
                "probability": 0.87,
                "label": 1
            }
        }


@app.get("/")
def read_root():
    return {"message": "Welcome to the ML Model API!"
            "This is an API for checking if questions are duplicates using ML."}
# --------------------
# Prediction
# --------------------
@app.post("/predict")
def predict(request: PredictionRequest):

    df = pd.DataFrame([{
        "question1": request.question1,
        "question2": request.question2
    }])

    # LightGBM
    X = deploy_features(df, vectorizer)
    lgbm_proba = lgbm_model.predict_proba(X)[:, 1]

    # MLP
    device = torch.device("cpu")
    mlp_model = SBERTMLP(input_dim=INPUT_DIM)
    mlp_model.load_state_dict(torch.load("models/sbert_mlp.pt", map_location=device))
    mlp_model.eval()
    emb_q1 = sbert.encode(
    df["question1"].tolist(),
    batch_size=64
    )
    emb_q2 = sbert.encode(
        df["question2"].tolist(),
        batch_size=64
    )
    X_sbert = np.hstack([emb_q1, emb_q2, np.abs(emb_q1 - emb_q2)])
    X_sbert_scaled = scaler.transform(X_sbert)
    X_tensor = torch.tensor(X_sbert_scaled, dtype=torch.float32).to(device)
    INPUT_DIM = X_tensor.shape[1]

    with torch.no_grad():
        mlp_proba = torch.sigmoid(mlp_model(X_tensor)).cpu().item()

    # Weighted ensemble
    final_proba = 0.4 * lgbm_proba[0] + 0.6 * mlp_proba

    return {
        "duplicate_probability": round(float(final_proba), 4),
        "is_duplicate": final_proba > 0.4
    }




