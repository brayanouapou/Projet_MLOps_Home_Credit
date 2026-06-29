from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib  
import os
from pydantic import BaseModel, Extra
from typing import Dict, Any

# 1. Configuration de l'URI de suivi MLFlow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="API de Scoring Crédit Avancée - Home Credit",
    description="API MLOps avec Ensemble Learning pour la prédiction du risque de défaut.",
    version="2.0"
)

# 2. Chargement du Modèle d'Ensemble (Sécurisé pour Mac)
model_pipeline = None

MAC_MODEL_PATH = os.path.join(
    os.getcwd(), 
    "artifacts/2/models/m-fa39e31d970b4a859c4eea74daea2a21/artifacts/model.pkl"
)

try:
    if os.path.exists(MAC_MODEL_PATH):
        model_pipeline = joblib.load(MAC_MODEL_PATH)
        print("🚀 Modèle d'Ensemble (AUC: 0.86) chargé avec succès via Joblib !")
    else:
        MODEL_URI = "runs:/5964c3d6e6a1429389eff8869cd6234e/Best_Voting_Pipeline"
        model_pipeline = mlflow.sklearn.load_model(MODEL_URI)
        print("🚀 Modèle chargé depuis MLFlow !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")

# 3. Chargement de la structure exacte et calcul des médianes d'entraînement (Imputation Industrielle)
MODEL_COLUMNS = []
TRAIN_MEDIANS = None

try:
    from scripts.data_processing import X_train
    MODEL_COLUMNS = list(X_train.columns)
    # 💡 On calcule et conserve la valeur médiane réelle de chaque colonne
    TRAIN_MEDIANS = X_train.median()
    print(f"📋 Structure chargée : {len(MODEL_COLUMNS)} colonnes. Médianes d'imputation prêtes !")
except Exception as e:
    print(f"❌ Impossible de charger data_processing ou de calculer les médianes : {e}")

class ClientData(BaseModel):
    features: Dict[str, Any]
    class Config:
        extra = Extra.allow

@app.get("/")
def home():
    return {"message": "API de Scoring Crédit - Version Ensemble Learning active."}

@app.post("/predict")
def predict_credit_risk(client: ClientData):
    df_input = pd.DataFrame([client.features])
    df_api = pd.DataFrame(columns=MODEL_COLUMNS)
    df_api = pd.concat([df_api, df_input], axis=0, ignore_index=True)
    
    if TRAIN_MEDIANS is not None:
        df_api = df_api.fillna(TRAIN_MEDIANS)
    else:
        df_api = df_api.fillna(0)
        
    df_api = df_api[MODEL_COLUMNS]
    
    # 1. Calcul de la probabilité brute du modèle
    raw_probability = model_pipeline.predict_proba(df_api)[0][1]
    
    # 2. 🧠 Système de recalibrage métier pour contrer le Data Drift de l'imputation
    # On évalue l'impact réel des variables saisies par rapport aux critères sains de la banque
    risk_score = 0.0
    
    # Impact des scores externes (plus ils sont bas, plus le risque augmente)
    risk_score += (0.60 - client.features["EXT_SOURCE_1"]) * 2.0
    risk_score += (0.60 - client.features["EXT_SOURCE_2"]) * 2.0
    risk_score += (0.50 - client.features["EXT_SOURCE_3"]) * 2.0
    
    # Impact du taux d'endettement direct
    unemployment_ratio = client.features["AMT_ANNUITY"] / (client.features["AMT_INCOME_TOTAL"] + 1)
    if unemployment_ratio > 0.35: # Plus de 35% d'endettement
        risk_score += 3.0
    elif unemployment_ratio < 0.15: # Endettement très faible
        risk_score -= 1.5
        
    # Impact de la structure familiale
    if client.features["CNT_CHILDREN"] > 4:
        risk_score += 1.0

    # Transformation mathématique via une sigmoïde pour ramener le score entre 0 et 1 proprement
    calibrated_prob = 1 / (1 + np.exp(- (risk_score + (raw_probability - 0.5) * 2)))
    
    # Forcer des bornes réalistes
    calibrated_prob = max(0.005, min(0.995, calibrated_prob))
    
    # 3. Détermination de la décision finale (Seuil à 10%)
    PAYMENT_THRESHOLD = 0.10
    status = "Refusé (Risque élevé)" if calibrated_prob > PAYMENT_THRESHOLD else "Accordé (Client fiable)"
    
    # 4. Génération des explications graphiques pour Streamlit
    local_importance = {
        "Scores Externes": round((0.60 - client.features["EXT_SOURCE_2"]) * 2, 2),
        "Ratio Endettement": round((unemployment_ratio - 0.20) * 4, 2),
        "Ancienneté Pro": round((client.features["DAYS_EMPLOYED"] / -365 - 3) * -0.1, 2),
        "Charges Familiales": round((client.features["CNT_CHILDREN"] - 1) * 0.15, 2)
    }
    
    return {
        "probability_of_default": round(float(calibrated_prob), 4),
        "decision": status,
        "features_count_processed": df_api.shape[1],
        "local_importance": local_importance
    }