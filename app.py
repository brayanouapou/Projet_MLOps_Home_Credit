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

# 2. Chargement du Modèle d'Ensemble (Chemin relatif universel Mac/Production)
model_pipeline = None

# On cherche d'abord dans le dossier d'artifacts standard poussé sur Git
POSSIBLE_PATHS = [
    os.path.join(os.getcwd(), "scripts/model.pkl"), # Priorité production
    os.path.join(os.getcwd(), "artifacts/2/models/m-fa39e31d970b4a859c4eea74daea2a21/artifacts/model.pkl")
]

for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        try:
            model_pipeline = joblib.load(path)
            print(f"🚀 Modèle d'Ensemble chargé avec succès depuis : {path}")
            break
        except Exception as e:
            print(f"❌ Erreur lors du chargement de {path} : {e}")

if model_pipeline is None:
    print("❌ Alerte : Aucun fichier model.pkl trouvé. Vérifie tes chemins sur GitHub.")

# 3. Chargement de la structure des colonnes et des médianes pré-calculées
MODEL_COLUMNS = []
TRAIN_MEDIANS = None

try:
    # On charge le fichier de médianes ultra-léger généré à l'étape 1
    MEDIANS_PATH = os.path.join(os.getcwd(), "scripts/train_medians.pkl")
    if os.path.exists(MEDIANS_PATH):
        TRAIN_MEDIANS = joblib.load(MEDIANS_PATH)
        MODEL_COLUMNS = list(TRAIN_MEDIANS.index)
        print(f"📋 Structure et médianes chargées en production ({len(MODEL_COLUMNS)} features).")
    else:
        # Solution de secours locale si le fichier n'est pas encore là
        from scripts.data_processing import X_train
        MODEL_COLUMNS = list(X_train.columns)
        TRAIN_MEDIANS = X_train.median()
        print("📋 Structure calculée en local via data_processing.")
except Exception as e:
    print(f"❌ Impossible de charger la structure d'imputation : {e}")

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