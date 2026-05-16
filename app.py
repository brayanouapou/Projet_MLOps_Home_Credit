from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import numpy as np
from pydantic import BaseModel, Extra
from typing import Dict, Any

# 1. Configuration de l'URI de suivi MLFlow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

app = FastAPI(
    title="API de Scoring Crédit - Home Credit",
    description="API MLOps industrielle pour prédire le risque de défaut de paiement.",
    version="1.1"
)

# 2. Chargement du Pipeline de Production
MODEL_URI = "runs:/5964c3d6e6a1429389eff8869cd6234e/Best_Logistic_Pipeline"
try:
    model_pipeline = mlflow.sklearn.load_model(MODEL_URI)
    print("🚀 Modèle chargé avec succès depuis MLFlow !")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")

# 3. Chargement de la structure exacte des colonnes (Feature Store local)
# On importe la liste des colonnes d'entraînement pour garantir l'alignement
try:
    from data_processing import X_train
    MODEL_COLUMNS = list(X_train.columns)
    print(f"📋 Structure des features chargée : {len(MODEL_COLUMNS)} colonnes attendues.")
except Exception as e:
    print(f"❌ Impossible de charger la structure des colonnes : {e}")
    MODEL_COLUMNS = []

class ClientData(BaseModel):
    features: Dict[str, Any]
    class Config:
        extra = Extra.allow

@app.get("/")
def home():
    return {"message": "Bienvenue sur l'API de Scoring Crédit. Utilisez /docs pour voir la documentation."}

@app.post("/predict")
def predict_credit_risk(client: ClientData):
    # a. Créer un DataFrame avec les données saisies par l'utilisateur
    df_input = pd.DataFrame([client.features])
    
    # b. Reconstruire un DataFrame vide contenant TOUTES les colonnes requises par le modèle
    df_api = pd.DataFrame(columns=MODEL_COLUMNS)
    
    # c. Fusionner les données saisies dans la structure globale
    # Les colonnes manquantes prendront la valeur NaN temporairement
    df_api = pd.concat([df_api, df_input], axis=0, ignore_index=True)
    
    # d. Remplir les colonnes manquantes (Imputation locale à la volée)
    # Pour les variables créées par get_dummies (ex: OCCUPATION_TYPE_...), l'absence signifie 0
    # Pour les variables numériques principales, on met 0 ou la valeur par défaut
    df_api = df_api.fillna(0)
    
    # e. S'assurer que l'ordre des colonnes est STRICTEMENT identique à l'entraînement
    df_api = df_api[MODEL_COLUMNS]
    
    # f. Calcul des prédictions
    probability = model_pipeline.predict_proba(df_api)[0][1]
    prediction = int(model_pipeline.predict(df_api)[0])
    
    status = "Refusé (Risque élevé)" if prediction == 1 else "Accordé (Client fiable)"
    
    return {
        "probability_of_default": round(float(probability), 4),
        "decision": status,
        "features_count_received": len(client.features),
        "features_count_processed": df_api.shape[1]
    }