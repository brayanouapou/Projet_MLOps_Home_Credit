import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_processing import *

# Connexion au serveur
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Comparaison_Modeles_Credit")

def train_and_log(model, model_name, X_train, y_train, X_val, y_val):
    """
    Entraîne un modèle et loggue les résultats dans MLflow.
    Args:
        model: L'instance du modèle à entraîner (ex: LogisticRegression())
        model_name: Un nom descriptif pour le modèle (ex: "Logistic_Baseline")
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation pour évaluation
    """


    with mlflow.start_run(run_name=model_name): 
        # Entraînement
        model.fit(X_train, y_train) 
        
        # Prédiction
        preds = model.predict(X_val)
        
        # Calcul du score métier (notre ratio 10:1)
        tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel() # tn=Vrai Négatif, fp=Faux Positif, fn=Faux Négatif, tp=Vrai Positif
        business_score = (fn * 10) + (fp * 1) # Coût total combiné des erreurs
        
        # Log des métriques
        mlflow.log_metric("business_score", business_score)
        mlflow.log_metric("accuracy", model.score(X_val, y_val))
        
        # Log du modèle
        mlflow.sklearn.log_model(model, model_name)
        print(f"Modèle {model_name} enregistré avec un score métier de {business_score}")

# Exécution du test (avec tes données de l'étape 2)
# On limite à 10 000 lignes pour que ça aille vite sur ton PC au début
train_and_log(LogisticRegression(max_iter=1000, class_weight='balanced'), "Baseline_Logistic", X_train[:10000], y_train[:10000], X_val, y_val)
train_and_log(RandomForestClassifier(n_estimators=100, class_weight='balanced'), "Random_Forest_Initial", X_train[:10000], y_train[:10000], X_val, y_val)