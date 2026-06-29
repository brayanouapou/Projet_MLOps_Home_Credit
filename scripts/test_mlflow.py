import mlflow
import os

# 1. Connexion au serveur que tu as lancé
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Configuration_Examen_MLOps_Brayan_Ouapou")

with mlflow.start_run(run_name="Test_Initial"):
    # Enregistrement d'un paramètre (ex: type de modèle)
    mlflow.log_param("env", "Windows_Venv")
    
    # Enregistrement d'une métrique (ex: une précision fictive)
    mlflow.log_metric("dummy_accuracy", 0.95)
    
    # Création et enregistrement d'un artefact (un fichier texte simple)
    with open("notes.txt", "w") as f:
        f.write("Validation de l'étape 1 du projet Home Credit.")
    mlflow.log_artifact("notes.txt")

print("Test terminé ! Rafraîchis ton navigateur à l'adresse http://127.0.0.1:5000")