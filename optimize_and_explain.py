import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# 1. Charger les données
from data_processing import X_train, y_train, X_val, y_val

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Comparaison_Modeles_Credit")

# 2. Définition du Score Métier (Coût à minimiser)
def custom_loss_func(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return (fn * 10) + (fp * 1)

business_scorer = make_scorer(custom_loss_func, greater_is_better=False)

with mlflow.start_run(run_name="Optimized_Logistic_Grid"):
    
    # 3. Utilisation d'un Pipeline : Scaler + Modele (Evite la surconsommation RAM et la dérive de données)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic', LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'))
    ])
    
    # Grille de paramètres adaptée au pipeline (on teste la régularisation C)
    param_grid = {
        'logistic__C': [0.1, 1.0]
    }
    
    # Grid Search avec échantillon optimisé pour la mémoire (5000 lignes)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=business_scorer, cv=3)
    grid_search.fit(X_train[:5000], y_train[:5000])
    
    best_pipeline = grid_search.best_estimator_
    
    # 4. Évaluation sur l'ensemble de validation complet
    val_preds = best_pipeline.predict(X_val)
    final_cost = custom_loss_func(y_val, val_preds)
    
    # 5. Enregistrement des paramètres optimaux
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("business_score", final_cost)
    
    # 6. Feature Importance (Extraite depuis l'étape 'logistic' du pipeline)
    importance = abs(best_pipeline.named_steps['logistic'].coef_[0])
    feat_importances = pd.Series(importance, index=X_train.columns)
    top_features = feat_importances.nlargest(10)
    
    # Sauvegarde graphique
    plt.figure(figsize=(10,6))
    top_features.plot(kind='barh', color='skyblue')
    plt.title('Top 10 - Variables les plus décisionnelles (Feature Importance)')
    plt.xlabel('Poids absolu du coefficient')
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    
    # Log des artefacts et du pipeline de production
    mlflow.log_artifact("feature_importance.png")
    mlflow.sklearn.log_model(best_pipeline, "Best_Logistic_Pipeline")
    
    print(f"Optimisation validée avec succès !")
    print(f"Meilleur paramètre trouvé : {grid_search.best_params_}")
    print(f"Nouveau score de coût métier : {final_cost}")