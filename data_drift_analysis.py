import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os

print("📊 Chargement des données pour l'analyse du Data Drift...")
from data_processing import X_train, test_proc

# 1. Sélection des variables clés importantes
target_features = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'ANNUITY_INCOME_PERC', 'PAYMENT_RATE'
]

# Créer un dossier pour stocker nos graphiques temporaires
os.makedirs("static", exist_ok=True)

html_cards = ""

print("🧬 Calcul des dérives et génération des graphiques...")
for col in target_features:
    # Récupération des distributions (en supprimant les NaN)
    train_dist = X_train[col].dropna()
    test_dist = test_proc[col].dropna()
    
    # Test statistique de Kolmogorov-Smirnov (le standard du Data Drift)
    statistic, p_value = ks_2samp(train_dist, test_dist)
    
    # Seuil standard : si p-value < 0.05, la distribution a changé (Drift)
    drift_detected = p_value < 0.05
    
    # Génération du graphique pour cette variable
    plt.figure(figsize=(6, 3.5))
    plt.hist(train_dist, bins=30, alpha=0.5, label='Entraînement', density=True, color='#1f77b4')
    plt.hist(test_dist, bins=30, alpha=0.5, label='Production (Test)', density=True, color='#ff7f0e')
    plt.title(f"Distribution : {col}", fontsize=10)
    plt.legend(fontsize=8)
    plt.tight_layout()
    
    # Sauvegarde de l'image
    img_path = f"static/drift_{col}.png"
    plt.savefig(img_path)
    plt.close()
    
    # Assignation des couleurs selon le résultat
    if drift_detected:
        status_badge = '<span class="badge bg-danger fs-6">DÉRAILLEMENT (Drift)</span>'
        border_color = "border-danger"
    else:
        status_badge = '<span class="badge bg-success fs-6">STABLE</span>'
        border_color = "border-success"
        
    # Création de la carte HTML pour cette variable
    html_cards += f"""
    <div class="col-md-6 mb-4">
        <div class="card h-100 border-start border-4 {border_color}">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h4 class="card-title text-dark mb-0"><code>{col}</code></h4>
                    {status_badge}
                </div>
                <p class="mb-2"><strong>Score de dérive (p-value) :</strong> {p_value:.4f}</p>
                <p class="text-muted small mb-3">Méthode : Test statistique de Kolmogorov-Smirnov</p>
                <div class="text-center">
                    <img src="{img_path}" class="img-fluid rounded" alt="Graphique de distribution">
                </div>
            </div>
        </div>
    </div>
    """

# 2. Construction de la page Web complète
html_template = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Dashboard de Data Drift - Home Credit</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body {{ background-color: #f4f6f9; padding: 40px 0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        .card {{ box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: none; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="d-flex align-items-center justify-content-between mb-4">
            <div>
                <h1 class="fw-bold text-dark">🏦 Portail MLOps — Surveillance du Data Drift</h1>
                <p class="text-muted mb-0">Analyse de stabilité des variables clés pour le scoring de crédit</p>
            </div>
            <span class="badge bg-dark fs-6 py-2 px-3">Master 2 Big Data & AI</span>
        </div>
        <hr class="mb-5">
        
        <div class="row">
            {html_cards}
        </div>
    </div>
</body>
</html>
"""

# 3. Écriture physique du rapport HTML sur le disque
report_path = "data_drift_report.html"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(html_template)

print(f"\n🎉 VICTOIRE TOTALE : Ton magnifique rapport interactif est créé sous : {report_path}")
print("💡 Tape : explorer.exe 'data_drift_report.html' pour voir tes graphiques !")