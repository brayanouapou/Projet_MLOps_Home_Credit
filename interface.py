import streamlit as st
import requests

st.set_page_config(
    page_title="Portail Décisionnel - Home Credit",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Système de Scoring Crédit Automatisé")
st.markdown("---")
st.markdown("### Saisie des informations du demandeur")

# 1. Formulaire d'évaluation avec des valeurs par défaut réalistes
col1, col2 = st.columns(2)

with col1:
    income = st.number_input("Revenu Annuel Total ($)", min_value=0, value=150000)
    credit = st.number_input("Montant du Crédit Demandé ($)", min_value=0, value=450000)
    annuity = st.number_input("Montant des Annuités / Mensualités ($)", min_value=0, value=22000)
    age_years = st.number_input("Âge du client (en années)", min_value=18, max_value=100, value=32)
    # Conversion automatique en jours pour le modèle
    age_days = age_years * 365

with col2:
    days_employed = st.number_input("Ancienneté Pro (jours négatifs)", value=-500)
    ext_source_1 = st.slider("Score Externe 1", 0.0, 1.0, 0.5)
    ext_source_2 = st.slider("Score Externe 2", 0.0, 1.0, 0.6)
    ext_source_3 = st.slider("Score Externe 3", 0.0, 1.0, 0.4)

st.markdown("---")

# 2. Bouton d'action qui appelle ton API FastAPI
if st.button("📊 Analyser la Solvabilité du Client", type="primary"):
    
    # Préparation du dictionnaire de caractéristiques attendu par ton API
    payload = {
        "features": {
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "DAYS_BIRTH": age_days,
            "DAYS_EMPLOYED": days_employed,
            "EXT_SOURCE_1": ext_source_1,
            "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_3": ext_source_3
        }
    }
    
    try:
        # Requête vers ton API locale FastAPI
        with st.spinner("Analyse du profil en cours par l'algorithme..."):
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            result = response.json()
        
        # 3. Affichage visuel des résultats
        prob = result["probability_of_default"]
        decision = result["decision"]
        
        st.subheader("📋 Résultat de l'Évaluation Métier")
        
        if "Accordé" in decision:
            st.success(f"### 🎉 Décision : {decision}")
        else:
            st.error(f"### ❌ Décision : {decision}")
            
        # Affichage du score de probabilité
        st.metric(label="Probabilité de défaut de paiement", value=f"{prob * 100:.2f} %")
        st.progress(prob)
        
        st.info(f"💡 Métrique système : {result['features_count_processed']} variables alignées et injectées en production.")
        
    except Exception as e:
        st.error(f"Impossible de se connecter à l'API de scoring : {e}")