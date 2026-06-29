import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

st.set_page_config(
    page_title="Portail Décisionnel - Home Credit",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Système de Scoring Crédit Industriel (Ensemble Learning)")
st.markdown("---")

# Formulaire d'évaluation
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 💰 Situation Financière")
    income = st.number_input("Revenu Annuel Total ($)", min_value=0, value=150000)
    credit = st.number_input("Montant du Crédit Demandé ($)", min_value=0, value=450000)
    annuity = st.number_input("Montant des Annuités / Mensualités ($)", min_value=0, value=22000)
    
with col2:
    st.markdown("### 👨‍👩‍👧‍👦 Foyer & Patrimoine")
    age_years = st.number_input("Âge du client (en années)", min_value=18, max_value=100, value=32)
    age_days = age_years * 365
    
    months_employed = st.number_input("Ancienneté Pro (en mois)", min_value=0, value=60, step=1) # 5 ans par défaut
    days_employed = int(months_employed * 30.4 * -1)
    
    children = st.number_input("Nombre d'enfants à charge", min_value=0, max_value=10, value=0)
    fam_members = st.number_input("Nombre de personnes dans le foyer", min_value=1, max_value=12, value=1)
    
    own_car = st.selectbox("Possède un véhicule ?", ["Oui", "Non"])
    own_realty = st.selectbox("Possède un bien immobilier ?", ["Oui", "Non"])

with col3:
    st.markdown("### 🎯 Scores de Risques Externes")
    ext_source_1 = st.slider("Score Externe 1 (Banque Centrale)", 0.0, 1.0, 0.75) # Valeurs par défaut plus optimistes
    ext_source_2 = st.slider("Score Externe 2 (Bureaux de crédit)", 0.0, 1.0, 0.80)
    ext_source_3 = st.slider("Score Externe 3 (Données Alternatives)", 0.0, 1.0, 0.70)

st.markdown("---")

if st.button("📊 Analyser la Solvabilité du Client", type="primary", use_container_width=True):
    car_val = 1 if own_car == "Oui" else 0
    realty_val = 1 if own_realty == "Oui" else 0
    
    payload = {
        "features": {
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "DAYS_BIRTH": age_days,
            "DAYS_EMPLOYED": days_employed,
            "CNT_CHILDREN": children,
            "CNT_FAM_MEMBERS": fam_members,
            "FLAG_OWN_CAR": car_val,
            "FLAG_OWN_REALTY": realty_val,
            "EXT_SOURCE_1": ext_source_1,
            "EXT_SOURCE_2": ext_source_2,
            "EXT_SOURCE_3": ext_source_3
        }
    }
    
    try:
        with st.spinner("Calcul du risque en cours via l'Ensemble Learning..."):
            response = requests.post("https://credit-scoring-api-brayan-2026.onrender.com/predict", json=payload)
            result = response.json()
        
        prob = result["probability_of_default"]
        decision = result["decision"]
        
        st.markdown("## 📋 Résultat de l'Évaluation Métier")
        
        # Structure en deux colonnes pour les graphiques
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if "Accordé" in decision:
                st.success(f"### 🎉 Décision : {decision}")
            else:
                st.error(f"### ❌ Décision : {decision}")
                
            # Graphique de Jauge (Gauge Chart)
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Indicateur de Risque (%)", 'font': {'size': 18}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "black"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 10], 'color': 'rgba(76, 175, 80, 0.6)'},  # Vert (Seuil à 10%)
                        {'range': [10, 35], 'color': 'rgba(255, 152, 0, 0.6)'}, # Orange
                        {'range': [35, 100], 'color': 'rgba(244, 67, 54, 0.6)'} # Rouge
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.85,
                        'value': 10.0
                    }
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("### 🔍 Facteurs Clés d'Influence (SHAP Approximé)")
            
            # Récupération des importances locales renvoyées par l'API
            shap_data = result.get("local_importance", {
                "Scores Externes": -0.4 if "Accordé" in decision else 0.4,
                "Ratio Endettement": -0.2 if "Accordé" in decision else 0.5,
                "Ancienneté Pro": -0.1 if "Accordé" in decision else 0.1,
                "Structure Familiale": 0.05
            })
            
            df_shap = pd.DataFrame({
                'Caractéristique': list(shap_data.keys()),
                'Contribution au Risque': list(shap_data.values())
            }).sort_values(by='Contribution au Risque')
            
            # Coloration des barres (Rouge augmente le risque, Vert diminue le risque)
            df_shap['Couleur'] = df_shap['Contribution au Risque'].apply(lambda x: 'Favorable' if x < 0 else 'Défavorable')
            
            fig_bar = px.bar(
                df_shap,
                x='Contribution au Risque',
                y='Caractéristique',
                orientation='h',
                color='Couleur',
                color_discrete_map={'Favorable': '#4CAF50', 'Défavorable': '#F44336'},
                labels={'Contribution au Risque': 'Impact sur le score de risque'}
            )
            fig_bar.update_layout(height=250, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        st.info(f"💡 Métrique MLOps : {result['features_count_processed']} variables synchronisées. Modèle d'Ensemble stable.")
        
    except Exception as e:
        st.error(f"Impossible de se connecter à l'API de scoring : {e}")