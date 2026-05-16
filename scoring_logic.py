from sklearn.metrics import confusion_matrix, make_scorer

def custom_business_score(y_true, y_pred):
    """
    Calcule le coût métier : 
    Pénalité forte pour les Faux Négatifs (perte de capital)
    Pénalité modérée pour les Faux Positifs (perte d'opportunité)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() # tn=Vrai Négatif, fp=Faux Positif, fn=Faux Négatif, tp=Vrai Positif
    
    # Définition des poids (Ratio 10:1 typique en banque)
    cost_fn = 10 # Coût d'un Faux Négatif (client risqué accepté)
    cost_fp = 1 # Coût d'un Faux Positif (client sain refusé)
    
    total_cost = (fn * cost_fn) + (fp * cost_fp) # Coût total combiné des erreurs
    
    # On normalise par le nombre de prédictions pour avoir une moyenne
    return total_cost / len(y_true)

# On transforme cette fonction en un "scorer" compatible avec GridSearchCV (Etape 4)
business_scorer = make_scorer(custom_business_score, greater_is_better=False)