import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # 1. Traitement des anomalies temporelles
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True) 
    if 'DAYS_BIRTH' in df.columns:
        df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH']) 
    
    # 2. Feature Engineering
    if 'AMT_ANNUITY' in df.columns and 'AMT_INCOME_TOTAL' in df.columns:
        df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'] 
    if 'AMT_ANNUITY' in df.columns and 'AMT_CREDIT' in df.columns:
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT'] 
        
    # 3. Imputation des valeurs manquantes pour les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'TARGET': 
            df[col] = df[col].fillna(df[col].median())
            
    # 4. Encodage des variables binaires textuelles (<= 2 catégories)
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            if len(list(df[col].unique())) <= 2:
                df[col] = le.fit_transform(df[col].astype(str))
                
    # 5. One-Hot Encoding pour les autres variables textuelles
    df = pd.get_dummies(df)
    
    return df

# --- Chargement par morceaux (Chunking) sécurisé ---

print("Chargement sécurisé de application_train.csv...")
with pd.read_csv("data/application_train.csv", chunksize=10000) as reader:
    for chunk in reader:
        train_raw = chunk
        break  

print("Chargement sécurisé de application_test.csv...")
with pd.read_csv("data/application_test.csv", chunksize=2000) as reader:
    for chunk in reader:
        test_raw = chunk
        break  

# --- Bloc d'exécution du prétraitement (Décommenté !) ---

print("Exécution du prétraitement...")
train_proc = preprocess_data(train_raw)
test_proc = preprocess_data(test_raw)

# Alignement des colonnes entre Train et Test
train_labels = train_proc['TARGET'] 
train_proc, test_proc = train_proc.align(test_proc, join='inner', axis=1) 
train_proc['TARGET'] = train_labels 

# Division Train / Validation
X = train_proc.drop('TARGET', axis=1) 
y = train_proc['TARGET'] 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

print(f"Prétraitement validé. Colonnes finales : {X_train.shape[1]}")