"""
Treino do modelo de risco, reproduzido a partir do notebook "Limpeza,
organização e padronização + Modelo de Machine Learning (Pergunta 9)".

Modelo: XGBoost. Target: Em_Risco_Vigente (1 se IDA < 6.0 OU IEG < 6.0 OU
pedra do ciclo atual == 'Quartzo'). IAN não é usado como feature (evita
data leakage, já que é derivado da própria defasagem histórica).

Uso: python3 train_model.py
Gera: modelo_xgb_passos_magicos.pkl
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import joblib

df = pd.read_csv("dataset_treino.csv")

df["Em_Risco_Vigente"] = np.where(
    (df["IDA"] < 6.0) | (df["IEG"] < 6.0) | (df["pedra_ano_atual"] == "Quartzo"),
    1, 0,
)

COLUNAS = [
    "Fase", "Gênero", "Instituição de ensino",
    "IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "Mat", "Por",
    "Em_Risco_Vigente",
]
df_ml = df[COLUNAS].copy()

X = df_ml.drop(columns=["Em_Risco_Vigente"])
y = df_ml["Em_Risco_Vigente"]

X_encoded = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.20, random_state=42, stratify=y
)

modelo_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss")
modelo_xgb.fit(X_train, y_train)

y_pred = modelo_xgb.predict(X_test)
y_proba = modelo_xgb.predict_proba(X_test)[:, 1]

print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall (Em Risco): {recall_score(y_test, y_pred):.4f}")
print(f"Precision (Em Risco): {precision_score(y_test, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

joblib.dump(
    {"modelo_xgb": modelo_xgb, "colunas_treino": list(X_encoded.columns)},
    "modelo_xgb_passos_magicos.pkl",
)
print("Modelo salvo em modelo_xgb_passos_magicos.pkl")
