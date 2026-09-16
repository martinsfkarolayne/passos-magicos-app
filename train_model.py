"""
Treino do modelo provisório de risco de defasagem.

Usado enquanto o modelo definitivo da Fase 3 não é concluído. Ao ser
substituído, basta trocar o arquivo modelo_provisorio.pkl e ajustar as
FEATURES em app.py caso as colunas de entrada mudem.

Uso: python3 train_model.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv("dataset_treino.csv")

# IAN foi deixado de fora por ser praticamente equivalente à Defasagem
# (causaria vazamento de dados).
FEATURES = ["IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "INDE_ano_atual", "Mat", "Por"]

# Alvo: aluno considerado em risco quando Defasagem < 0 (abaixo da fase ideal).
df["em_risco"] = (df["Defasagem"] < 0).astype(int)

X = df[FEATURES]
y = df["em_risco"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
print("Acurácia (teste):", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump({"modelo": modelo, "features": FEATURES}, "modelo_provisorio.pkl")
print("Modelo salvo em modelo_provisorio.pkl")
