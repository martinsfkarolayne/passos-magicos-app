"""
Script para treinar o modelo PROVISÓRIO de risco de defasagem.

Este script existe só para nós testarmos o aplicativo Streamlit inteiro
(Fase 4) enquanto a Thaty ainda não termina o modelo real (Fase 3).

Quando a Thaty mandar o modelo de verdade, este arquivo deixa de ser
necessário — a gente só troca o "modelo_provisorio.pkl" pelo arquivo
que ela mandar, e ajusta o app.py se as colunas de entrada forem diferentes.

Como rodar (no Terminal, dentro da pasta do projeto):
    python3 train_model.py

Isso vai gerar (ou atualizar) o arquivo "modelo_provisorio.pkl".
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Carregar a base de dados que a Thaty enviou (já limpa pelo Fabio na Fase 1)
df = pd.read_csv("dataset_treino.csv")

# 2. Definir as colunas que o modelo vai usar como "pistas" (features)
#    Escolhemos indicadores numéricos que já existem prontos na base,
#    sem valores faltando, para manter o modelo provisório simples.
# (Não usamos "IAN" aqui de propósito: o IAN já é basicamente a mesma
#  informação da Defasagem, então usá-lo "entregaria a resposta" pro
#  modelo em vez de ele aprender de verdade — isso se chama "vazamento
#  de dados". O modelo real da Thaty pode decidir usar outras colunas.)
FEATURES = ["IDA", "IEG", "IAA", "IPS", "IPP", "IPV", "INDE_ano_atual", "Mat", "Por"]

# 3. Definir o "alvo" (o que o modelo tenta prever)
#    Regra simples: consideramos que o aluno está "em risco de defasagem"
#    quando o valor de Defasagem é menor que 0 (ou seja, já está atrasado
#    em relação à fase ideal). Isso é uma simplificação só para o modelo
#    provisório — o modelo real da Thaty pode usar uma lógica mais completa.
df["em_risco"] = (df["Defasagem"] < 0).astype(int)

X = df[FEATURES]
y = df["em_risco"]

# 4. Separar dados de treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Treinar um modelo simples (Random Forest)
modelo = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
modelo.fit(X_train, y_train)

# 6. Avaliar o modelo (só para termos uma ideia de qualidade)
y_pred = modelo.predict(X_test)
print("Acurácia no conjunto de teste:", accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))

# 7. Salvar o modelo treinado em um arquivo .pkl
#    Esse é o arquivo que o app.py vai carregar depois.
joblib.dump({"modelo": modelo, "features": FEATURES}, "modelo_provisorio.pkl")
print("\nModelo salvo em modelo_provisorio.pkl")
