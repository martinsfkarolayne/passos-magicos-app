# App Streamlit - Passos Mágicos (Datathon Fase 5)

Aplicativo que carrega o modelo preditivo do grupo e estima a probabilidade
de um aluno da Passos Mágicos entrar em risco no ciclo atual
(`Em_Risco_Vigente`).

## Modelo

XGBoost, treinado pela equipe (notebook "Limpeza, organização e
padronização + Modelo de Machine Learning - Pergunta 9"). Métricas no
conjunto de teste: acurácia 99,65%, recall 99,12%, precisão 100%.

Indicadores usados: IDA, IEG, IAA, IPS, IPP, IPV, Mat, Por, Gênero,
Instituição de ensino, Fase (IAN não é usado, para evitar data leakage).

Régua de risco: >= 70% risco alto, 40-69% risco moderado, < 40% sem risco.

## Arquivos

- `app.py` - aplicativo Streamlit.
- `modelo_xgb_passos_magicos.pkl` - modelo treinado + lista de colunas do treino.
- `train_model.py` - script que reproduz o treino do modelo.
- `dataset_treino.csv` - base usada no treino.
- `requirements.txt` - dependências do projeto.

## Rodando localmente

```
pip3 install -r requirements.txt
streamlit run app.py
```

## Deploy

Publicado no Streamlit Community Cloud, com `app.py` como arquivo principal.
Atualiza automaticamente a cada push no repositório.
