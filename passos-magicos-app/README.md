# App Streamlit - Passos Mágicos (Datathon Fase 5)

App que carrega o modelo preditivo e estima a probabilidade de um aluno da Passos Mágicos entrar em risco no ciclo atual (`Em_Risco_Vigente`).

Modelo XGBoost treinado pela equipe (notebook "Limpeza, organização e padronização + Modelo de Machine Learning - Pergunta 9"). No conjunto de teste: acurácia 99,65%, recall 99,12%, precisão 100%.

Indicadores usados: IDA, IEG, IAA, IPS, IPP, IPV, Mat, Por, Gênero, Instituição de ensino e Fase. O IAN não entra no modelo pra evitar data leakage.

Régua de risco: >= 70% risco alto, 40-69% risco moderado, < 40% sem risco.

Arquivos da pasta:
- `app.py` - aplicativo Streamlit
- `modelo_xgb_passos_magicos.pkl` - modelo treinado + colunas do treino
- `train_model.py` - script que reproduz o treino do modelo
- `dataset_treino.csv` - base usada no treino
- `requirements.txt` - dependências

Pra rodar local:
```
pip3 install -r requirements.txt
streamlit run app.py
```

Publicado no Streamlit Community Cloud, atualiza automaticamente a cada push nesse repositório.
