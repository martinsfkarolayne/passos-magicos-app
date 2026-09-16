# App Streamlit — Passos Mágicos (Datathon Fase 5)

Aplicativo da Fase 4 do Datathon: carrega o modelo preditivo do grupo e estima
a probabilidade de um aluno da Passos Mágicos entrar em risco de defasagem.

## Arquivos

- `app.py` — aplicativo Streamlit.
- `modelo_provisorio.pkl` — modelo atualmente em uso (treinado com a base da
  Fase 1/2, enquanto o modelo definitivo da Fase 3 não fica pronto).
- `train_model.py` — script de treino do modelo.
- `dataset_treino.csv` — base usada no treino.
- `requirements.txt` — dependências do projeto.

## Rodando localmente

```
pip3 install -r requirements.txt
streamlit run app.py
```

## Deploy

Publicado no Streamlit Community Cloud, com `app.py` como arquivo principal.

## Próximos passos

Substituir `modelo_provisorio.pkl` pelo modelo final da Fase 3 e ajustar as
variáveis de entrada em `app.py` caso as colunas usadas sejam diferentes.
