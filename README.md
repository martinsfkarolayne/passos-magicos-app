# App Streamlit — Passos Mágicos (Fase 4 do Datathon)

## O que tem aqui dentro

- `app.py` — o aplicativo em si (a tela que as pessoas vão ver e usar).
- `modelo_provisorio.pkl` — o modelo treinado que o app carrega (PROVISÓRIO,
  só para testarmos o processo até a Thaty terminar o modelo real).
- `requirements.txt` — lista das bibliotecas que o app precisa para rodar.
- `train_model.py` — script que treinou o modelo provisório (não precisa
  rodar de novo, só serve caso a gente precise treinar outra versão).
- `dataset_treino.csv` — a base de dados usada para treinar o modelo provisório.

## Como testar no seu computador (antes de publicar)

1. Abra o **Terminal**.
2. Entre na pasta do projeto (troque o caminho pelo lugar onde você salvou):
   ```
   cd caminho/para/passos-magicos-app
   ```
3. Instale as bibliotecas necessárias:
   ```
   pip3 install -r requirements.txt
   ```
4. Rode o aplicativo:
   ```
   streamlit run app.py
   ```
5. Uma aba deve abrir sozinha no seu navegador, com o app funcionando.
   Para parar, volte ao Terminal e aperte `Control + C`.

## Como publicar (deploy) no Streamlit Community Cloud

1. Suba essa pasta inteira para um repositório novo no GitHub.
2. Entre em share.streamlit.io, clique em **"Create app"**.
3. Escolha o repositório que você criou, e em "Main file path" escreva `app.py`.
4. Clique em **Deploy**. Em alguns minutos o app estará no ar com um link público.

## Quando a Thaty terminar o modelo real

1. Substitua o arquivo `modelo_provisorio.pkl` pelo arquivo do modelo dela
   (pode manter o mesmo nome de arquivo, ou trocar o nome no `app.py`,
   na linha `joblib.load("modelo_provisorio.pkl")`).
2. Confira quais colunas (indicadores) o modelo dela espera receber — se
   forem diferentes das usadas aqui, ajuste a lista `FEATURES` e os campos
   do formulário em `app.py`.
3. Teste local de novo (`streamlit run app.py`) antes de subir a
   atualização para o GitHub (o Streamlit Cloud atualiza o app publicado
   sozinho sempre que você atualiza o repositório).
